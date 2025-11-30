import os
import uuid
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

# ============================================================
# Config OpenAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquant dans les variables d'environnement")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Config Cloudflare R2 (S3 compatible)
# ============================================================

R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ENDPOINT_URL = (
    os.getenv("R2_ENDPOINT_URL")
    or os.getenv("R2_ENDPOINT")
    or os.getenv("R2_S3_ENDPOINT")
)
R2_REGION = os.getenv("R2_REGION") or "auto"

if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT_URL]):
    raise RuntimeError("Config R2 incomplète (ACCESS_KEY / SECRET / BUCKET / ENDPOINT)")

s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name=R2_REGION,
    config=Config(signature_version="s3v4"),
)

# ============================================================
# Modèles & stockage en mémoire
# ============================================================


class ProjectCreate(BaseModel):
    name: Optional[str] = "Unnamed project"


class Project(BaseModel):
    id: str
    name: str
    created_at: datetime


# Projets et données associées stockées en mémoire
PROJECTS: Dict[str, Project] = {}
PROJECT_DATA: Dict[str, Dict] = {}  # fichiers, report_markdown, svg, clés R2, etc.

# ============================================================
# Helpers R2
# ============================================================


def r2_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def r2_get_bytes(key: str) -> bytes:
    try:
        obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"File not found in storage: {key}")
    return obj["Body"].read()


# ============================================================
# Helpers exports (PDF / DOCX)
# ============================================================

from docx import Document  # python-docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def markdown_to_docx_bytes(markdown_text: str) -> bytes:
    """Conversion ultra simple : 1 paragraphe par ligne de markdown."""
    doc = Document()
    for line in markdown_text.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    """PDF très simple avec ReportLab (texte brut, wrap approx.)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x_margin = 72
    y = height - 72
    max_chars = 90

    for line in markdown_text.splitlines():
        # Wrap très basique
        while len(line) > max_chars:
            chunk = line[:max_chars]
            line = line[max_chars:]
            c.drawString(x_margin, y, chunk)
            y -= 14
            if y < 72:
                c.showPage()
                y = height - 72
        c.drawString(x_margin, y, line)
        y -= 14
        if y < 72:
            c.showPage()
            y = height - 72

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Fallback PNG (si OpenAI Images indisponible)
# ============================================================

# 1x1 px PNG blanc (base64)
HERO_PLACEHOLDER_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Yp3GxkAAAAASUVORK5CYII="
)


def get_placeholder_png_bytes() -> bytes:
    return base64.b64decode(HERO_PLACEHOLDER_B64)


# ============================================================
# App FastAPI
# ============================================================

app = FastAPI(
    title="Archito-Genie Backend",
    description="MVP backend avec stockage Cloudflare R2",
    version="0.2.1",
)

# ============================================================
# Routes Projet
# ============================================================


@app.post("/projects", response_model=Project)
def create_project(payload: ProjectCreate):
    project_id = str(uuid.uuid4())
    project = Project(
        id=project_id,
        name=payload.name or "Unnamed project",
        created_at=datetime.utcnow(),
    )
    PROJECTS[project_id] = project
    PROJECT_DATA[project_id] = {
        "files": {},
        "report_markdown": "",
        "schematics_svg": "",
        "hero_key": None,
        "report_pdf_key": None,
        "report_docx_key": None,
    }
    return project


@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: str):
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


# ============================================================
# Upload des fichiers
# ============================================================


@app.post("/projects/{project_id}/files")
async def upload_project_files(
    project_id: str,
    architectural_plan: UploadFile = File(...),
    soil_report: UploadFile = File(None),
    additional_files: List[UploadFile] = File(default=[]),
):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    base_prefix = f"projects/{project_id}/input"

    # Plan archi
    arch_bytes = await architectural_plan.read()
    arch_key = f"{base_prefix}/architectural_plan_{architectural_plan.filename}"
    r2_put_bytes(
        arch_key,
        arch_bytes,
        architectural_plan.content_type or "application/pdf",
    )

    # Étude de sol (optionnel)
    soil_key = None
    if soil_report is not None:
        soil_bytes = await soil_report.read()
        soil_key = f"{base_prefix}/soil_report_{soil_report.filename}"
        r2_put_bytes(soil_key, soil_bytes, soil_report.content_type or "application/pdf")

    # Fichiers additionnels (optionnels)
    additional_keys: List[str] = []
    for i, f in enumerate(additional_files or []):
        content = await f.read()
        k = f"{base_prefix}/additional_{i}_{f.filename}"
        r2_put_bytes(k, content, f.content_type or "application/octet-stream")
        additional_keys.append(k)

    PROJECT_DATA[project_id]["files"] = {
        "architectural_plan_key": arch_key,
        "soil_report_key": soil_key,
        "additional_files_keys": additional_keys,
    }

    return {
        "project_id": project_id,
        "architectural_plan_key": arch_key,
        "soil_report_key": soil_key,
        "additional_files_keys": additional_keys,
    }


# ============================================================
# Analyse & rapport (MVP simple)
# ============================================================


@app.post("/projects/{project_id}/analyze")
def analyze_project(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS[project_id]
    files = PROJECT_DATA[project_id].get("files") or {}
    arch_key = files.get("architectural_plan_key")

    if not arch_key:
        raise HTTPException(
            status_code=400,
            detail="Architectural plan must be uploaded before analysis",
        )

    prompt = f"""
Tu es un assistant d'ingénierie pour un SaaS nommé Archito-Genie.

On a reçu un projet immobilier nommé "{project.name}".
Les fichiers sont stockés dans un bucket S3/R2.

Produit un rapport en **Markdown** avec les sections suivantes :

1. Description rapide du projet
2. Bloc Structure (3 à 5 puces très concrètes)
3. Bloc MEPF & Automation (3 à 5 puces)
4. Bloc Durabilité & Efficacité énergétique (3 à 5 puces)
5. Risques & points de vigilance (liste courte)

Tu n'as pas accès aux plans en détail, donc reste à un niveau
générique mais crédible pour un projet résidentiel ou tertiaire moyen.
"""
    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Tu es un assistant d'ingénierie bâtiment."},
            {"role": "user", "content": prompt},
        ],
    )

    report_md = completion.choices[0].message.content or "# Rapport Archito-Genie"
    PROJECT_DATA[project_id]["report_markdown"] = report_md

    # Génération d'un SVG très simple avec 3 blocs
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300">
  <style>
    .title {{ font: bold 20px sans-serif; }}
    .label {{ font: 14px sans-serif; }}
  </style>
  <rect x="20" y="60" width="260" height="180" rx="12" ry="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="150" y="110" text-anchor="middle" class="label">Structure</text>

  <rect x="320" y="60" width="260" height="180" rx="12" ry="12" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="450" y="110" text-anchor="middle" class="label">MEPF & Automation</text>

  <rect x="620" y="60" width="260" height="180" rx="12" ry="12" fill="#fff3e0" stroke="#ef6c00" stroke-width="2"/>
  <text x="750" y="110" text-anchor="middle" class="label">Durabilité</text>

  <text x="450" y="30" text-anchor="middle" class="title">{project.name}</text>
</svg>"""

    PROJECT_DATA[project_id]["schematics_svg"] = svg

    return {"project_id": project_id, "status": "analyzed"}


@app.get("/projects/{project_id}/report")
def get_report(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    md = PROJECT_DATA[project_id].get("report_markdown") or ""
    return {"project_id": project_id, "report_markdown": md}


# ============================================================
# SVG & Hero PNG
# ============================================================


@app.get("/projects/{project_id}/schematics/svg")
def get_schematics_svg(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("schematics_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="Schematics not generated yet")

    stream = io.BytesIO(svg.encode("utf-8"))
    filename = f"{project_id}_schematics.svg"

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/projects/{project_id}/schematics/hero")
async def generate_hero_image(project_id: str):
    """
    Génère une image 'hero' (PNG) pour le pitch deck.
    - Tente d'abord OpenAI Images (gpt-image-1).
    - Si ça échoue (403 / quota / autre), renvoie un PNG placeholder.
    """
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    report_md = PROJECT_DATA[project_id].get("report_markdown") or ""

    # Contexte pour l'image (très simple pour le moment)
    context_text = (
        report_md[:800]
        if report_md
        else "Modern mixed-use building with clean lines, large glazing, and elegant volumetry."
    )

    prompt = (
        "Architectural hero image for an investor pitch deck.\n"
        f"Project name: {project.name}.\n"
        f"{context_text}\n"
        "Style: cinematic 3D render, 16:9 horizontal composition, realistic lighting, "
        "soft daylight, no people, no text, no logo, no title block."
    )

    image_bytes: bytes

    try:
        img = openai_client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1536x1024",
            n=1,
            response_format="b64_json",
        )

        if not img.data or img.data[0].b64_json is None:
            # OpenAI a répondu mais sans image exploitable
            print("Image generation returned no data, using placeholder.")
            image_bytes = get_placeholder_png_bytes()
        else:
            image_b64 = img.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)

    except Exception as e:
        # Ici on LOG l'erreur, mais on ne casse plus l'API :
        # on renvoie un PNG placeholder "propre".
        print(f"Hero image generation failed, using placeholder instead: {e}")
        image_bytes = get_placeholder_png_bytes()

    # Réponse PNG (OpenAI ou placeholder)
    stream = io.BytesIO()
    stream.write(image_bytes)
    stream.seek(0)

    filename = f"{project_id}_hero.png"

    return StreamingResponse(
        stream,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Export PDF & DOCX
# ============================================================


@app.get("/projects/{project_id}/export/pdf")
def export_report_pdf(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    report_md = PROJECT_DATA[project_id].get("report_markdown")
    if not report_md:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    pdf_bytes = markdown_to_pdf_bytes(report_md)
    key = f"projects/{project_id}/output/report.pdf"
    r2_put_bytes(key, pdf_bytes, "application/pdf")
    PROJECT_DATA[project_id]["report_pdf_key"] = key

    stream = io.BytesIO(pdf_bytes)
    filename = f"{project_id}_report.pdf"

    return StreamingResponse(
        stream,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/projects/{project_id}/export/docx")
def export_report_docx(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    report_md = PROJECT_DATA[project_id].get("report_markdown")
    if not report_md:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    docx_bytes = markdown_to_docx_bytes(report_md)
    key = f"projects/{project_id}/output/report.docx"
    r2_put_bytes(
        key,
        docx_bytes,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    PROJECT_DATA[project_id]["report_docx_key"] = key

    stream = io.BytesIO(docx_bytes)
    filename = f"{project_id}_report.docx"

    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Healthcheck très simple pour Render
# ============================================================


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Archito-Genie backend is live"}

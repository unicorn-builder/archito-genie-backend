import os
import uuid
import base64
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

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
PROJECT_DATA: Dict[str, Dict] = {}  # fichiers, report_markdown, SVG, plan_spec, etc.

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
    description="MVP backend avec stockage Cloudflare R2 + génération de plans STRUCTURE/MEP",
    version="0.3.0",
)

# ============================================================
# Rendu SVG STRUCTURE & MEP (moteur interne)
# ============================================================

import math
import svgwrite


def _layout_rooms(rooms: List[Dict[str, Any]], scale: float = 40.0):
    """
    Dispose les pièces en grille pour le MVP.
    rooms: liste de dicts avec au moins name, width_m, length_m, is_wet_area.
    Retourne: (layout_positions, total_width_m, total_height_m)
    """
    max_rooms_per_row = max(1, math.ceil(math.sqrt(len(rooms)))) if rooms else 1

    current_x = 0.0
    current_y = 0.0
    row_height_m = 0.0
    layout_positions = []

    for idx, room in enumerate(rooms):
        w = float(room.get("width_m", 3.0))
        l = float(room.get("length_m", 3.0))

        if idx % max_rooms_per_row == 0 and idx != 0:
            # nouvelle ligne
            current_x = 0.0
            current_y += row_height_m
            row_height_m = 0.0

        layout_positions.append(
            {
                "room": room,
                "x_m": current_x,
                "y_m": current_y,
                "w_m": w,
                "l_m": l,
            }
        )
        current_x += w
        row_height_m = max(row_height_m, l)

    total_width_m = max((p["x_m"] + p["w_m"]) for p in layout_positions) if layout_positions else 8.0
    total_height_m = max((p["y_m"] + p["l_m"]) for p in layout_positions) if layout_positions else 8.0

    return layout_positions, total_width_m, total_height_m


def render_structure_svg(spec: Dict[str, Any]) -> str:
    """
    Génère un plan STRUCTURE simplifié en SVG (niveau BE “light”).
    Retourne une chaîne SVG (UTF-8).
    """
    scale = 40.0  # pixels par mètre
    margin = 60

    floors = spec.get("floors", [])
    floor = floors[0] if floors else {"name": "Niveau 0", "rooms": []}
    rooms = floor.get("rooms", [])

    layout_positions, total_w_m, total_h_m = _layout_rooms(rooms, scale=scale)
    width_px = int(total_w_m * scale + 2 * margin)
    height_px = int(total_h_m * scale + 2 * margin)

    dwg = svgwrite.Drawing(size=(width_px, height_px))
    dwg.add(dwg.rect(insert=(0, 0), size=(width_px, height_px), fill="white"))

    # Titre
    dwg.add(
        dwg.text(
            f"PLAN STRUCTUREL - {floor.get('name', 'Niveau 0')} - ARCHITO-GENIE",
            insert=(margin, margin - 25),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    # Pièces → murs “porteurs” simplifiés + poteaux
    for p in layout_positions:
        x = margin + p["x_m"] * scale
        y = margin + p["y_m"] * scale
        w = p["w_m"] * scale
        h = p["l_m"] * scale
        room = p["room"]

        # Contour = murs porteurs (épais)
        dwg.add(
            dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill="none",
                stroke="black",
                stroke_width=3,
            )
        )

        # Nom de la pièce
        dwg.add(
            dwg.text(
                room.get("name", "Pièce"),
                insert=(x + w / 2, y + h / 2),
                text_anchor="middle",
                alignment_baseline="middle",
                font_size="11px",
                font_family="Arial",
            )
        )

        # Poteaux simplifiés aux 4 coins
        col_size = 12
        for dx, dy in [(0, 0), (w - col_size, 0), (0, h - col_size), (w - col_size, h - col_size)]:
            dwg.add(
                dwg.rect(
                    insert=(x + dx, y + dy),
                    size=(col_size, col_size),
                    fill="black",
                )
            )

    # Trame horizontale indicative
    dwg.add(
        dwg.text(
            "Trame indicative (m)",
            insert=(margin, height_px - 40),
            font_size="10px",
            font_family="Arial",
        )
    )
    start_x = margin
    end_x = margin + total_w_m * scale
    base_y = height_px - 20
    dwg.add(dwg.line(start=(start_x, base_y), end=(end_x, base_y), stroke="#bbbbbb", stroke_width=1))
    step_m = 2.0
    x_m = 0.0
    while x_m <= total_w_m:
        x_px = margin + x_m * scale
        dwg.add(dwg.line(start=(x_px, base_y - 5), end=(x_px, base_y + 5), stroke="black", stroke_width=1))
        dwg.add(
            dwg.text(
                f"{x_m:.0f}",
                insert=(x_px + 2, base_y - 7),
                font_size="8px",
                font_family="Arial",
            )
        )
        x_m += step_m

    return dwg.tostring()


def render_mep_svg(spec: Dict[str, Any]) -> str:
    """
    Génère un plan MEP combiné (plomberie + électricité + CVC) en SVG.
    Utilise is_wet_area, has_window et le nom des pièces pour déduire les réseaux.
    Retourne une chaîne SVG.
    """
    scale = 40.0
    margin = 60

    floors = spec.get("floors", [])
    floor = floors[0] if floors else {"name": "Niveau 0", "rooms": []}
    rooms = floor.get("rooms", [])

    layout_positions, total_w_m, total_h_m = _layout_rooms(rooms, scale=scale)
    width_px = int(total_w_m * scale + 2 * margin)
    height_px = int(total_h_m * scale + 2 * margin)

    dwg = svgwrite.Drawing(size=(width_px, height_px))
    dwg.add(dwg.rect(insert=(0, 0), size=(width_px, height_px), fill="white"))

    # Titre
    dwg.add(
        dwg.text(
            f"PLAN MEP - {floor.get('name', 'Niveau 0')} - ARCHITO-GENIE",
            insert=(margin, margin - 25),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    # Contours de pièces
    for p in layout_positions:
        x = margin + p["x_m"] * scale
        y = margin + p["y_m"] * scale
        w = p["w_m"] * scale
        h = p["l_m"] * scale
        room = p["room"]

        dwg.add(
            dwg.rect(
                insert=(x, y),
                size=(w, h),
                fill="none",
                stroke="#aaaaaa",
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                room.get("name", "Pièce"),
                insert=(x + w / 2, y + h / 2),
                text_anchor="middle",
                alignment_baseline="middle",
                font_size="11px",
                font_family="Arial",
                fill="#444444",
            )
        )

    # Noyau technique fictif (colonne EU/EV + tableau elec)
    core_x = margin + total_w_m * scale + 20
    core_y = margin
    core_w = 80
    core_h = 200

    dwg.add(
        dwg.rect(
            insert=(core_x, core_y),
            size=(core_w, core_h),
            fill="none",
            stroke="#666666",
            stroke_width=3,
        )
    )
    dwg.add(
        dwg.text(
            "NOYAU\nTECHNIQUE",
            insert=(core_x + core_w / 2, core_y + core_h / 2),
            text_anchor="middle",
            alignment_baseline="middle",
            font_size="10px",
            font_family="Arial",
            fill="#444444",
        )
    )

    # Colonne EU/EV
    dwg.add(
        dwg.line(
            start=(core_x + core_w / 2, core_y + 20),
            end=(core_x + core_w / 2, core_y + core_h - 20),
            stroke="#0044aa",
            stroke_width=4,
        )
    )
    dwg.add(
        dwg.text(
            "COL. EU/EV",
            insert=(core_x + core_w / 2 + 5, core_y + 35),
            font_size="9px",
            font_family="Arial",
            fill="#0044aa",
        )
    )

    # Parcours des pièces : plomberie / elec / CVC
    for p in layout_positions:
        x = margin + p["x_m"] * scale
        y = margin + p["y_m"] * scale
        w = p["w_m"] * scale
        h = p["l_m"] * scale
        room = p["room"]

        name = (room.get("name") or "").lower()
        is_wet = room.get("is_wet_area", False)

        # Zones d'eau (cuisine, salle de bain, etc.)
        if is_wet or any(k in name for k in ["bain", "sdb", "wc", "toilet", "cuisine", "kitchen"]):
            # Point d'eau
            dwg.add(
                dwg.circle(
                    center=(x + w * 0.2, y + h * 0.2),
                    r=7,
                    fill="none",
                    stroke="#0077ff",
                    stroke_width=2,
                )
            )
            dwg.add(
                dwg.text(
                    "PE",
                    insert=(x + w * 0.2 + 10, y + h * 0.2 + 3),
                    font_size="9px",
                    font_family="Arial",
                    fill="#0077ff",
                )
            )
            # Evacuation
            dwg.add(
                dwg.polygon(
                    points=[
                        (x + w * 0.25, y + h * 0.25),
                        (x + w * 0.27, y + h * 0.25),
                        (x + w * 0.26, y + h * 0.23),
                    ],
                    fill="none",
                    stroke="#0044aa",
                    stroke_width=2,
                )
            )
            # Raccord EV vers colonne
            dwg.add(
                dwg.polyline(
                    points=[
                        (x + w * 0.26, y + h * 0.25),
                        (x + w, y + h * 0.25),
                        (core_x, core_y + core_h * 0.7),
                    ],
                    fill="none",
                    stroke="#0044aa",
                    stroke_width=1.5,
                    stroke_dasharray="4 2",
                )
            )
            # Raccord EF/EC vers noyau
            dwg.add(
                dwg.polyline(
                    points=[
                        (x + w * 0.2, y + h * 0.2),
                        (x + w * 0.2, y),
                        (core_x, core_y + core_h * 0.5),
                    ],
                    fill="none",
                    stroke="#00aaff",
                    stroke_width=1.5,
                    stroke_dasharray="6 3",
                )
            )

        # Electricité : prise
        dwg.add(
            dwg.rect(
                insert=(x + 5, y + 5),
                size=(10, 10),
                fill="none",
                stroke="#ff3333",
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                "PR",
                insert=(x + 20, y + 14),
                font_size="8px",
                font_family="Arial",
                fill="#ff3333",
            )
        )

        # Electricité : luminaire au centre
        lum_x = x + w / 2
        lum_y = y + h / 2
        dwg.add(
            dwg.line(
                start=(lum_x - 6, lum_y),
                end=(lum_x + 6, lum_y),
                stroke="#ff8800",
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.line(
                start=(lum_x, lum_y - 6),
                end=(lum_x, lum_y + 6),
                stroke="#ff8800",
                stroke_width=2,
            )
        )

        # CVC : diffuseur linéaire dans séjour/chambre
        if any(k in name for k in ["séjour", "sejour", "salon", "living", "chambre", "bed"]):
            dwg.add(
                dwg.line(
                    start=(x + w * 0.1, y + 10),
                    end=(x + w * 0.9, y + 10),
                    stroke="#22aa22",
                    stroke_width=3,
                )
            )

    # Tableau électrique dans noyau
    dwg.add(
        dwg.rect(
            insert=(core_x + 10, core_y + 20),
            size=(40, 25),
            fill="none",
            stroke="#ff3333",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "TGBT",
            insert=(core_x + 30, core_y + 37),
            text_anchor="middle",
            font_size="9px",
            font_family="Arial",
            fill="#ff3333",
        )
    )

    # Légende
    legend_x = margin
    legend_y = height_px - 110
    legend_w = 320
    legend_h = 90

    dwg.add(
        dwg.rect(
            insert=(legend_x, legend_y),
            size=(legend_w, legend_h),
            fill="none",
            stroke="#999999",
            stroke_width=1,
        )
    )
    dwg.add(
        dwg.text(
            "LÉGENDE MEP (MVP)",
            insert=(legend_x + 10, legend_y + 18),
            font_size="12px",
            font_family="Arial",
            font_weight="bold",
        )
    )

    # Plomberie
    dwg.add(
        dwg.circle(
            center=(legend_x + 15, legend_y + 35),
            r=5,
            fill="none",
            stroke="#0077ff",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Point d'eau EF/EC",
            insert=(legend_x + 30, legend_y + 39),
            font_size="10px",
            font_family="Arial",
        )
    )

    dwg.add(
        dwg.polygon(
            points=[
                (legend_x + 12, legend_y + 55),
                (legend_x + 22, legend_y + 55),
                (legend_x + 17, legend_y + 45),
            ],
            fill="none",
            stroke="#0044aa",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Évacuation EU/EV",
            insert=(legend_x + 30, legend_y + 59),
            font_size="10px",
            font_family="Arial",
        )
    )

    # Electricité
    dwg.add(
        dwg.rect(
            insert=(legend_x + 10, legend_y + 70),
            size=(10, 10),
            fill="none",
            stroke="#ff3333",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Prise électrique",
            insert=(legend_x + 30, legend_y + 79),
            font_size="10px",
            font_family="Arial",
        )
    )

    # CVC
    dwg.add(
        dwg.line(
            start=(legend_x + 10, legend_y + 90),
            end=(legend_x + 40, legend_y + 90),
            stroke="#22aa22",
            stroke_width=3,
        )
    )
    dwg.add(
        dwg.text(
            "Diffuseur linéaire CVC",
            insert=(legend_x + 50, legend_y + 94),
            font_size="10px",
            font_family="Arial",
        )
    )

    return dwg.tostring()


# ============================================================
# Extraction plan_spec via OpenAI
# ============================================================


def get_plan_spec_from_ai(file_name: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Utilise OpenAI pour produire une description JSON du bâtiment (plan_spec).
    Pour le MVP, on ne lit pas encore le contenu PDF, on se base sur le contexte et la taille.
    """
    approx_size_kb = max(len(file_bytes) // 1024, 1)
    user_content = f"""
Tu es un ingénieur/architecte assistant pour une application SaaS nommée Archito-Genie.

On a reçu un plan architectural nommé "{file_name}", taille approximative {approx_size_kb} Ko.

TA TÂCHE :
- Construire une hypothèse réaliste de bâtiment basée sur ce que tu sais des immeubles
  résidentiels/mixtes en Afrique de l'Ouest (ou globalement, si tu n'as pas de contexte).
- Renvoie STRICTEMENT un JSON avec le format suivant :

{{
  "building_type": "residential" | "office" | "mixed",
  "floors": [
    {{
      "name": "RDC",
      "level": 0,
      "rooms": [
        {{
          "name": "Séjour",
          "width_m": 5.0,
          "length_m": 6.0,
          "has_window": true,
          "is_wet_area": false
        }}
      ]
    }}
  ]
}}

CONTRAINTES :
- ENTRE 5 ET 15 pièces par niveau.
- dimensions crédibles (2m à 8m par côté pour les pièces standard).
- Si tu mets plusieurs niveaux, adapte légèrement la distribution.
- Tu NE RENVOIES QUE du JSON (pas de texte avant/après).
    """.strip()

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant d'ingénierie bâtiment. "
                        "Tu produis une description JSON cohérente de la distribution des pièces."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        )
        content = completion.choices[0].message.content
        spec = json.loads(content)
        return spec
    except Exception as e:
        print(f"[WARN] get_plan_spec_from_ai failed, using fallback spec: {e}")
        # Fallback minimal
        return {
            "building_type": "residential",
            "floors": [
                {
                    "name": "RDC",
                    "level": 0,
                    "rooms": [
                        {
                            "name": "Séjour",
                            "width_m": 5.0,
                            "length_m": 6.0,
                            "has_window": True,
                            "is_wet_area": False,
                        },
                        {
                            "name": "Cuisine",
                            "width_m": 3.0,
                            "length_m": 4.0,
                            "has_window": True,
                            "is_wet_area": True,
                        },
                        {
                            "name": "Chambre 1",
                            "width_m": 4.0,
                            "length_m": 4.0,
                            "has_window": True,
                            "is_wet_area": False,
                        },
                        {
                            "name": "Chambre 2",
                            "width_m": 3.5,
                            "length_m": 3.5,
                            "has_window": True,
                            "is_wet_area": False,
                        },
                        {
                            "name": "Salle de bain",
                            "width_m": 3.0,
                            "length_m": 3.0,
                            "has_window": False,
                            "is_wet_area": True,
                        },
                    ],
                }
            ],
        }


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
        "plan_spec": None,
        "structure_svg": "",
        "mep_svg": "",
        "hero_key": None,         # plus utilisé pour l'instant
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
# Analyse & génération de plans STRUCTURE/MEP
# ============================================================


@app.post("/projects/{project_id}/analyze")
def analyze_project(project_id: str):
    """
    Nouvelle version :
    - Vérifie la présence du plan archi
    - Appelle OpenAI pour générer un plan_spec (JSON)
    - Génère les SVG STRUCTURE & MEP
    - Génère un rapport Markdown (comme avant) pour le storytelling
    """
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

    # 1) Récupérer le plan archi depuis R2
    arch_bytes = r2_get_bytes(arch_key)

    # 2) OpenAI → plan_spec JSON
    plan_spec = get_plan_spec_from_ai(file_name=arch_key, file_bytes=arch_bytes)
    PROJECT_DATA[project_id]["plan_spec"] = plan_spec

    # 3) Génération des SVG STRUCTURE & MEP
    structure_svg = render_structure_svg(plan_spec)
    mep_svg = render_mep_svg(plan_spec)

    PROJECT_DATA[project_id]["structure_svg"] = structure_svg
    PROJECT_DATA[project_id]["mep_svg"] = mep_svg

    # 4) Rapport Markdown (garde ton ancienne logique, mais enrichie)
    prompt = f"""
Tu es un assistant d'ingénierie pour un SaaS nommé Archito-Genie.

On a reçu un projet immobilier nommé "{project.name}".
Voici une description synthétique JSON de la distribution du bâtiment :

{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

Produit un rapport en **Markdown** avec les sections suivantes :

1. Description rapide du projet (basée sur le JSON)
2. Bloc Structure (3 à 7 puces très concrètes)
3. Bloc MEPF & Automation (3 à 7 puces)
4. Bloc Durabilité & Efficacité énergétique (3 à 7 puces)
5. Risques & points de vigilance (liste courte)
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

    return {
        "project_id": project_id,
        "status": "analyzed",
        "has_plan_spec": bool(plan_spec),
        "has_structure_svg": bool(structure_svg),
        "has_mep_svg": bool(mep_svg),
    }


@app.get("/projects/{project_id}/report")
def get_report(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    md = PROJECT_DATA[project_id].get("report_markdown") or ""
    return {"project_id": project_id, "report_markdown": md}


# ============================================================
# Plans STRUCTURE & MEP en SVG (téléchargement)
# ============================================================


@app.get("/projects/{project_id}/plans/structure.svg")
def get_structure_svg(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("structure_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="Structure plan not generated yet. Call /analyze first.")

    stream = io.BytesIO(svg.encode("utf-8"))
    filename = f"{project_id}_plan_structure.svg"

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/projects/{project_id}/plans/mep.svg")
def get_mep_svg(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("mep_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="MEP plan not generated yet. Call /analyze first.")

    stream = io.BytesIO(svg.encode("utf-8"))
    filename = f"{project_id}_plan_mep.svg"

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Compat : /schematics/svg renvoie le plan STRUCTURE
# (ancien endpoint réutilisé pour ne pas casser Lovable)
# ============================================================


@app.get("/projects/{project_id}/schematics/svg")
def get_schematics_svg(project_id: str):
    """
    Ancien endpoint /schematics/svg.
    Désormais, il renvoie le plan STRUCTURE en SVG
    pour garder la compatibilité avec ton front Lovable.
    """
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("structure_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="Structure plan not generated yet. Call /analyze first.")

    stream = io.BytesIO(svg.encode("utf-8"))
    filename = f"{project_id}_plan_structure.svg"

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Hero PNG – DÉSACTIVÉ (tu avais demandé de le désactiver)
# ============================================================
# L'endpoint /schematics/hero est volontairement supprimé.
# Si ton front l'appelle encore, il recevra un 404.


# ============================================================
# Export PDF & DOCX (rapport)
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
def expo

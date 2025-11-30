import os
import base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict
from typing import Optional, List
from openai import OpenAI

# ============================================================================
# INITIALISATION
# ============================================================================

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECTS: Dict[str, dict] = {}
ENGINEERING_RESULTS: Dict[str, dict] = {}

# ============================================================================
# MODELES
# ============================================================================

class CreateProjectRequest(BaseModel):
    name: str

class AnalyzeResult(BaseModel):
    structure_summary: str
    mepf_summary: str
    sustainability_summary: str

# ============================================================================
# ROUTES PROJET
# ============================================================================

@app.post("/projects")
def create_project(req: CreateProjectRequest):
    project_id = os.urandom(8).hex()

    PROJECTS[project_id] = {
        "name": req.name,
        "files": [],
    }

    return {"project_id": project_id}


@app.post("/projects/{project_id}/files")
async def upload_project_files(
    project_id: str,
    architectural_plan: UploadFile = File(...),
    soil_report: Optional[UploadFile] = File(None),
    additional_files: Optional[List[UploadFile]] = File(None),
):
    ...

    if project_id not in PROJECTS:
        raise HTTPException(404, "Unknown project_id")

    content = await file.read()
    PROJECTS[project_id]["files"].append({"filename": file.filename, "content": content})

    return {"status": "file uploaded"}


@app.post("/projects/{project_id}/analyze")
async def analyze_project(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(404, "Unknown project_id")

    # Analyse simplifiée (tu peux la remplacer)
    result = {
        "structure_summary": "Structure OK",
        "mepf_summary": "MEPF OK",
        "sustainability_summary": "Sustainability OK",
        "report_markdown": "# Rapport d’ingénierie\nTout est OK.",
    }

    ENGINEERING_RESULTS[project_id] = result
    return {"status": "analysis completed"}


@app.get("/projects/{project_id}/report")
def get_report(project_id: str):
    if project_id not in ENGINEERING_RESULTS:
        raise HTTPException(404, "No analysis found for project_id")

    return {"report_markdown": ENGINEERING_RESULTS[project_id]["report_markdown"]}

# ============================================================================
# SVG (déjà fonctionnel)
# ============================================================================

@app.get("/projects/{project_id}/schematics/svg")
def get_svg(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(404, "Unknown project_id")

    svg_content = """
    <svg width="600" height="300" xmlns="http://www.w3.org/2000/svg">
      <rect x="10" y="10" width="580" height="280" fill="#f0f0f0" stroke="#333" stroke-width="2"/>
      <text x="50%" y="50%" font-size="24" text-anchor="middle">Schematic for project {}</text>
    </svg>
    """.format(project_id)

    stream = BytesIO(svg_content.encode("utf-8"))
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f"attachment; filename='{project_id}.svg'"}
    )

# ============================================================================
# HERO IMAGE — VERSION FINALE (GPT-IMAGE-1)
# ============================================================================

@app.get("/projects/{project_id}/schematics/hero")
async def generate_hero_image(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(404, "Unknown project_id")

    project_name = PROJECTS[project_id]["name"]
    eng = ENGINEERING_RESULTS.get(project_id, {})

    structure = eng.get("structure_summary", "No structure result")
    mepf = eng.get("mepf_summary", "No MEPF result")
    sustainability = eng.get("sustainability_summary", "No sustainability result")

    prompt = f"""
    Create a clean architectural hero image for the project '{project_name}'.

    Structure: {structure}
    MEPF: {mepf}
    Sustainability: {sustainability}

    The image must look like a modern blueprint hybrid with light shadows,
    soft 3D hints, and minimalistic engineering aesthetics.
    """

    try:
        ai_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1536x1024",
            response_format="b64_json"
        )
    except Exception as e:
        raise HTTPException(500, f"OpenAI Image API error: {e}")

    img_b64 = ai_resp.data[0].b64_json
    img_bytes = base64.b64decode(img_b64)

    stream = BytesIO(img_bytes)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename='{project_id}_hero.png'"}
    )

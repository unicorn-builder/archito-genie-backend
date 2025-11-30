import os
import base64
from io import BytesIO
from typing import Dict, Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# In-memory storage (to replace with DB later)
PROJECTS: Dict[str, dict] = {}
ENGINEERING_RESULTS: Dict[str, dict] = {}
REPORTS: Dict[str, dict] = {}


# ---------------------------
# MODELS
# ---------------------------

class Project(BaseModel):
    project_id: str


class EngineeringResult(BaseModel):
    sustainability_summary: str
    materials_list: List[str]
    compliance: str
    recommendations: str


# ---------------------------
# ROUTES
# ---------------------------

@app.post("/projects")
async def create_project():
    """Create a new project and return its ID"""
    import uuid
    pid = str(uuid.uuid4())
    PROJECTS[pid] = {"files": {}}
    return {"project_id": pid}


@app.post("/projects/{project_id}/files")
async def upload_project_files(
    project_id: str,
    architectural_plan: UploadFile = File(...),
    soil_report: Optional[UploadFile] = File(None),
    additional_files: Optional[List[UploadFile]] = File(None)
):
    """Upload files for a project"""

    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    # architectural plan
    ap_content = await architectural_plan.read()
    PROJECTS[project_id]["files"]["architectural_plan"] = ap_content

    # soil report (optional)
    if soil_report:
        sr_content = await soil_report.read()
        PROJECTS[project_id]["files"]["soil_report"] = sr_content
    else:
        PROJECTS[project_id]["files"]["soil_report"] = None

    # additional files
    PROJECTS[project_id]["files"]["additional"] = []
    if additional_files:
        for f in additional_files:
            PROJECTS[project_id]["files"]["additional"].append(await f.read())

    return {"status": "Files uploaded successfully"}


@app.post("/projects/{project_id}/analyze")
async def analyze_project(project_id: str):
    """Analyze project and generate engineering insights"""

    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    # Dummy AI generation (replace later with real call)
    eng_result = EngineeringResult(
        sustainability_summary="This building is energy efficient...",
        materials_list=["Concrete", "Steel", "Glass"],
        compliance="Fully compliant with local regulations",
        recommendations="Increase natural ventilation..."
    )

    ENGINEERING_RESULTS[project_id] = eng_result.dict()
    return eng_result


@app.get("/projects/{project_id}/report")
async def get_report(project_id: str):
    """Generate and return a text report"""

    if project_id not in ENGINEERING_RESULTS:
        raise HTTPException(status_code=404, detail="No analysis found")

    result = ENGINEERING_RESULTS[project_id]
    report_text = (
        "SUSTAINABILITY SUMMARY:\n" + result["sustainability_summary"] + "\n\n" +
        "MATERIALS:\n" + ", ".join(result["materials_list"]) + "\n\n" +
        "COMPLIANCE:\n" + result["compliance"] + "\n\n" +
        "RECOMMENDATIONS:\n" + result["recommendations"]
    )

    REPORTS[project_id] = {"text": report_text}
    return {"report": report_text}


@app.get("/projects/{project_id}/schematics/svg")
async def generate_svg(project_id: str):
    """Dummy SVG generation"""
    svg_data = """
    <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="200" fill="lightblue"/>
      <text x="50" y="100" font-size="20">GENIE SVG</text>
    </svg>
    """
    return {"svg": svg_data}


@app.get("/projects/{project_id}/schematics/hero")
async def generate_hero_image(project_id: str):
    """Generate a PNG hero image using OpenAI Image API"""

    prompt = "A clean architectural hero image, minimalist blueprint style"
    result = client.images.generate(
        prompt=prompt,
        size="1024x1024"
    )

    b64 = result.data[0].b64_json
    image_bytes = base64.b64decode(b64)

    return StreamingResponse(
        BytesIO(image_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=hero.png"}
    )


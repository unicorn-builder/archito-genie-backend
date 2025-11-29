"""
Archito-Genie FastAPI Backend
=============================
Deploy this on your external server (e.g., Railway, Render, AWS, etc.)

Requirements:
- Python 3.9+
- pip install fastapi uvicorn openai python-multipart pydantic

Environment variables:
- OPENAI_API_KEY: Your OpenAI API key

Run locally:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

# ===========================
# 📌 IMPORTS — ARCHITO-GENIE
# ===========================

# Standard libs
import os
import uuid
import json
import base64  # (on s'en servira pour la version B)
from datetime import datetime
from typing import List, Optional

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

# Pydantic
from pydantic import BaseModel

# External requests
import requests

# File generation (DOCX + PDF)
from io import BytesIO
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===========================
# 📌 FIN DES IMPORTS
# ===========================



# ==============================================================================
# CONFIGURATION
# ==============================================================================

app = FastAPI(
    title="Archito-Genie API",
    description="AI-assisted structural, MEPF and sustainability conceptual design generator",
    version="1.0.0"
)

# CORS - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create files directory if not exists
os.makedirs("files", exist_ok=True)

# Mount static files
app.mount("/files", StaticFiles(directory="files"), name="files")

# In-memory storage (replace with database in production)
PROJECTS: dict = {}
ENGINEERING_RESULTS: dict = {}

# REPORTS stocke un dict de chaînes (toutes les sections markdown)
REPORTS: dict[str, dict[str, str]] = {}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class ProjectCreate(BaseModel):
    name: str
    location: str
    building_type: str
    levels: int
    gross_floor_area_m2: float
    climate_zone: Optional[str] = "hot-humid"
    target_certifications: Optional[List[str]] = ["EDGE"]


class Project(BaseModel):
    id: str
    name: str
    location: str
    building_type: str
    levels: int
    gross_floor_area_m2: float
    climate_zone: str
    target_certifications: List[str]
    files: List[str]
    analyzed: bool
    created_at: str


class BOQItem(BaseModel):
    item: str
    description: str
    unit: str
    quantity: float
    unit_rate: float
    total: float


class BOQOption(BaseModel):
    tier: str  # Basic, High-End, Luxury
    items: List[BOQItem]
    subtotal: float
    cost_per_m2: float


class SustainabilityResult(BaseModel):
    edge_energy_savings_percent: float
    edge_water_savings_percent: float
    edge_materials_savings_percent: float
    edge_level: str
    leed_points: Optional[int] = None
    leed_level: Optional[str] = None


class EngineeringResult(BaseModel):
    project_id: str
    summary: dict
    structural: dict
    mepf: dict
    automation: dict
    sustainability: SustainabilityResult
    boq_options: List[BOQOption]


class ReportResponse(BaseModel):
    project_id: str
    report_markdown: str

    # Sections détaillées (toutes optionnelles pour éviter les 500)
    narrative_markdown: Optional[str] = None
    calc_notes_markdown: Optional[str] = None
    schematics_markdown: Optional[str] = None
    datasheets_markdown: Optional[str] = None
    boq_basic_markdown: Optional[str] = None
    boq_high_end_markdown: Optional[str] = None
    boq_luxury_markdown: Optional[str] = None
    structural_spec_markdown: Optional[str] = None
    mepf_spec_markdown: Optional[str] = None
    disclaimer_markdown: Optional[str] = None




# ==============================================================================
# ARCHITO-GENIE SYSTEM PROMPT
# ==============================================================================

ARCHITO_GENIE_SYSTEM_PROMPT = """You are Archito-Genie, an AI structural & MEPF conceptual design generator.

You receive a single JSON called `engineering_result`.
You MUST NOT invent new numerical values.
Use only the numbers present inside the JSON.

Always begin the output with this disclaimer:

"**Disclaimer:** The following outputs are conceptual and for preliminary studies only. They must be reviewed, completed, and validated by licensed structural and MEPF engineers and official sustainability consultants (EDGE/LEED) before any construction or certification."

Then output the following **7 sections, in order**:

## 1. DESIGN NARRATIVE & PRINCIPLES
- Project overview and design philosophy
- Key design drivers and constraints
- Sustainability approach

## 2. CALCULATIONS (CONCEPTUAL / PRELIMINARY)
- Structural load assumptions
- MEPF sizing basis
- All values must come from the input JSON

## 3. DESIGN SCHEMATICS
- Conceptual layout descriptions
- System integration approach
- Zoning concepts

## 4. STRUCTURAL DRAWINGS (CONCEPTUAL)
- Foundation concept
- Framing system description
- Structural grid layout

## 5. MEPF / INTEGRATION / AUTOMATION DRAWINGS (CONCEPTUAL)
- HVAC system layout concept
- Plumbing riser concept
- Electrical distribution concept
- Fire protection concept
- BMS/automation scope

## 6. DATASHEETS
- Equipment specifications summary
- Material specifications
- Performance targets

## 7. BILL OF QUANTITIES – 3 OPTIONS (Basic, High-End, Luxury)
Present the BOQ data from the JSON in a clear table format for each tier.

Use markdown headings, tables, and bullet lists.
Avoid fluff. Be precise.
Do not omit any section."""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def is_coastal_city(location: str) -> bool:
    """Check if location is a known coastal city."""
    coastal_cities = [
        "dakar", "abidjan", "bissau", "lagos", "accra", "lome", "cotonou",
        "libreville", "douala", "luanda", "maputo", "mombasa", "dar es salaam",
        "zanzibar", "cape town", "durban", "casablanca", "tunis", "algiers",
        "alexandria", "saint-louis", "conakry", "freetown", "monrovia"
    ]
    location_lower = location.lower()
    return any(city in location_lower for city in coastal_cities)


def calculate_structural_design(project: Project) -> dict:
    """Generate conceptual structural design based on project parameters."""
    area = project.gross_floor_area_m2
    levels = project.levels
    building_type = project.building_type.lower()

    # 🚧 Protection : empêcher division par zéro
    if levels is None or levels <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid project data: 'levels' must be greater than 0 to run analysis.",
        )

    
    # Determine structural system based on building type and size
    if levels <= 3:
        system_type = "Load-bearing masonry walls with RC slabs"
        foundation_type = "Strip foundations"
        typical_span = "4.0m - 5.0m"
        slab_thickness = 150
    elif levels <= 8:
        system_type = "RC frame with infill masonry"
        foundation_type = "Isolated pad footings with tie beams"
        typical_span = "6.0m - 7.5m"
        slab_thickness = 180
    else:
        system_type = "RC shear wall core with perimeter frame"
        foundation_type = "Raft foundation or piled foundations"
        typical_span = "7.5m - 9.0m"
        slab_thickness = 200
    
    # Grid calculation
    floor_area_per_level = area / levels
    grid_x = round((floor_area_per_level ** 0.5) / 6) + 1
    grid_y = round((floor_area_per_level ** 0.5) / 6) + 1
    
    return {
        "system_type": system_type,
        "grid_spacing": f"{grid_x} x {grid_y} grid @ 6.0m c/c",
        "slab_thickness_mm": slab_thickness,
        "foundation_type": foundation_type,
        "spans": typical_span,
        "structural_concrete_grade": "C30/37",
        "reinforcement_grade": "B500B",
        "design_code": "Eurocode 2 / ACI 318"
    }


def calculate_mepf_design(project: Project) -> dict:
    """Generate conceptual MEPF design based on project parameters."""
    area = project.gross_floor_area_m2
    levels = project.levels
    climate = project.climate_zone
    building_type = project.building_type.lower()
    
    # HVAC sizing (W/m² based on climate and building type)
    cooling_load_factor = {
        "hot-humid": 150,
        "hot-dry": 130,
        "tropical": 160,
        "temperate": 100,
        "mediterranean": 110,
        "cold": 80
    }.get(climate, 120)
    
    cooling_load_kw = round(area * cooling_load_factor / 1000, 1)
    
    # HVAC system type based on building size
    if area < 500:
        hvac_system = "Split system AC units"
    elif area < 2000:
        hvac_system = "VRF (Variable Refrigerant Flow) system"
    else:
        hvac_system = "Chilled water system with AHUs"
    
    # Electrical sizing
    electrical_factor = 50  # VA/m² typical
    electrical_load_kva = round(area * electrical_factor / 1000, 1)
    
    # Water demand
    water_demand_lpd = round(area * 10)  # 10 L/m²/day typical
    
    return {
        "hvac_system": hvac_system,
        "cooling_load_kw": cooling_load_kw,
        "cooling_load_per_m2": cooling_load_factor,
        "water_supply": f"Municipal connection with {round(water_demand_lpd/1000, 1)}m³/day demand",
        "drainage": "Gravity drainage to municipal sewer",
        "electrical_load_kva": electrical_load_kva,
        "electrical_system": "3-phase 400V/230V supply",
        "fire_protection": "Wet sprinkler system per NFPA 13" if area > 500 else "Fire extinguishers and smoke detectors",
        "ventilation": f"Mechanical ventilation with {round(area * 10)} L/s fresh air"
    }


def calculate_automation(project: Project) -> dict:
    """Generate BMS/automation scope."""
    area = project.gross_floor_area_m2
    
    if area < 1000:
        return {
            "bms_scope": ["Basic lighting control", "Individual AC control"],
            "iot_sensors": ["Motion sensors", "Temperature sensors"]
        }
    elif area < 5000:
        return {
            "bms_scope": [
                "Centralized HVAC control",
                "Lighting automation with daylight harvesting",
                "Energy monitoring",
                "Access control integration"
            ],
            "iot_sensors": [
                "Occupancy sensors",
                "Temperature/humidity sensors",
                "CO2 sensors",
                "Energy meters"
            ]
        }
    else:
        return {
            "bms_scope": [
                "Full BMS integration",
                "HVAC optimization with AI",
                "Lighting automation",
                "Fire alarm integration",
                "Vertical transport monitoring",
                "Energy management system",
                "Fault detection and diagnostics"
            ],
            "iot_sensors": [
                "Occupancy sensors",
                "Environmental sensors (T/RH/CO2/PM2.5)",
                "Water leak sensors",
                "Energy meters per floor",
                "Air quality monitors"
            ]
        }


def calculate_sustainability(project: Project) -> SustainabilityResult:
    """Calculate sustainability metrics."""
    # Baseline EDGE calculations
    energy_savings = 25  # Base 25% from efficient HVAC
    water_savings = 20   # Base 20% from efficient fixtures
    materials_savings = 15  # Base 15% from local materials
    
    # Adjust based on certifications
    if "EDGE" in project.target_certifications:
        energy_savings += 5
        water_savings += 5
    if "LEED" in project.target_certifications:
        energy_savings += 10
        water_savings += 10
        materials_savings += 5
    
    # Determine EDGE level
    if energy_savings >= 40 and water_savings >= 40 and materials_savings >= 20:
        edge_level = "EDGE Advanced"
    elif energy_savings >= 20 and water_savings >= 20:
        edge_level = "EDGE Certified"
    else:
        edge_level = "Pre-certification"
    
    # LEED points estimation
    leed_points = min(round((energy_savings + water_savings + materials_savings) * 0.8), 80)
    leed_level = (
        "Platinum" if leed_points >= 80 else
        "Gold" if leed_points >= 60 else
        "Silver" if leed_points >= 50 else
        "Certified" if leed_points >= 40 else None
    )
    
    return SustainabilityResult(
        edge_energy_savings_percent=energy_savings,
        edge_water_savings_percent=water_savings,
        edge_materials_savings_percent=materials_savings,
        edge_level=edge_level,
        leed_points=leed_points,
        leed_level=leed_level
    )


def generate_boq_options(project: Project, structural: dict, mepf: dict) -> List[BOQOption]:
    """Generate 3 BOQ options: Basic, High-End, Luxury."""
    area = project.gross_floor_area_m2
    
    # Cost multipliers per m² (USD)
    tiers = {
        "Basic": {"structure": 150, "mepf": 100, "finishes": 50, "automation": 20, "multiplier": 1.0},
        "High-End": {"structure": 200, "mepf": 150, "finishes": 100, "automation": 50, "multiplier": 1.5},
        "Luxury": {"structure": 280, "mepf": 220, "finishes": 180, "automation": 100, "multiplier": 2.2}
    }
    
    options = []
    for tier_name, costs in tiers.items():
        items = [
            BOQItem(
                item="Structure",
                description=f"{structural['system_type']} - {structural['foundation_type']}",
                unit="m²",
                quantity=area,
                unit_rate=costs["structure"],
                total=area * costs["structure"]
            ),
            BOQItem(
                item="MEPF",
                description=f"{mepf['hvac_system']} + {mepf['electrical_system']}",
                unit="m²",
                quantity=area,
                unit_rate=costs["mepf"],
                total=area * costs["mepf"]
            ),
            BOQItem(
                item="Finishes",
                description=f"{tier_name} grade finishes and fixtures",
                unit="m²",
                quantity=area,
                unit_rate=costs["finishes"],
                total=area * costs["finishes"]
            ),
            BOQItem(
                item="Automation",
                description="BMS and smart building features",
                unit="m²",
                quantity=area,
                unit_rate=costs["automation"],
                total=area * costs["automation"]
            )
        ]
        
        subtotal = sum(item.total for item in items)
        
        options.append(BOQOption(
            tier=tier_name,
            items=items,
            subtotal=subtotal,
            cost_per_m2=round(subtotal / area, 2)
        ))
    
    return options


# =====================================================================
# 📎 Helpers pour construire DOCX et PDF à partir du rapport Markdown
# =====================================================================

SECTION_ORDER = [
    "narrative_markdown",
    "calc_notes_markdown",
    "schematics_markdown",
    "datasheets_markdown",
    "boq_basic_markdown",
    "boq_high_end_markdown",
    "boq_luxury_markdown",
    "structural_spec_markdown",
    "mepf_spec_markdown",
    "disclaimer_markdown",
]


def _build_plain_text_from_report(report: dict) -> str:
    """
    Transforme le dict REPORTS[project_id] en un gros texte brut.
    (simple mais suffisant pour le MVP DOCX/PDF)
    """
    if report.get("report_markdown"):
        return report["report_markdown"]

    parts = []
    for key in SECTION_ORDER:
        value = report.get(key)
        if value:
            title = key.replace("_markdown", "").replace("_", " ").upper()
            parts.append(f"## {title}\n\n{value}")

    return "\n\n\n".join(parts) if parts else "No report content available."


def _build_docx_stream(project_id: str, report: dict) -> BytesIO:
    """
    Construit un DOCX en mémoire et renvoie un BytesIO prêt à être streamé.
    """
    document = Document()

    document.add_heading(
        f"Archito-Genie Conceptual Design Report – {project_id}", level=1
    )
    document.add_paragraph("")  # ligne vide

    text = _build_plain_text_from_report(report)

    for line in text.split("\n"):
        # On garde ça très simple : 1 paragraphe par ligne
        document.add_paragraph(line)

    stream = BytesIO()
    document.save(stream)
    stream.seek(0)
    return stream


def _build_pdf_stream(project_id: str, report: dict) -> BytesIO:
    """
    Construit un PDF simple (texte brut) en mémoire avec reportlab.
    """
    text = _build_plain_text_from_report(report)

    stream = BytesIO()
    c = canvas.Canvas(stream, pagesize=A4)

    width, height = A4
    x_margin = 40
    y = height - 50

    title = f"Archito-Genie Conceptual Design Report – {project_id}"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, title)
    y -= 30

    c.setFont("Helvetica", 10)

    # Simple word wrap
    max_width = width - 2 * x_margin
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            y -= 12
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            continue

        words = line.split(" ")
        current = ""
        for w in words:
            test = (current + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 10) <= max_width:
                current = test
            else:
                c.drawString(x_margin, y, current)
                y -= 12
                if y < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 50
                current = w
        if current:
            c.drawString(x_margin, y, current)
            y -= 12
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50

    c.showPage()
    c.save()
    stream.seek(0)
    return stream


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Archito-Genie API", "version": "1.0.0"}


@app.post("/projects", response_model=Project)
async def create_project(data: ProjectCreate):
    """Create a new project."""
    project_id = str(uuid.uuid4())
    
    project = Project(
        id=project_id,
        name=data.name,
        location=data.location,
        building_type=data.building_type,
        levels=data.levels,
        gross_floor_area_m2=data.gross_floor_area_m2,
        climate_zone=data.climate_zone or "hot-humid",
        target_certifications=data.target_certifications or ["EDGE"],
        files=[],
        analyzed=False,
        created_at=datetime.now().isoformat()
    )
    
    PROJECTS[project_id] = project
    return project


@app.post("/projects/{project_id}/files", response_model=Project)
async def upload_files(
    project_id: str,
    architectural_plan: UploadFile = File(...),
    soil_report: Optional[UploadFile] = File(None),
    additional_files: List[UploadFile] = File(default=[])
):
    """Upload project files."""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = PROJECTS[project_id]
    uploaded_files = []
    
    # Validate architectural plan extension
    allowed_extensions = ['.rvt', '.dwg', '.pdf']
    ext = os.path.splitext(architectural_plan.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Architectural plan must be one of: {', '.join(allowed_extensions)}"
        )
    
    # Save architectural plan (mandatory)
    arch_filename = f"{project_id}_arch_{architectural_plan.filename}"
    arch_path = os.path.join("files", arch_filename)
    with open(arch_path, "wb") as f:
        content = await architectural_plan.read()
        f.write(content)
    uploaded_files.append(arch_filename)
    
    # Save soil report (optional)
    if soil_report:
        soil_filename = f"{project_id}_soil_{soil_report.filename}"
        soil_path = os.path.join("files", soil_filename)
        with open(soil_path, "wb") as f:
            content = await soil_report.read()
            f.write(content)
        uploaded_files.append(soil_filename)
    
    # Save additional files (optional)
    for file in additional_files:
        if file.filename:
            add_filename = f"{project_id}_add_{file.filename}"
            add_path = os.path.join("files", add_filename)
            with open(add_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_files.append(add_filename)
    
    # Update project
    project.files = uploaded_files
    PROJECTS[project_id] = project
    
    return project


@app.post("/projects/{project_id}/analyze", response_model=EngineeringResult)
async def analyze_project(project_id: str):
    """Run technical conceptual analysis."""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = PROJECTS[project_id]
    
    if not project.files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Infer environmental data
    proximity_to_sea = is_coastal_city(project.location)
    
    # Generate conceptual designs
    structural = calculate_structural_design(project)
    mepf = calculate_mepf_design(project)
    automation = calculate_automation(project)
    sustainability = calculate_sustainability(project)
    boq_options = generate_boq_options(project, structural, mepf)
    
    result = EngineeringResult(
        project_id=project_id,
        summary={
            "building_type": project.building_type,
            "location": project.location,
            "levels": project.levels,
            "gross_floor_area_m2": project.gross_floor_area_m2,
            "climate_zone": project.climate_zone,
            "proximity_to_sea": proximity_to_sea,
            "target_certifications": project.target_certifications
        },
        structural=structural,
        mepf=mepf,
        automation=automation,
        sustainability=sustainability,
        boq_options=boq_options
    )
    
    # Store result
    ENGINEERING_RESULTS[project_id] = result
    project.analyzed = True
    PROJECTS[project_id] = project
    
    return result


@app.get("/projects/{project_id}/report", response_model=ReportResponse)
async def generate_report(project_id: str) -> ReportResponse:
    """Generate the full AI report using OpenAI API."""

    # 1) Vérifier qu’on a bien un résultat d’ingénierie
    if project_id not in ENGINEERING_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="Engineering result not found. Run analysis first.",
        )

    # 2) Clé OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured on server",
        )

    engineering_result = ENGINEERING_RESULTS[project_id]
    project = PROJECTS[project_id]

    # 3) Sérialiser en JSON pour le prompt
    result_json = json.dumps(engineering_result.dict(), indent=2)
    project_json = json.dumps(project.dict(), indent=2)

    # 4) Prompt : on demande un JSON structuré
    prompt = f"""
You are Archito-Genie, an assistant generating conceptual engineering & sustainability design reports.

You receive two JSON objects:

1) PROJECT DATA (architectural + context):
{project_json}

2) ENGINEERING ANALYSIS RESULT (structural, MEPF, sustainability):
{result_json}

Using ONLY this data and industry best practices, produce ONE valid JSON object with the following fields, all as Markdown strings:

{{
  "narrative_markdown": "Full design narrative and principles...",
  "calc_notes_markdown": "Detailed calculation notes (with formulas, assumptions, and typical values)...",
  "schematics_markdown": "Textual description of schematics and system diagrams (with references to future DWG/RVT)...",
  "datasheets_markdown": "Summary of key equipment datasheets and performance parameters...",
  "boq_basic_markdown": "Bill of quantities, BASIC option (tabular Markdown)...",
  "boq_high_end_markdown": "Bill of quantities, HIGH-END option (tabular Markdown)...",
  "boq_luxury_markdown": "Bill of quantities, LUXURY option (tabular Markdown)...",
  "structural_spec_markdown": "Structural design brief and performance specifications...",
  "mepf_spec_markdown": "MEPF & Automation design brief and performance specifications...",
  "disclaimer_markdown": "Regulatory disclaimer and professional responsibility notes..."
}}

Rules:
- Respond with valid JSON only, no extra text.
- Each field must contain well-structured Markdown, with headings (##, ###), bullet lists and tables where useful.
- Use realistic but generic values when exact data is missing, and clearly mark assumptions.
- Aim for professional, client-ready wording.
"""

    # 5) Appel à l’API OpenAI Responses
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4.1-mini",
        "input": prompt,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # === Extraction simplifiée et robuste ===
        # Le Responses API renvoie déjà le texte complet dans `output_text`
        ai_text = data.get("output_text")
        if not ai_text:
            # Fallback ultra-sécurisé : on sérialise toute la réponse
            # (ça ne devrait normalement pas arriver)
            ai_text = json.dumps(data)

        ai_text = ai_text.strip()
        # ========================================

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {e}"
        )


    # 6) On tente de parser JSON
    try:
        sections = json.loads(ai_text)

        required_fields = [
            "narrative_markdown",
            "calc_notes_markdown",
            "schematics_markdown",
            "datasheets_markdown",
            "boq_basic_markdown",
            "boq_high_end_markdown",
            "boq_luxury_markdown",
            "structural_spec_markdown",
            "mepf_spec_markdown",
            "disclaimer_markdown",
        ]

        missing = [f for f in required_fields if f not in sections]

        # 🔹 Si le JSON n'a pas tous les champs attendus,
        # on NE lève plus d'erreur 500 : on utilise simplement
        # tout le texte de l'IA comme rapport brut.
        if missing:
            full_report = ai_text.strip()

            REPORTS[project_id] = {
                "project_id": project_id,
                "report_markdown": full_report,
            }

            return ReportResponse(
                project_id=project_id,
                report_markdown=full_report,
            )

        # 🔹 Cas idéal : tous les champs sont présents
        parts = []
        parts.append("## DESIGN NARRATIVE & PRINCIPLES\n\n" + sections.get("narrative_markdown", ""))
        parts.append("## CALCULATION NOTES\n\n" + sections.get("calc_notes_markdown", ""))
        parts.append("## SCHEMATICS OVERVIEW\n\n" + sections.get("schematics_markdown", ""))
        parts.append("## EQUIPMENT DATASHEETS SUMMARY\n\n" + sections.get("datasheets_markdown", ""))
        parts.append("## BILL OF QUANTITIES – BASIC OPTION\n\n" + sections.get("boq_basic_markdown", ""))
        parts.append("## BILL OF QUANTITIES – HIGH-END OPTION\n\n" + sections.get("boq_high_end_markdown", ""))
        parts.append("## BILL OF QUANTITIES – LUXURY OPTION\n\n" + sections.get("boq_luxury_markdown", ""))
        parts.append("## STRUCTURAL DESIGN BRIEF\n\n" + sections.get("structural_spec_markdown", ""))
        parts.append("## MEPF & AUTOMATION DESIGN BRIEF\n\n" + sections.get("mepf_spec_markdown", ""))
        parts.append("## DISCLAIMER\n\n" + sections.get("disclaimer_markdown", ""))

        full_report = "\n\n---\n\n".join(parts)

        # On stocke dans REPORTS pour les exports DOCX/PDF
        REPORTS[project_id] = {
            "project_id": project_id,
            "report_markdown": full_report,
            "narrative_markdown": sections.get("narrative_markdown", ""),
            "calc_notes_markdown": sections.get("calc_notes_markdown", ""),
            "schematics_markdown": sections.get("schematics_markdown", ""),
            "datasheets_markdown": sections.get("datasheets_markdown", ""),
            "boq_basic_markdown": sections.get("boq_basic_markdown", ""),
            "boq_high_end_markdown": sections.get("boq_high_end_markdown", ""),
            "boq_luxury_markdown": sections.get("boq_luxury_markdown", ""),
            "structural_spec_markdown": sections.get("structural_spec_markdown", ""),
            "mepf_spec_markdown": sections.get("mepf_spec_markdown", ""),
            "disclaimer_markdown": sections.get("disclaimer_markdown", ""),
        }

        return ReportResponse(
            project_id=project_id,
            report_markdown=full_report,
            narrative_markdown=sections.get("narrative_markdown", ""),
            calc_notes_markdown=sections.get("calc_notes_markdown", ""),
            schematics_markdown=sections.get("schematics_markdown", ""),
            datasheets_markdown=sections.get("datasheets_markdown", ""),
            boq_basic_markdown=sections.get("boq_basic_markdown", ""),
            boq_high_end_markdown=sections.get("boq_high_end_markdown", ""),
            boq_luxury_markdown=sections.get("boq_luxury_markdown", ""),
            structural_spec_markdown=sections.get("structural_spec_markdown", ""),
            mepf_spec_markdown=sections.get("mepf_spec_markdown", ""),
            disclaimer_markdown=sections.get("disclaimer_markdown", ""),
        )

    except HTTPException:
        # on relance tel quel si on a déjà construit un message clair
        raise
    except Exception:
        # 8) Fallback : si le modèle n’a pas renvoyé du vrai JSON,
        # on renvoie le texte brut quand même.
        full_report = ai_text.strip()

        REPORTS[project_id] = {
            "project_id": project_id,
            "report_markdown": full_report,
        }

        return ReportResponse(
            project_id=project_id,
            report_markdown=full_report,
        )


    except HTTPException:
        # on relance tel quel si on a déjà construit un message clair
        raise
    except Exception:
        # Fallback : si le modèle n’a pas renvoyé du vrai JSON,
        # on stocke quand même quelque chose d'exportable
        REPORTS[project_id] = {
            "project_id": project_id,
            "report_markdown": ai_text,
        }
        return ReportResponse(
            project_id=project_id,
            report_markdown=ai_text,
        )

# ============================================
# EXPORT DOCX
# ============================================
@app.get("/projects/{project_id}/export/docx")
async def export_docx(project_id: str):

    if project_id not in REPORTS:
        raise HTTPException(status_code=404, detail="Report not found")

    report = REPORTS[project_id]["report_markdown"]

    # Génération DOCX
    document = Document()
    for line in report.split("\n"):
        document.add_paragraph(line)

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)

    filename = f"{project_id}.docx"

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================
# EXPORT PDF
# ============================================
@app.get("/projects/{project_id}/export/pdf")
async def export_pdf(project_id: str):

    if project_id not in REPORTS:
        raise HTTPException(status_code=404, detail="Report not found")

    report = REPORTS[project_id]["report_markdown"]

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Placement basique du texte dans la page
    x, y = 40, 800
    for line in report.split("\n"):
        c.drawString(x, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = 800

    c.save()
    buffer.seek(0)

    filename = f"{project_id}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ============================================
# SCHEMATICS SVG (Version A – backend only)
# ============================================

@app.get("/projects/{project_id}/schematics/svg")
async def export_schematics_svg(project_id: str):
    """
    Génère un schéma conceptuel simple (SVG) à partir du projet
    et du résultat d'analyse technique.
    """

    # Vérifier que le projet et l'analyse existent
    if project_id not in PROJECTS or project_id not in ENGINEERING_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="Project or engineering result not found"
        )

    project = PROJECTS[project_id]
    engineering_result = ENGINEERING_RESULTS[project_id]

    # On récupère quelques infos de base pour annoter le schéma
    try:
        project_dict = project.dict()
    except Exception:
        project_dict = {}

    name = project_dict.get("name") or "Conceptual project"
    location = project_dict.get("location") or ""
    levels = project_dict.get("levels")
    levels_label = f"{levels} levels" if levels else "Multi-level"

    # Texte pour les trois blocs (structure, MEPF, sustainability)
    # On reste volontairement simple pour éviter les bugs.
    struct_label = "Structure"
    mepf_label = "MEPF & Automation"
    sust_label = "Sustainability"

    # Construction du SVG (version simplifiée mais propre)
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600">
  <style>
    .title {{ font: bold 24px sans-serif; }}
    .subtitle {{ font: 14px sans-serif; fill: #555; }}
    .box {{ fill: #f5f5f5; stroke: #333; stroke-width: 2; rx: 12; ry: 12; }}
    .label {{ font: bold 14px sans-serif; }}
    .text {{ font: 12px sans-serif; fill: #333; }}
  </style>

  <!-- Titre -->
  <text x="40" y="50" class="title">{name}</text>
  <text x="40" y="80" class="subtitle">{location} – {levels_label}</text>
  <text x="40" y="110" class="subtitle">Conceptual Systems Overview</text>

  <!-- Bloc Structure -->
  <rect x="40" y="140" width="820" height="110" class="box" />
  <text x="60" y="170" class="label">{struct_label}</text>
  <text x="60" y="195" class="text">Primary frame: columns / beams / slabs designed for gravity & lateral loads.</text>
  <text x="60" y="215" class="text">Foundation sized for local soil conditions and typical residential loads.</text>

  <!-- Bloc MEPF -->
  <rect x="40" y="280" width="820" height="110" class="box" />
  <text x="60" y="310" class="label">{mepf_label}</text>
  <text x="60" y="335" class="text">HVAC zoning per floor, fresh air & exhaust ducts, main plant room on roof or basement.</text>
  <text x="60" y="355" class="text">Electrical single line: main LV panel, sub-distribution per floor, critical loads on backup.</text>

  <!-- Bloc Sustainability -->
  <rect x="40" y="420" width="820" height="110" class="box" />
  <text x="60" y="450" class="label">{sust_label}</text>
  <text x="60" y="475" class="text">Envelope optimized for local climate (solar control, shading, natural ventilation).</text>
  <text x="60" y="495" class="text">Water & energy efficiency measures sized conceptually for early-stage design.</text>

</svg>"""

    # On renvoie le SVG comme fichier téléchargeable
    from io import BytesIO

    buffer = BytesIO(svg_content.encode("utf-8"))
    buffer.seek(0)

    filename = f"{project_id}_schematics.svg"

    return StreamingResponse(
        buffer,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# =====================================================================
# 📤 Endpoints d'export DOCX et PDF
# =====================================================================

def _get_report_dict_or_404(project_id: str) -> dict:
    """
    Récupère le rapport en mémoire ; si absent mais l'analyse existe,
    on déclenche un generate_report(); sinon 404.
    """
    if project_id in REPORTS:
        return REPORTS[project_id]

    # Si on a déjà de l'ingénierie, on peut générer le rapport à la volée
    if project_id in ENGINEERING_RESULTS:
        # generate_report est async → on ne peut pas l'appeler ici directement
        # Cette fonction est utilisée seulement dans les endpoints async ci-dessous
        raise RuntimeError("generate_report must be awaited in async context.")

    raise HTTPException(
        status_code=404,
        detail="Report not found. Run analysis and report generation first.",
    )



@app.get("/projects/{project_id}/export/docx")
async def export_report_docx(project_id: str):
    """
    Export du rapport Archito-Genie en DOCX.
    """
    if project_id not in REPORTS:
        if project_id in ENGINEERING_RESULTS:
            # On génère le rapport si pas encore fait
            await generate_report(project_id)
        else:
            raise HTTPException(
                status_code=404,
                detail="Report not found. Run analysis and report generation first.",
            )

    report = REPORTS[project_id]
    stream = _build_docx_stream(project_id, report)

    filename = f"archito-genie-report-{project_id}.docx"

    return StreamingResponse(
        stream,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/projects/{project_id}/export/pdf")
async def export_report_pdf(project_id: str):
    """
    Export du rapport Archito-Genie en PDF.
    """
    if project_id not in REPORTS:
        if project_id in ENGINEERING_RESULTS:
            await generate_report(project_id)
        else:
            raise HTTPException(
                status_code=404,
                detail="Report not found. Run analysis and report generation first.",
            )

    report = REPORTS[project_id]
    stream = _build_pdf_stream(project_id, report)

    filename = f"archito-genie-report-{project_id}.pdf"

    return StreamingResponse(
        stream,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )












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

import os
import uuid
import json
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

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


@app.get("/projects/{project_id}/engineering-result", response_model=EngineeringResult)
async def get_engineering_result(project_id: str):
    """Get the engineering analysis result."""
    if project_id not in ENGINEERING_RESULTS:
        raise HTTPException(status_code=404, detail="Engineering result not found. Run analysis first.")
    
    return ENGINEERING_RESULTS[project_id]


@app.get("/projects/{project_id}/report", response_model=ReportResponse)
async def generate_report(project_id: str):
    """Generate the full AI report using OpenAI API."""
    if project_id not in ENGINEERING_RESULTS:
        raise HTTPException(status_code=404, detail="Engineering result not found. Run analysis first.")
    
    # Get OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")
    
    engineering_result = ENGINEERING_RESULTS[project_id]
    project = PROJECTS[project_id]
    
    # Convert to JSON for the prompt
    result_json = json.dumps(engineering_result.dict(), indent=2)
    
    # Call OpenAI API
    client = OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-5" when available
            messages=[
                {"role": "system", "content": ARCHITO_GENIE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Project: {project.name}\n\nEngineering Result JSON:\n```json\n{result_json}\n```\n\nGenerate the complete 7-section Archito-Genie design report."}
            ],
            max_tokens=8000,
            temperature=0.7
        )
        
        report_markdown = response.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
    return ReportResponse(
        project_id=project_id,
        report_markdown=report_markdown
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

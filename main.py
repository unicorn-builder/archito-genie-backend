import os
import uuid
import io
import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Any

import boto3
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from docx import Document  # python-docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import svgwrite


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
    edge_compliant: Optional[bool] = False


class Project(BaseModel):
    id: str
    name: str
    created_at: datetime
    edge_compliant: bool = False


class EdgeUpdate(BaseModel):
    edge_compliant: bool


PROJECTS: Dict[str, Project] = {}
PROJECT_DATA: Dict[str, Dict[str, Any]] = {}


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
# Helpers exports (rapport PDF / DOCX)
# ============================================================

def markdown_to_docx_bytes(markdown_text: str) -> bytes:
    doc = Document()
    for line in markdown_text.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x_margin = 72
    y = height - 72
    max_chars = 90

    for line in markdown_text.splitlines():
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
# App FastAPI
# ============================================================

app = FastAPI(
    title="Archito-Genie Backend",
    description=(
        "MVP Archito-Genie : Narratifs MEPF & Automation + Schematics + "
        "BOQ multi-variantes + Datasheets + EDGE, à partir d'un plan archi."
    ),
    version="0.9.0",
)


# ============================================================
# Outil interne : layout de pièces (pour raisonner MEP)
# ============================================================

def _layout_rooms(rooms: List[Dict[str, Any]], scale: float = 40.0):
    max_rooms_per_row = max(1, math.ceil(math.sqrt(len(rooms)))) if rooms else 1
    current_x = 0.0
    current_y = 0.0
    row_height_m = 0.0
    layout_positions = []

    for idx, room in enumerate(rooms):
        w = float(room.get("width_m", 3.0))
        l = float(room.get("length_m", 3.0))

        if idx % max_rooms_per_row == 0 and idx != 0:
            current_x = 0.0
            current_y += row_height_m
            row_height_m = 0.0

        layout_positions.append(
            {"room": room, "x_m": current_x, "y_m": current_y, "w_m": w, "l_m": l}
        )
        current_x += w
        row_height_m = max(row_height_m, l)

    total_width_m = max((p["x_m"] + p["w_m"]) for p in layout_positions) if layout_positions else 8.0
    total_height_m = max((p["y_m"] + p["l_m"]) for p in layout_positions) if layout_positions else 8.0
    return layout_positions, total_width_m, total_height_m


# ============================================================
# AI : génération du modèle de projet (plan_spec)
# ============================================================

def get_plan_spec_from_ai(file_name: str, file_bytes: bytes, edge_compliant: bool) -> Dict[str, Any]:
    approx_size_kb = max(len(file_bytes) // 1024, 1)
    edge_text = "OUI, le client souhaite une étude compatible avec la certification EDGE." if edge_compliant else "NON, le client ne demande pas explicitement la certification EDGE."

    user_content = f"""
Tu es un ingénieur/architecte assistant pour une application SaaS nommé Archito-Genie.

On a reçu un plan architectural nommé "{file_name}", taille approximative {approx_size_kb} Ko.
Contexte EDGE : {edge_text}

TA TÂCHE :
- Construire une hypothèse réaliste de bâtiment basée sur ce que tu sais des immeubles
  résidentiels/mixtes (ou tertiaires) typiques.
- Si EDGE = OUI, favorise compacité, éclairage naturel, ventilation naturelle, etc.

- Renvoie STRICTEMENT un JSON avec le format suivant :

{{
  "building_type": "residential" | "office" | "mixed",
  "edge_compliant": true | false,
  "floors": [
    {{
      "name": "RDC",
      "level": 0,
      "rooms": [
        {{
          "name": "Séjour",
          "usage": "living_room | bedroom | kitchen | circulation | sanitary | technical | office | retail | meeting",
          "width_m": 5.0,
          "length_m": 6.0,
          "has_window": true,
          "is_wet_area": false,
          "is_technical_room": false,
          "expected_occupancy": 3
        }}
      ]
    }}
  ]
}}

CONTRAINTES :
- ENTRE 5 ET 15 pièces par niveau.
- Dimensions crédibles (2m à 8m par côté).
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
                        "Tu produis une description JSON cohérente de la distribution des pièces, "
                        "en préparant le terrain pour des études MEPF & Automation."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        )
        content = completion.choices[0].message.content
        spec = json.loads(content)
        if "edge_compliant" not in spec:
            spec["edge_compliant"] = edge_compliant
        return spec
    except Exception as e:
        print(f"[WARN] get_plan_spec_from_ai failed, using fallback spec: {e}")
        return {
            "building_type": "residential",
            "edge_compliant": edge_compliant,
            "floors": [
                {
                    "name": "RDC",
                    "level": 0,
                    "rooms": [
                        {
                            "name": "Séjour",
                            "usage": "living_room",
                            "width_m": 5.0,
                            "length_m": 6.0,
                            "has_window": True,
                            "is_wet_area": False,
                            "is_technical_room": False,
                            "expected_occupancy": 4,
                        },
                        {
                            "name": "Cuisine",
                            "usage": "kitchen",
                            "width_m": 3.0,
                            "length_m": 4.0,
                            "has_window": True,
                            "is_wet_area": True,
                            "is_technical_room": False,
                            "expected_occupancy": 2,
                        },
                        {
                            "name": "Chambre 1",
                            "usage": "bedroom",
                            "width_m": 4.0,
                            "length_m": 4.0,
                            "has_window": True,
                            "is_wet_area": False,
                            "is_technical_room": False,
                            "expected_occupancy": 2,
                        },
                        {
                            "name": "Chambre 2",
                            "usage": "bedroom",
                            "width_m": 3.5,
                            "length_m": 3.5,
                            "has_window": True,
                            "is_wet_area": False,
                            "is_technical_room": False,
                            "expected_occupancy": 2,
                        },
                        {
                            "name": "Salle de bain",
                            "usage": "sanitary",
                            "width_m": 3.0,
                            "length_m": 3.0,
                            "has_window": False,
                            "is_wet_area": True,
                            "is_technical_room": False,
                            "expected_occupancy": 1,
                        },
                    ],
                }
            ],
        }


# ============================================================
# AI : BOQ (3 variantes) & Datasheets
# ============================================================

def generate_boq_from_spec(plan_spec: Dict[str, Any], edge_compliant: bool) -> Dict[str, Any]:
    edge_text = (
        "Le maître d'ouvrage souhaite une étude compatible avec la certification EDGE : "
        "réduction des consommations d'énergie, d'eau et d'énergie grise."
        if edge_compliant
        else "Le maître d'ouvrage ne demande pas explicitement la certification EDGE, mais le projet doit rester efficace."
    )

    user_prompt = f"""
Tu es un ingénieur MEPF & structure pour une application SaaS nommé Archito-Genie.

On te donne une description JSON du bâtiment (plan_spec) :

{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

Contexte EDGE :
{edge_text}

TA TÂCHE :
- Proposer un **BOQ de haut niveau** (quantités indicatives) avec **3 alternatives** :
  1. Economique (coût minimum, mais conforme)
  2. Standard (bon compromis qualité/prix)
  3. Premium (haut de gamme)

- Tu dois couvrir au minimum :
  - Structure (béton, acier, fondations, dalles, poteaux, poutres).
  - MEP :
    - Plomberie (EF/EC, EU/EV, sanitaires principaux, pompes si nécessaires).
    - Electricité courant fort (tableaux, câbles, prises, luminaires, groupes électrogènes, transformateur si besoin).
    - Courant faible (VDI/data, CCTV, contrôle d'accès, détection incendie).
    - CVC / ventilation si pertinent.
  - Automation & contrôle (capteurs, actionneurs, supervision légère si EDGE).

FORMAT DE SORTIE :
Tu DOIS renvoyer STRICTEMENT un JSON :

{{
  "currency": "XOF",
  "global_notes": "texte court sur les hypothèses",
  "edge_compliant": true | false,
  "options": [
    {{
      "name": "Economique",
      "description": "court texte",
      "items": [
        {{
          "category": "Structure | Plomberie | Electricité - Courant fort | Electricité - Courant faible | CVC | Automation | Finitions",
          "item": "Béton C25/30",
          "unit": "m3",
          "quantity": 120.0,
          "unit_price": 90000,
          "total_price": 10800000,
          "notes": "Fondations + voiles principaux"
        }}
      ]
    }},
    {{
      "name": "Standard",
      "description": "court texte",
      "items": [ ... ]
    }},
    {{
      "name": "Premium",
      "description": "court texte",
      "items": [ ... ]
    }}
  ]
}}

CONTRAINTES :
- EXACTEMENT 3 options : Economique, Standard, Premium.
- 10 à 40 lignes d'items par option.
- Toujours "XOF" comme monnaie.
"""
    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un ingénieur coût et études de prix. "
                    "Tu produis des BOQ synthétiques mais crédibles au format JSON strict, "
                    "avec une cohérence forte avec la structure et les systèmes MEPF & Automation."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    boq = json.loads(content)
    if "edge_compliant" not in boq:
        boq["edge_compliant"] = edge_compliant
    if "currency" not in boq:
        boq["currency"] = "XOF"
    return boq


def generate_datasheets_from_boq(plan_spec: Dict[str, Any], boq: Dict[str, Any], edge_compliant: bool) -> List[Dict[str, Any]]:
    user_prompt = f"""
Tu es un ingénieur technique (structure + MEPF + automation) pour une application SaaS.

On te donne :
1) Un descriptif JSON du bâtiment (plan_spec) :
{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

2) Un BOQ JSON avec 3 alternatives (Economique / Standard / Premium) :
{json.dumps(boq, indent=2, ensure_ascii=False)}

Contexte EDGE :
{"Le projet vise la certification EDGE, tu mets donc en avant les performances énergétiques et la réduction d'eau." if edge_compliant else "Le projet ne vise pas nécessairement la certification EDGE, mais doit rester techniquement solide."}

TA TÂCHE :
- Identifier les **matériaux et équipements clés** :
  - Structure (béton, acier, etc.)
  - Plomberie (tuyaux, pompes, appareils sanitaires)
  - Electricité courant fort (TGBT, disjoncteurs, câbles, luminaires)
  - Electricité courant faible (baies VDI, caméras CCTV, lecteurs badges, centrale incendie)
  - CVC / ventilation (si présent)
  - Automation / GTB (capteurs, contrôleurs, supervision légère)

- Pour chacun, produire une **fiche technique** synthétique mais crédible.
- Si EDGE = vrai, insiste sur les performances (U-value, rendement, débit réduit, etc.) quand c'est pertinent.

FORMAT DE SORTIE :
Tu DOIS renvoyer STRICTEMENT une LISTE JSON de fiches, par exemple :

[
  {{
    "material_name": "Béton C25/30",
    "category": "Structure",
    "used_in_options": ["Economique", "Standard"],
    "suitable_for": ["Fondations", "Voiles porteurs", "Dalles"],
    "key_specs": [
      "Résistance caractéristique à la compression 25 MPa à 28 jours",
      "Dosage ciment ~350 kg/m3",
      "Type de ciment : CEM II 42.5"
    ],
    "installation_notes": [
      "Vibration systématique au coulage",
      "Cure humide 7 jours minimum"
    ],
    "applicable_standards": [
      "Eurocode 2",
      "NF EN 206/C25/30"
    ],
    "edge_relevance": "Pas spécifique EDGE mais contribue à la durabilité",
    "manufacturer_hint": "Centrale à béton locale conforme EN 206"
  }}
]

CONTRAINTES :
- 10 à 30 fiches maximum.
- Les fiches doivent rester agnostiques (pas de marque imposée), mais avec 1 ou 2 suggestions possibles.
- Tu NE RENVOIES QUE la LISTE JSON (pas de texte autour).
"""
    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un ingénieur technique. "
                    "Tu produis des fiches techniques synthétiques et crédibles au format JSON strict."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    datasheets = json.loads(content)
    if isinstance(datasheets, dict):
        datasheets = [datasheets]
    return datasheets


# ============================================================
# AI : Narratifs MEPF & Automation
# ============================================================

def generate_mep_narratives(plan_spec: Dict[str, Any], boq: Dict[str, Any], edge_compliant: bool) -> Dict[str, str]:
    edge_text = (
        "Le projet vise une approche compatible EDGE (réduction énergie/eau)."
        if edge_compliant
        else "Le projet ne vise pas spécifiquement la certification EDGE, mais doit rester performant."
    )

    user_prompt = f"""
Tu es un ingénieur MEPF & Automation senior.

On te donne :
- Une description JSON du bâtiment (plan_spec) :
{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

- Un BOQ JSON avec 3 variantes (Economique / Standard / Premium) :
{json.dumps(boq, indent=2, ensure_ascii=False)}

Contexte EDGE :
{edge_text}

TA TÂCHE :
- Rédiger des **narratifs techniques** par lot, au format MARKDOWN,
  pour les lots suivants :
  - "power" (Electricité courant fort : TGBT, tableaux, câbles, prises, lighting, groupes électrogènes, transformateurs si besoin)
  - "low_current" (Courant faible : VDI/data, CCTV, contrôle d'accès, détection incendie, interphonie)
  - "plumbing" (Plomberie EF/EC/EU, pompes, surpression, appareils sanitaires, évacuation EU/EV)
  - "automation" (Capteurs, actionneurs, GTB légère / supervision, scénarios, liens avec EDGE)

STRUCTURE ATTENDUE (sortie JSON) :

{{
  "power_markdown": "# Narratif courant fort\\n...",
  "low_current_markdown": "# Narratif courant faible\\n...",
  "plumbing_markdown": "# Narratif plomberie\\n...",
  "automation_markdown": "# Narratif automation & GTB\\n..."
}}

CONTRAINTES :
- Chaque narratif doit tenir entre 0.5 et 2 pages A4 en texte.
- Style : ton professionnel BE, mais lisible par un promoteur.
- Tu dois mentionner, quand pertinent, les cohérences avec le BOQ et l'approche EDGE.
"""

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un ingénieur MEPF & Automation senior. "
                    "Tu écris des narratifs techniques structurés et cohérents avec les quantités et équipements."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    data = json.loads(content)

    return {
        "power_markdown": data.get("power_markdown", "# Narratif courant fort"),
        "low_current_markdown": data.get("low_current_markdown", "# Narratif courant faible"),
        "plumbing_markdown": data.get("plumbing_markdown", "# Narratif plomberie"),
        "automation_markdown": data.get("automation_markdown", "# Narratif automation & GTB"),
    }


# ============================================================
# Schematics SVG MEPF & Automation (diagrams fonctionnels)
# ============================================================

def render_power_schematic_svg(plan_spec: Dict[str, Any]) -> str:
    width, height = 900, 500
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    dwg.add(
        dwg.text(
            "SCHEMATIC COURANT FORT - ARCHITO-GENIE (TGBT, Groupes, Etages)",
            insert=(20, 40),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    x = 80
    y = 120
    box_w = 200
    box_h = 60

    dwg.add(
        dwg.rect(
            insert=(x, y),
            size=(box_w, box_h),
            fill="#f5f5f5",
            stroke="#000000",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Poste source\n+ Compteur",
            insert=(x + box_w / 2, y + 22),
            text_anchor="middle",
            font_size="11px",
            font_family="Arial",
        )
    )

    gen_x = x
    gen_y = y + 120

    dwg.add(
        dwg.rect(
            insert=(gen_x, gen_y),
            size=(box_w, box_h),
            fill="#fff3e0",
            stroke="#ef6c00",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Groupe électrogène",
            insert=(gen_x + box_w / 2, gen_y + 25),
            text_anchor="middle",
            font_size="11px",
            font_family="Arial",
        )
    )

    tgbt_x = x + 320
    tgbt_y = y + 60

    dwg.add(
        dwg.rect(
            insert=(tgbt_x, tgbt_y),
            size=(220, 90),
            fill="#e3f2fd",
            stroke="#1565c0",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "TGBT",
            insert=(tgbt_x + 110, tgbt_y + 30),
            text_anchor="middle",
            font_size="13px",
            font_family="Arial",
        )
    )
    dwg.add(
        dwg.text(
            "Arrivée réseau + GE\nSectionneur + jeux de barres",
            insert=(tgbt_x + 110, tgbt_y + 55),
            text_anchor="middle",
            font_size="10px",
            font_family="Arial",
        )
    )

    dwg.add(
        dwg.line(
            start=(x + box_w, y + box_h / 2),
            end=(tgbt_x, tgbt_y + box_h / 2),
            stroke="#000000",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.line(
            start=(gen_x + box_w, gen_y + box_h / 2),
            end=(tgbt_x, tgbt_y + box_h),
            stroke="#ef6c00",
            stroke_width=2,
        )
    )

    floors = plan_spec.get("floors", [])
    n_floors = len(floors) or 3

    start_y = 120
    step_y = 80
    dist_x = 280

    for i in range(n_floors):
        fy = start_y + i * step_y
        fx = tgbt_x + dist_x

        dwg.add(
            dwg.rect(
                insert=(fx, fy),
                size=(180, 50),
                fill="#e8f5e9",
                stroke="#2e7d32",
                stroke_width=2,
            )
        )
        label = floors[i]["name"] if i < len(floors) else f"Etage {i+1}"
        dwg.add(
            dwg.text(
                f"Tableau divisionnaire\n{label}",
                insert=(fx + 90, fy + 20),
                text_anchor="middle",
                font_size="10px",
                font_family="Arial",
            )
        )

        dwg.add(
            dwg.line(
                start=(tgbt_x + 220, tgbt_y + 45),
                end=(fx, fy + 25),
                stroke="#1565c0",
                stroke_width=2,
            )
        )

    return dwg.tostring()


def render_low_current_schematic_svg(plan_spec: Dict[str, Any]) -> str:
    width, height = 900, 500
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    dwg.add(
        dwg.text(
            "SCHEMATIC COURANT FAIBLE - VDI / CCTV / Contrôle d'accès / SSI",
            insert=(20, 40),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    core_x = 100
    core_y = 120
    core_w = 220
    core_h = 160

    dwg.add(
        dwg.rect(
            insert=(core_x, core_y),
            size=(core_w, core_h),
            fill="#f3e5f5",
            stroke="#6a1b9a",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Local technique CF\nBaie VDI + NVR + Centrale incendie\nContrôle d'accès",
            insert=(core_x + core_w / 2, core_y + 25),
            text_anchor="middle",
            font_size="11px",
            font_family="Arial",
        )
    )

    subsystems = [
        ("CCTV", "#ffcdd2", "#b71c1c"),
        ("Contrôle d'accès", "#c8e6c9", "#1b5e20"),
        ("VDI / DATA", "#bbdefb", "#0d47a1"),
        ("Détection incendie", "#ffe0b2", "#e65100"),
    ]

    base_x = 420
    base_y = 120
    box_w = 200
    box_h = 60
    step_y = 70

    for idx, (label, fill_color, stroke_color) in enumerate(subsystems):
        y = base_y + idx * step_y
        dwg.add(
            dwg.rect(
                insert=(base_x, y),
                size=(box_w, box_h),
                fill=fill_color,
                stroke=stroke_color,
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                label,
                insert=(base_x + box_w / 2, y + 25),
                text_anchor="middle",
                font_size="11px",
                font_family="Arial",
            )
        )
        dwg.add(
            dwg.line(
                start=(core_x + core_w, core_y + core_h / 2),
                end=(base_x, y + box_h / 2),
                stroke=stroke_color,
                stroke_width=2,
                stroke_dasharray="4,3",
            )
        )

    return dwg.tostring()


def render_plumbing_schematic_svg(plan_spec: Dict[str, Any]) -> str:
    width, height = 900, 500
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    dwg.add(
        dwg.text(
            "SCHEMATIC PLOMBERIE - EF / EC / EU / Pompes",
            insert=(20, 40),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    room_count = 0
    for f in plan_spec.get("floors", []):
        for r in f.get("rooms", []):
            if r.get("is_wet_area") or (r.get("usage") in ["kitchen", "sanitary"]):
                room_count += 1

    tank_x = 120
    tank_y = 120

    dwg.add(
        dwg.rect(
            insert=(tank_x, tank_y),
            size=(180, 80),
            fill="#e3f2fd",
            stroke="#1565c0",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Réservoir / Ballon ECS\n+ Groupe de pompage",
            insert=(tank_x + 90, tank_y + 25),
            text_anchor="middle",
            font_size="11px",
            font_family="Arial",
        )
    )

    stack_x = 400
    stack_y = 100
    stack_h = 260

    dwg.add(
        dwg.rect(
            insert=(stack_x, stack_y),
            size=(40, stack_h),
            fill="#e1f5fe",
            stroke="#0277bd",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Col. EF/EC\n+ Col. EU",
            insert=(stack_x + 20, stack_y + 20),
            text_anchor="middle",
            font_size="10px",
            font_family="Arial",
        )
    )

    dwg.add(
        dwg.line(
            start=(tank_x + 180, tank_y + 40),
            end=(stack_x, stack_y + stack_h / 2),
            stroke="#1565c0",
            stroke_width=3,
        )
    )

    start_y = 120
    step_y = 50
    draw_count = min(room_count, 5)

    for i in range(draw_count):
        y = start_y + i * step_y
        dwg.add(
            dwg.rect(
                insert=(stack_x + 100, y),
                size=(220, 35),
                fill="#f1f8e9",
                stroke="#689f38",
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                f"Bloc sanitaire {i+1}",
                insert=(stack_x + 210, y + 22),
                text_anchor="middle",
                font_size="10px",
                font_family="Arial",
            )
        )
        dwg.add(
            dwg.line(
                start=(stack_x + 40, stack_y + 40 + i * 30),
                end=(stack_x + 100, y + 17),
                stroke="#0277bd",
                stroke_width=2,
            )
        )

    return dwg.tostring()


def render_automation_schematic_svg(plan_spec: Dict[str, Any]) -> str:
    width, height = 900, 500
    dwg = svgwrite.Drawing(size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    dwg.add(
        dwg.text(
            "SCHEMATIC AUTOMATION & SENSORS - ARCHITO-GENIE",
            insert=(20, 40),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    bms_x = 120
    bms_y = 120

    dwg.add(
        dwg.rect(
            insert=(bms_x, bms_y),
            size=(220, 90),
            fill="#fffde7",
            stroke="#f9a825",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Contrôleur central / Mini-GTB\n+ Passerelle IP",
            insert=(bms_x + 110, bms_y + 25),
            text_anchor="middle",
            font_size="11px",
            font_family="Arial",
        )
    )

    nodes = [
        ("Capteurs de présence", "#e3f2fd", "#1565c0"),
        ("Capteurs de température", "#fce4ec", "#ad1457"),
        ("Capteurs fumée / CO", "#ffebee", "#c62828"),
        ("Actionneurs (éclairage, VR)", "#e8f5e9", "#2e7d32"),
    ]

    base_x = 430
    base_y = 120
    box_w = 220
    box_h = 60
    step_y = 70

    for idx, (label, fill_color, stroke_color) in enumerate(nodes):
        y = base_y + idx * step_y
        dwg.add(
            dwg.rect(
                insert=(base_x, y),
                size=(box_w, box_h),
                fill=fill_color,
                stroke=stroke_color,
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                label,
                insert=(base_x + box_w / 2, y + 25),
                text_anchor="middle",
                font_size="11px",
                font_family="Arial",
            )
        )
        dwg.add(
            dwg.line(
                start=(bms_x + 220, bms_y + 45),
                end=(base_x, y + box_h / 2),
                stroke=stroke_color,
                stroke_width=2,
                stroke_dasharray="4,3",
            )
        )

    return dwg.tostring()


# ============================================================
# Routes Projet
# ============================================================

@app.post("/projects", response_model=Project)
def create_project(payload: ProjectCreate):
    project_id = str(uuid.uuid4())
    edge_flag = bool(payload.edge_compliant) if payload.edge_compliant is not None else False
    project = Project(
        id=project_id,
        name=payload.name or "Unnamed project",
        created_at=datetime.utcnow(),
        edge_compliant=edge_flag,
    )
    PROJECTS[project_id] = project
    PROJECT_DATA[project_id] = {
        "files": {},
        "plan_spec": None,
        "report_markdown": "",
        "boq": None,
        "datasheets": None,
        "narratives": None,
        "schematics": {
            "power": None,
            "low_current": None,
            "plumbing": None,
            "automation": None,
        },
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


@app.post("/projects/{project_id}/edge", response_model=Project)
def update_project_edge(project_id: str, payload: EdgeUpdate):
    project = PROJECTS.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project.edge_compliant = payload.edge_compliant
    PROJECTS[project_id] = project
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

    arch_bytes = await architectural_plan.read()
    arch_key = f"{base_prefix}/architectural_plan_{architectural_plan.filename}"
    r2_put_bytes(
        arch_key,
        arch_bytes,
        architectural_plan.content_type or "application/pdf",
    )

    soil_key = None
    if soil_report is not None:
        soil_bytes = await soil_report.read()
        soil_key = f"{base_prefix}/soil_report_{soil_report.filename}"
        r2_put_bytes(soil_key, soil_bytes, soil_report.content_type or "application/pdf")

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
# Analyse & rapport global
# ============================================================

@app.post("/projects/{project_id}/analyze")
def analyze_project(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    project = PROJECTS[project_id]
    edge_compliant = project.edge_compliant

    files = PROJECT_DATA[project_id].get("files") or {}
    arch_key = files.get("architectural_plan_key")

    if not arch_key:
        raise HTTPException(
            status_code=400,
            detail="Architectural plan must be uploaded before analysis",
        )

    arch_bytes = r2_get_bytes(arch_key)

    plan_spec = get_plan_spec_from_ai(file_name=arch_key, file_bytes=arch_bytes, edge_compliant=edge_compliant)
    PROJECT_DATA[project_id]["plan_spec"] = plan_spec

    edge_label = "EDGE-compatible" if edge_compliant else "standard (sans certification explicite EDGE)"

    prompt = f"""
Tu es un assistant d'ingénierie pour un SaaS nommé Archito-Genie.

Projet : "{project.name}"
Profil énergétique : {edge_label}

Voici une description synthétique JSON de la distribution du bâtiment :

{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

Produit un rapport en **Markdown** avec les sections suivantes :

1. Description rapide du projet (basée sur le JSON)
2. Bloc Structure (3 à 7 puces concrètes)
3. Bloc MEPF (Plomberie, CF, CFaibles) & Automation (3 à 10 puces)
4. Bloc Durabilité & Efficacité énergétique (3 à 7 puces, mentionner explicitement EDGE si pertinent)
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
        "edge_compliant": edge_compliant,
        "has_plan_spec": bool(plan_spec),
    }


@app.get("/projects/{project_id}/report")
def get_report(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    md = PROJECT_DATA[project_id].get("report_markdown") or ""
    return {"project_id": project_id, "report_markdown": md}


# ============================================================
# BOQ & Datasheets endpoints
# ============================================================

@app.post("/projects/{project_id}/boq-and-datasheets")
def generate_boq_and_datasheets_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    plan_spec = PROJECT_DATA[project_id].get("plan_spec")
    if not plan_spec:
        raise HTTPException(
            status_code=400,
            detail="plan_spec not available. Call /projects/{project_id}/analyze first.",
        )

    edge_compliant = PROJECTS[project_id].edge_compliant

    try:
        boq = generate_boq_from_spec(plan_spec, edge_compliant=edge_compliant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating BOQ: {e}")

    PROJECT_DATA[project_id]["boq"] = boq

    try:
        datasheets = generate_datasheets_from_boq(plan_spec, boq, edge_compliant=edge_compliant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating datasheets: {e}")

    PROJECT_DATA[project_id]["datasheets"] = datasheets

    return {
        "project_id": project_id,
        "edge_compliant": edge_compliant,
        "has_boq": True,
        "has_datasheets": True,
    }


@app.get("/projects/{project_id}/boq")
def get_project_boq(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    boq = PROJECT_DATA[project_id].get("boq")
    if not boq:
        raise HTTPException(status_code=404, detail="BOQ not generated yet. Call /boq-and-datasheets first.")

    return {
        "project_id": project_id,
        "boq": boq,
    }


@app.get("/projects/{project_id}/datasheets")
def get_project_datasheets(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    datasheets = PROJECT_DATA[project_id].get("datasheets")
    if not datasheets:
        raise HTTPException(status_code=404, detail="Datasheets not generated yet. Call /boq-and-datasheets first.")

    return {
        "project_id": project_id,
        "datasheets": datasheets,
    }


# ============================================================
# Narratifs MEPF & Automation
# ============================================================

@app.post("/projects/{project_id}/narratives")
def generate_mep_narratives_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    plan_spec = PROJECT_DATA[project_id].get("plan_spec")
    if not plan_spec:
        raise HTTPException(
            status_code=400,
            detail="plan_spec not available. Call /projects/{project_id}/analyze first.",
        )

    boq = PROJECT_DATA[project_id].get("boq")
    if not boq:
        raise HTTPException(
            status_code=400,
            detail="BOQ not available. Call /projects/{project_id}/boq-and-datasheets first.",
        )

    edge_compliant = PROJECTS[project_id].edge_compliant

    narratives = generate_mep_narratives(plan_spec, boq, edge_compliant=edge_compliant)
    PROJECT_DATA[project_id]["narratives"] = narratives

    return {
        "project_id": project_id,
        "edge_compliant": edge_compliant,
        "has_narratives": True,
    }


@app.get("/projects/{project_id}/narratives")
def get_mep_narratives_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    narratives = PROJECT_DATA[project_id].get("narratives")
    if not narratives:
        raise HTTPException(status_code=404, detail="Narratives not generated yet. Call /narratives (POST) first.")

    return {
        "project_id": project_id,
        "narratives": narratives,
    }


# ============================================================
# Schematics MEPF & Automation (SVG)
# ============================================================

@app.post("/projects/{project_id}/schematics")
def generate_schematics_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    plan_spec = PROJECT_DATA[project_id].get("plan_spec")
    if not plan_spec:
        raise HTTPException(
            status_code=400,
            detail="plan_spec not available. Call /projects/{project_id}/analyze first.",
        )

    power_svg = render_power_schematic_svg(plan_spec)
    low_current_svg = render_low_current_schematic_svg(plan_spec)
    plumbing_svg = render_plumbing_schematic_svg(plan_spec)
    automation_svg = render_automation_schematic_svg(plan_spec)

    PROJECT_DATA[project_id]["schematics"] = {
        "power": power_svg,
        "low_current": low_current_svg,
        "plumbing": plumbing_svg,
        "automation": automation_svg,
    }

    return {
        "project_id": project_id,
        "has_schematics": True,
        "kinds": ["power", "low_current", "plumbing", "automation"],
    }


@app.get("/projects/{project_id}/schematics/{kind}.svg")
def get_schematic_svg_route(project_id: str, kind: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    schem = PROJECT_DATA[project_id].get("schematics") or {}
    svg = schem.get(kind)
    if not svg:
        raise HTTPException(status_code=404, detail="Schematics not generated yet or kind not found")

    stream = io.BytesIO(svg.encode("utf-8"))
    filename = f"{project_id}_schematic_{kind}.svg"

    return StreamingResponse(
        stream,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# Export rapport PDF & DOCX
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
# Healthcheck
# ============================================================

@app.get("/")
def healthcheck():
    return {
        "status": "ok",
        "message": "Archito-Genie backend (Narratifs MEPF + Schematics + BOQ + Datasheets + EDGE) is live",
    }

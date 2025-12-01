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

# Imports optionnels pour PDF/CAD à partir des SVG/plan_spec
try:
    import cairosvg  # pour SVG -> PDF
except ImportError:
    cairosvg = None

try:
    import ezdxf  # pour générer des DXF (CAD)
except ImportError:
    ezdxf = None


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


PROJECTS: Dict[str, Project] = {}
PROJECT_DATA: Dict[str, Dict] = {}


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
# Conversion SVG -> PDF
# ============================================================

def svg_to_pdf_bytes(svg_str: str) -> bytes:
    """
    Convertit un SVG (string) en PDF vectoriel.
    Nécessite la librairie cairosvg.
    """
    if cairosvg is None:
        raise HTTPException(
            status_code=500,
            detail="La conversion SVG->PDF nécessite 'cairosvg'. Ajoute-le à requirements.txt."
        )
    pdf_bytes = cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"))
    return pdf_bytes


# ============================================================
# App FastAPI
# ============================================================

app = FastAPI(
    title="Archito-Genie Backend",
    description=(
        "MVP Archito-Genie : plans STRUCTURE / MEP (SVG, PDF, DXF) + BOQ + datasheets + "
        "option EDGE (sans APS pour l'instant)."
    ),
    version="0.7.0",
)


# ============================================================
# Mise en page des pièces (pour STRUCTURE & MEP)
# ============================================================

def _layout_rooms(rooms: List[Dict[str, Any]], scale: float = 40.0):
    """
    Organise les pièces en grille simple pour un rendu MVP lisible.
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
# Rendu SVG : PLAN STRUCTURE
# ============================================================

def render_structure_svg(spec: Dict[str, Any]) -> str:
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

    dwg.add(
        dwg.text(
            f"PLAN STRUCTUREL - {floor.get('name', 'Niveau 0')} - ARCHITO-GENIE",
            insert=(margin, margin - 25),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    for p in layout_positions:
        x = margin + p["x_m"] * scale
        y = margin + p["y_m"] * scale
        w = p["w_m"] * scale
        h = p["l_m"] * scale
        room = p["room"]

        # Contour de la pièce
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

        # Poteaux aux 4 coins
        col_size = 12
        for dx, dy in [(0, 0), (w - col_size, 0), (0, h - col_size), (w - col_size, h - col_size)]:
            dwg.add(
                dwg.rect(
                    insert=(x + dx, y + dy),
                    size=(col_size, col_size),
                    fill="black",
                )
            )

    # Trame indicative horizontale
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


# ============================================================
# Rendu SVG : PLAN MEP (CF / CFa / Plomberie)
# ============================================================

def render_mep_svg(spec: Dict[str, Any]) -> str:
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

    dwg.add(
        dwg.text(
            f"PLAN MEP - {floor.get('name', 'Niveau 0')} - ARCHITO-GENIE",
            insert=(margin, margin - 25),
            font_size="18px",
            font_family="Arial",
            fill="black",
        )
    )

    # Contours & noms
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

    # Noyau technique simplifié
    core_x = margin + total_w_m * scale + 20
    core_y = margin
    core_w = 80
    core_h = 220

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

    # Colonne EU/EV (plomberie)
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

    # Tableau électrique (courant fort)
    dwg.add(
        dwg.rect(
            insert=(core_x + 10, core_y + 30),
            size=(40, 25),
            fill="none",
            stroke="#ff3333",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "TGBT",
            insert=(core_x + 30, core_y + 47),
            text_anchor="middle",
            font_size="9px",
            font_family="Arial",
            fill="#ff3333",
        )
    )

    # Baie courant faible (VDI / data)
    dwg.add(
        dwg.rect(
            insert=(core_x + 10, core_y + 70),
            size=(40, 25),
            fill="none",
            stroke="#9c27b0",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "VDI",
            insert=(core_x + 30, core_y + 87),
            text_anchor="middle",
            font_size="9px",
            font_family="Arial",
            fill="#9c27b0",
        )
    )

    # Réseaux dans les pièces
    for p in layout_positions:
        x = margin + p["x_m"] * scale
        y = margin + p["y_m"] * scale
        w = p["w_m"] * scale
        h = p["l_m"] * scale
        room = p["room"]

        name = (room.get("name") or "").lower()
        is_wet = room.get("is_wet_area", False)

        # Plomberie (sanitaires / points d'eau)
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

        # Courant fort : prise
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
                "PC",
                insert=(x + 20, y + 14),
                font_size="8px",
                font_family="Arial",
                fill="#ff3333",
            )
        )

        # Courant faible : prise RJ45 / data
        dwg.add(
            dwg.rect(
                insert=(x + 5, y + 22),
                size=(10, 10),
                fill="none",
                stroke="#9c27b0",
                stroke_width=2,
            )
        )
        dwg.add(
            dwg.text(
                "RJ",
                insert=(x + 20, y + 31),
                font_size="8px",
                font_family="Arial",
                fill="#9c27b0",
            )
        )

        # Luminaire
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

    # Légende
    legend_x = margin
    legend_y = height_px - 130
    legend_w = 360
    legend_h = 110

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

    # Courant fort
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
            "Prise courant fort (PC)",
            insert=(legend_x + 30, legend_y + 79),
            font_size="10px",
            font_family="Arial",
        )
    )

    # Courant faible
    dwg.add(
        dwg.rect(
            insert=(legend_x + 10, legend_y + 90),
            size=(10, 10),
            fill="none",
            stroke="#9c27b0",
            stroke_width=2,
        )
    )
    dwg.add(
        dwg.text(
            "Prise courant faible / data (RJ45)",
            insert=(legend_x + 30, legend_y + 99),
            font_size="10px",
            font_family="Arial",
        )
    )

    return dwg.tostring()


# ============================================================
# Génération DXF (CAD) STRUCTURE & MEP
# ============================================================

def render_structure_dxf(plan_spec: Dict[str, Any]) -> bytes:
    """
    Génère un DXF simplifié pour le plan STRUCTURE :
    - Contours de pièces
    - Noms de pièces
    - Poteaux en coins
    """
    if ezdxf is None:
        raise HTTPException(
            status_code=500,
            detail="La génération CAD (DXF) nécessite 'ezdxf'. Ajoute-le à requirements.txt."
        )

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    scale = 1000.0  # 1 m = 1000 unités DXF
    floors = plan_spec.get("floors", [])
    floor = floors[0] if floors else {"name": "Niveau 0", "rooms": []}
    rooms = floor.get("rooms", [])

    layout_positions, total_w_m, total_h_m = _layout_rooms(rooms, scale=1.0)

    for p in layout_positions:
        x_m = p["x_m"]
        y_m = p["y_m"]
        w_m = p["w_m"]
        l_m = p["l_m"]
        room = p["room"]

        x1 = x_m * scale
        y1 = y_m * scale
        x2 = (x_m + w_m) * scale
        y2 = (y_m + l_m) * scale

        # Contour
        msp.add_lwpolyline(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)],
            dxfattribs={"layer": "STRUCTURE_ROOMS"},
        )

        # Poteaux simples aux 4 coins
        col_size = 0.25 * scale  # 0.25 m
        for (cx, cy) in [
            (x1, y1),
            (x2 - col_size, y1),
            (x1, y2 - col_size),
            (x2 - col_size, y2 - col_size),
        ]:
            msp.add_solid(
                [(cx, cy), (cx + col_size, cy), (cx + col_size, cy + col_size), (cx, cy + col_size)],
                dxfattribs={"layer": "STRUCTURE_COLUMNS"},
            )

        # Nom de pièce
        msp.add_text(
            room.get("name", "Pièce"),
            dxfattribs={"height": 0.3 * scale, "layer": "ANNOTATIONS"},
        ).set_pos(((x1 + x2) / 2, (y1 + y2) / 2), align="MIDDLE_CENTER")

    buf = io.BytesIO()
    doc.write(stream=buf)
    buf.seek(0)
    return buf.getvalue()


def render_mep_dxf(plan_spec: Dict[str, Any]) -> bytes:
    """
    Génère un DXF simplifié pour le MEP :
    - Contours de pièces
    - Points d'eau / évacuations
    - Prises CF/CFa
    """
    if ezdxf is None:
        raise HTTPException(
            status_code=500,
            detail="La génération CAD (DXF) nécessite 'ezdxf'. Ajoute-le à requirements.txt."
        )

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    scale = 1000.0  # 1 m = 1000 unités DXF
    floors = plan_spec.get("floors", [])
    floor = floors[0] if floors else {"name": "Niveau 0", "rooms": []}
    rooms = floor.get("rooms", [])

    layout_positions, total_w_m, total_h_m = _layout_rooms(rooms, scale=1.0)

    for p in layout_positions:
        x_m = p["x_m"]
        y_m = p["y_m"]
        w_m = p["w_m"]
        l_m = p["l_m"]
        room = p["room"]

        x1 = x_m * scale
        y1 = y_m * scale
        x2 = (x_m + w_m) * scale
        y2 = (y_m + l_m) * scale

        # Contour
        msp.add_lwpolyline(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)],
            dxfattribs={"layer": "MEP_ROOMS"},
        )

        name = (room.get("name") or "").lower()
        is_wet = room.get("is_wet_area", False)

        # Plomberie (pièces humides)
        if is_wet or any(k in name for k in ["bain", "sdb", "wc", "toilet", "cuisine", "kitchen"]):
            px = x1 + 0.5 * scale
            py = y1 + 0.5 * scale

            # Point d'eau = cercle
            msp.add_circle((px, py), radius=0.15 * scale, dxfattribs={"layer": "MEP_PLOMBERIE"})

            # Evacuation = triangle
            msp.add_solid(
                [
                    (px + 0.3 * scale, py),
                    (px + 0.5 * scale, py),
                    (px + 0.4 * scale, py + 0.2 * scale),
                ],
                dxfattribs={"layer": "MEP_PLOMBERIE"},
            )

        # Prise courant fort (symbole simple)
        pcx = x1 + 0.2 * scale
        pcy = y1 + 0.2 * scale
        msp.add_lwpolyline(
            [
                (pcx, pcy),
                (pcx + 0.2 * scale, pcy),
                (pcx + 0.2 * scale, pcy + 0.2 * scale),
                (pcx, pcy + 0.2 * scale),
                (pcx, pcy),
            ],
            dxfattribs={"layer": "MEP_CF"},
        )

        # Prise courant faible / data
        rjx = x1 + 0.2 * scale
        rjy = y1 + 0.6 * scale
        msp.add_lwpolyline(
            [
                (rjx, rjy),
                (rjx + 0.2 * scale, rjy),
                (rjx + 0.2 * scale, rjy + 0.2 * scale),
                (rjx, rjy + 0.2 * scale),
                (rjx, rjy),
            ],
            dxfattribs={"layer": "MEP_CFA"},
        )

    buf = io.BytesIO()
    doc.write(stream=buf)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Extraction plan_spec via OpenAI (avec EDGE ou non)
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
  résidentiels/mixtes.
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
                        "Tu produis une description JSON cohérente de la distribution des pièces."
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
                        {"name": "Séjour", "width_m": 5.0, "length_m": 6.0, "has_window": True, "is_wet_area": False},
                        {"name": "Cuisine", "width_m": 3.0, "length_m": 4.0, "has_window": True, "is_wet_area": True},
                        {"name": "Chambre 1", "width_m": 4.0, "length_m": 4.0, "has_window": True, "is_wet_area": False},
                        {"name": "Chambre 2", "width_m": 3.5, "length_m": 3.5, "has_window": True, "is_wet_area": False},
                        {"name": "Salle de bain", "width_m": 3.0, "length_m": 3.0, "has_window": False, "is_wet_area": True},
                    ],
                }
            ],
        }


# ============================================================
# BOQ (3 alternatives) & Datasheets via OpenAI (EDGE-aware)
# ============================================================

def generate_boq_from_spec(plan_spec: Dict[str, Any], edge_compliant: bool) -> Dict[str, Any]:
    edge_text = (
        "Le maître d'ouvrage souhaite une étude compatible avec la certification EDGE : "
        "réduction des consommations d'énergie, d'eau et d'énergie grise."
        if edge_compliant
        else "Le maître d'ouvrage ne demande pas explicitement la certification EDGE, mais le projet doit rester efficace."
    )

    user_prompt = f"""
Tu es un ingénieur structure & MEP pour une application SaaS nommé Archito-Genie.

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
    - Plomberie (EF/EC, EU/EV, sanitaires principaux).
    - Electricité courant fort (tableaux, câbles, prises, luminaires).
    - Courant faible (VDI / data de base).
    - CVC / ventilation si pertinent.
  - Finitions principales si nécessaire.

- Si EDGE = vrai, privilégie :
  - Luminaires LED, appareillages économes en eau,
  - Solutions passives / isolation pertinentes.

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
          "category": "Structure | Plomberie | Electricité - Courant fort | Electricité - Courant faible | CVC | Finitions",
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
                    "Tu produis des BOQ synthétiques mais crédibles au format JSON strict."
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
Tu es un ingénieur technique (structure + MEP) pour une application SaaS.

On te donne :
1) Un descriptif JSON du bâtiment (plan_spec) :
{json.dumps(plan_spec, indent=2, ensure_ascii=False)}

2) Un BOQ JSON avec 3 alternatives (Economique / Standard / Premium) :
{json.dumps(boq, indent=2, ensure_ascii=False)}

Contexte EDGE :
{"Le projet vise la certification EDGE, tu mets donc en avant les performances énergétiques et la réduction d'eau." if edge_compliant else "Le projet ne vise pas nécessairement la certification EDGE, mais doit rester techniquement solide."}

TA TÂCHE :
- Identifier les **matériaux et équipements clés** (béton, acier, conduites, câbles, appareillages, luminaires types, équipements CVC, etc.).
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
# Routes Projet
# ============================================================

class EdgeUpdate(BaseModel):
    edge_compliant: bool


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
        "structure_svg": "",
        "mep_svg": "",
        "report_markdown": "",
        "boq": None,
        "datasheets": None,
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
# Analyse & génération des plans + rapport
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

    # 1) plan_spec via OpenAI
    plan_spec = get_plan_spec_from_ai(file_name=arch_key, file_bytes=arch_bytes, edge_compliant=edge_compliant)
    PROJECT_DATA[project_id]["plan_spec"] = plan_spec

    # 2) Plans STRUCTURE & MEP (SVG)
    structure_svg = render_structure_svg(plan_spec)
    mep_svg = render_mep_svg(plan_spec)
    PROJECT_DATA[project_id]["structure_svg"] = structure_svg
    PROJECT_DATA[project_id]["mep_svg"] = mep_svg

    # 3) Rapport Markdown
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
3. Bloc MEPF & Automation (3 à 7 puces, inclure courant fort, courant faible et plomberie)
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
# Téléchargement plans STRUCTURE & MEP : SVG, PDF, DXF
# ============================================================

@app.get("/projects/{project_id}/plans/structure.svg")
def get_structure_svg_route(project_id: str):
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
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.get("/projects/{project_id}/plans/mep.svg")
def get_mep_svg_route(project_id: str):
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
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


# Alias de compatibilité pour ton ancien front
@app.get("/projects/{project_id}/schematics/svg")
def get_schematics_svg_route(project_id: str):
    return get_structure_svg_route(project_id)


@app.get("/projects/{project_id}/plans/structure.pdf")
def get_structure_pdf_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("structure_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="Structure plan not generated yet. Call /analyze first.")

    pdf_bytes = svg_to_pdf_bytes(svg)
    stream = io.BytesIO(pdf_bytes)
    filename = f"{project_id}_plan_structure.pdf"

    return StreamingResponse(
        stream,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.get("/projects/{project_id}/plans/mep.pdf")
def get_mep_pdf_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    svg = PROJECT_DATA[project_id].get("mep_svg")
    if not svg:
        raise HTTPException(status_code=404, detail="MEP plan not generated yet. Call /analyze first.")

    pdf_bytes = svg_to_pdf_bytes(svg)
    stream = io.BytesIO(pdf_bytes)
    filename = f"{project_id}_plan_mep.pdf"

    return StreamingResponse(
        stream,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.get("/projects/{project_id}/plans/structure.dxf")
def get_structure_dxf_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    plan_spec = PROJECT_DATA[project_id].get("plan_spec")
    if not plan_spec:
        raise HTTPException(status_code=400, detail="plan_spec not available. Call /analyze first.")

    dxf_bytes = render_structure_dxf(plan_spec)
    stream = io.BytesIO(dxf_bytes)
    filename = f"{project_id}_plan_structure.dxf"

    return StreamingResponse(
        stream,
        media_type="application/dxf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.get("/projects/{project_id}/plans/mep.dxf")
def get_mep_dxf_route(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")

    plan_spec = PROJECT_DATA[project_id].get("plan_spec")
    if not plan_spec:
        raise HTTPException(status_code=400, detail="plan_spec not available. Call /analyze first.")

    dxf_bytes = render_mep_dxf(plan_spec)
    stream = io.BytesIO(dxf_bytes)
    filename = f"{project_id}_plan_mep.dxf"

    return StreamingResponse(
        stream,
        media_type="application/dxf",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


# ============================================================
# BOQ & Datasheets endpoints
# ============================================================

@app.post("/projects/{project_id}/boq-and-datasheets")
def generate_boq_and_datasheets(project_id: str):
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
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
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
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


# ============================================================
# Healthcheck
# ============================================================

@app.get("/")
def healthcheck():
    return {
        "status": "ok",
        "message": "Archito-Genie backend MVP (plans SVG/PDF/DXF + BOQ + datasheets + EDGE) is live",
    }

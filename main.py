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
    total_height_m = max((p["y_m"] + p["l_m"]) for p in layout_positions) if layout_p_]()_]()

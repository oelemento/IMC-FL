#!/usr/bin/env python3
"""Generate conceptual model figure: dual-compartment immune evasion in FL.

Fresh generation from scratch — no reference image.
Inspired by ABM spatial snapshots: clean concentric layout with colored cells.
"""

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """Create a scientific schematic figure for a Nature-style immunology paper.
The figure shows the "dual-compartment immune evasion model" in follicular lymphoma.

OVERALL LAYOUT: A single large neoplastic follicle shown in cross-section, centered
in the image. Four concentric zones from inside out, each with a distinct background
color. The follicle is roughly circular. Wide 16:9 format. Legend on the right side.
Clean, flat, publication-quality — like a Cell or Nature Reviews Immunology diagram.
White background outside the outermost zone.

THE FOUR CONCENTRIC ZONES (inside → out):

ZONE 1 — "Activated FDC network" (innermost core):
- Background: warm tan/sandy (#F5DEB3)
- This is the KEY visual feature of the figure.
- CD14+ FDCs (gold/orange #DAA520): Draw as STELLATE cells with 5-7 long, thin,
  branching dendritic processes. They interconnect to form a visible MESHWORK — like
  a web or neural network. Their processes should touch and overlap. This looks
  organic, like tree roots or neurons, NOT simple circles. About 8-10 FDCs forming
  this network.
- M2 macrophages (dark red #8B0000 circles): 3-4 scattered among the FDC processes.
- Exhausted CD8 T cells (dark navy #1B2A4A circles): 2-3 only, rare, trapped in the
  meshwork. These are the few CD8 T cells that penetrated the follicle — they are
  exhausted.
- Interaction arrows in this zone:
  * Red flat-headed inhibition arrow from FDC → exhausted CD8 T, labeled "VISTA, IDO"
  * Red flat-headed inhibition arrow from M2 Mac → exhausted CD8 T, labeled "VISTA"

ZONE 2 — "B cell zone" (ring around the FDC network):
- Background: light green (#C8E6C0)
- Filled with BCL2+ tumor B cells (dark green #2E7D32 circles): dense, many cells
  forming a thick ring. These are the neoplastic B cells.
- A few CD4 T / Tfh cells (olive #6B8E23 circles): scattered among B cells.
- Interaction arrow:
  * Green arrow labeled "Survival signals". The arrow TAIL/START is at an FDC cell
    (gold, in zone 1) and the arrow HEAD/TIP points OUTWARD toward a tumor B cell
    (dark green, in zone 2). The arrowhead touches the B cell. The direction is
    FROM INSIDE (FDC network) TO OUTSIDE (B cell zone) — centrifugal direction.

ZONE 3 — "Treg barrier" (thin ring at the follicle boundary):
- Background: light gold (#FFF3B0), semi-transparent
- Treg cells (yellow-green #9ACD32 circles): forming a visible ring/belt. Dense
  enough to look like a barrier between follicular and interfollicular.
- No statistics or numbers in this zone.

ZONE 4 — "Interfollicular zone" (outermost, surrounding the follicle):
- Background: light blue-gray (#D6E4F0)
- Effector CD8 T cells (bright blue #4169E1 circles): numerous, the dominant
  population here.
- S100A9+ MDSCs (magenta/hot pink #C71585 circles): scattered among effector T cells.
- Interaction arrow:
  * Red flat-headed inhibition arrow from ONE MDSC → ONE nearby effector CD8 T cell,
    labeled "VISTA". The arrow must go from a magenta MDSC cell to a blue effector
    CD8 T cell. NOT from MDSC to MDSC.

ARROW RULES — CRITICAL — READ CAREFULLY:
- Every arrow has ONE arrowhead at the TARGET end only. NO arrowheads at the source.
- NEVER draw bidirectional arrows or double-headed arrows. Only single-headed arrows.
- Each arrow connects exactly TWO DIFFERENT cell types (never same type to same type).
- The arrowhead (or flat inhibition head) ALWAYS points at the TARGET cell.
- There are exactly 4 arrows total:
  1. FDC → exhausted CD8 T: red line, flat inhibition head touching the CD8 T cell, labeled "VISTA, IDO"
  2. M2 Mac → exhausted CD8 T: red line, flat inhibition head touching the CD8 T cell, labeled "VISTA"
  3. FDC → tumor B cell: green line, arrowhead touching the B cell, labeled "Survival signals"
  4. MDSC → effector CD8 T: red line starting at a MAGENTA MDSC cell, flat inhibition
     head touching a BLUE effector CD8 T cell, labeled "VISTA". The MDSC is the SOURCE
     (no arrowhead on the MDSC end), the effector CD8 T is the TARGET (inhibition head
     on the CD8 T end). Direction: MDSC suppresses CD8 T, not the other way around.

DO NOT include any statistics or percentages on the figure. No numbers, no percentages,
no ratios. Zone labels and interaction arrow labels only.

LEGEND (right side, clean vertical list):
Each entry uses the SAME shape as in the diagram. Consistency is critical.
- CD14+ FDC (gold stellate icon — same shape as in diagram)
- BCL2+ tumor B (dark green circle)
- M2 Mac (dark red circle — same round shape as in diagram, NOT a different shape)
- Exhausted CD8 T (dark navy)
- Effector CD8 T (bright blue)
- CD4 T / Tfh (olive)
- Treg (yellow-green)
- S100A9+ MDSC (magenta)
Arrow key at bottom of legend:
- Red flat-head = Suppression
- Green arrow = Support

STYLE REQUIREMENTS:
- Nature Reviews Immunology style. Clean, professional, elegant.
- Flat colors, no gradients, no shadows, no 3D, no photorealism.
- Thin dark outlines on all cells.
- Zone labels in clean sans-serif font, slightly bold.
- The FDC meshwork in the center is the visual centerpiece — it should look
  strikingly different from all other cells (which are simple circles).
- Muted, harmonious color palette. The zones should be distinguishable by their
  background tint but not garish.
"""

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[prompt],
    config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE'],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="2K"
        ),
    ),
)

for part in response.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        image = part.as_image()
        out = "output/hypotheses_v8/fig_model_immune_evasion.jpg"
        image.save(out)
        # Convert to PNG
        from PIL import Image as PILImage
        img = PILImage.open(out)
        png_out = out.replace(".jpg", ".png")
        img.save(png_out, format="PNG")
        print(f"Saved: {png_out}")

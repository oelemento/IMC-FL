#!/usr/bin/env python3
"""Generate concept cartoons for the 4 spatial crosstalk analyses."""

import os
from PIL import Image as PILImage
from io import BytesIO
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
out_dir = "output/hypothesis_cartoons"
os.makedirs(out_dir, exist_ok=True)

CARTOONS = {
    "spatial_nhood_enrichment": {
        "file": "spatial_nhood_enrichment.png",
        "prompt": (
            "Scientific concept diagram for a research paper, clean minimal style with soft colors. "
            "Title: 'Neighborhood Enrichment Analysis'. "
            "Show a tissue cross-section with colored dots representing different cell types "
            "(blue=B cells, red=T cells, green=macrophages, purple=regulatory T cells). "
            "On the left, show a cell with dashed circle around its k=15 nearest neighbors, "
            "with arrows pointing to the neighbors. "
            "On the right, show a small heatmap matrix labeled 'Enrichment Z-scores' with rows "
            "and columns for cell types, colored red (enriched) and blue (depleted). "
            "Below, show a small diagram of 'Observed vs Permuted' with a histogram and an arrow "
            "pointing to where the observed value falls. "
            "Style: publication figure, white background, clean lines, labeled annotations. "
            "No photograph, illustration only."
        ),
    },
    "spatial_cellular_neighborhoods": {
        "file": "spatial_cellular_neighborhoods.png",
        "prompt": (
            "Scientific concept diagram for a research paper, clean minimal style with soft colors. "
            "Title: 'Cellular Neighborhood Discovery'. "
            "Three-step workflow shown left to right with arrows between steps: "
            "Step 1 (left): A tissue section with colored cells. One cell highlighted with a circle "
            "showing its k=20 nearest neighbors. An arrow points to a small bar chart showing "
            "the neighbor composition (fraction of each cell type). "
            "Step 2 (middle): Many such bar charts (composition vectors) flowing into a clustering "
            "algorithm (shown as a funnel or grouping icon labeled 'KMeans'). "
            "Step 3 (right): The same tissue section but now cells are colored by their assigned "
            "cellular neighborhood cluster (CN1, CN2, CN3 etc.), showing spatial domains of "
            "recurring multicellular motifs. "
            "Style: publication figure, white background, clean lines, labeled annotations. "
            "No photograph, illustration only."
        ),
    },
    "spatial_functional_proximity": {
        "file": "spatial_functional_proximity.png",
        "prompt": (
            "Scientific concept diagram for a research paper, clean minimal style with soft colors. "
            "Title: 'Exhaustion × Spatial Proximity'. "
            "Show a tissue section with CD8 T cells (red dots) surrounded by different neighbor types: "
            "B cells (blue), Tregs (purple), macrophages (green). "
            "Two CD8 T cells are highlighted: one labeled 'Exhausted (TOX+, PD-1+)' surrounded by "
            "many B cells and Tregs, and another labeled 'Non-exhausted' surrounded by CD4 T cells. "
            "An arrow from each points to a bar showing their neighbor composition. "
            "On the right side, show a scatter plot concept with 'B cell proximity' on x-axis and "
            "'TOX expression' on y-axis with a positive correlation trend line. "
            "Key message: spatial context determines exhaustion state. "
            "Style: publication figure, white background, clean lines, labeled annotations. "
            "No photograph, illustration only."
        ),
    },
    "spatial_covariation_boundary": {
        "file": "spatial_covariation_boundary.png",
        "prompt": (
            "Scientific concept diagram for a research paper, clean minimal style with soft colors. "
            "Title: 'Cell Type Co-variation & Boundary Analysis'. "
            "Two panels side by side: "
            "Left panel: A network graph with 6 nodes representing cell types (B cells, T cells, "
            "macrophages, Tregs, etc.) connected by red lines (positive correlation) and blue lines "
            "(negative correlation). Thicker lines = stronger correlation. Label: 'Co-variation network'. "
            "Right panel: A tissue section divided into two zones - a pink 'Follicular' zone (mostly "
            "blue B cells) and a light blue 'Interfollicular' zone (mostly red T cells). "
            "At the boundary between zones, cells are highlighted in yellow/gold, labeled 'Boundary cells'. "
            "An arrow points to a bar chart showing which cell types are enriched at boundaries. "
            "Style: publication figure, white background, clean lines, labeled annotations. "
            "No photograph, illustration only."
        ),
    },
}


def generate_one(key, info):
    out_path = os.path.join(out_dir, info["file"])
    if os.path.exists(out_path):
        print(f"  [{key}] Already exists: {out_path}")
        return

    print(f"  [{key}] Generating...")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[info["prompt"]],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="3:2",
                    image_size="2K",
                ),
            ),
        )

        for part in response.parts:
            if part.inline_data:
                # Save as JPEG first, then convert to PNG via PIL
                tmp_jpg = out_path.replace(".png", ".jpg")
                part.as_image().save(tmp_jpg)
                PILImage.open(tmp_jpg).save(out_path, format="PNG")
                os.remove(tmp_jpg)
                print(f"  [{key}] Saved: {out_path}")
                return

        print(f"  [{key}] WARNING: No image in response")
        for part in response.parts:
            if part.text:
                print(f"    Text: {part.text[:200]}")

    except Exception as e:
        print(f"  [{key}] ERROR: {e}")


if __name__ == "__main__":
    print("Generating spatial crosstalk concept cartoons...")
    for key, info in CARTOONS.items():
        generate_one(key, info)
    print("\nDone.")

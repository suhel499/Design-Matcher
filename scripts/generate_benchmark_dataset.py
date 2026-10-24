"""
Generates a realistic benchmark folder with 1,000+ mixed CAD & PDF files:
- 500 DXF drawings (with multiple layers, entities, text annotations)
- 300 DWG drawings (AutoCAD binary headers: AC1032, AC1027, AC1024, AC1021)
- 200 PDF documents (vector text, panels, title blocks)
- 50 Junk/temp files (~$temp.dwg, .DS_Store, Thumbs.db, backup.bak, file.dwl)
Organized into a multi-level directory hierarchy.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_minimal_dxf(filepath: Path, filename_stem: str, index: int) -> None:
    """Writes a standard ASCII DXF R2010 file with HEADER, TABLES (LAYERS), and ENTITIES."""
    layer1 = f"LAYER_FRAME_{index % 5}"
    layer2 = f"LAYER_CUT_{index % 10}"
    dxf_content = f"""  0
SECTION
  2
HEADER
  9
$ACADVER
  1
AC1024
  0
ENDSEC
  0
SECTION
  2
TABLES
  0
TABLE
  2
LAYER
 70
2
  0
LAYER
  2
{layer1}
 70
0
 62
1
  6
CONTINUOUS
  0
LAYER
  2
{layer2}
 70
0
 62
2
  6
CONTINUOUS
  0
ENDTAB
  0
ENDSEC
  0
SECTION
  2
ENTITIES
  0
LINE
  8
{layer1}
 10
0.0
 20
0.0
 30
0.0
 11
100.0
 21
100.0
 31
0.0
  0
CIRCLE
  8
{layer2}
 10
50.0
 20
50.0
 30
0.0
 40
25.0
  0
TEXT
  8
{layer2}
 10
10.0
 20
10.0
 30
0.0
 40
5.0
  1
PATTERN-DESIGN-{filename_stem}-{index}
  0
ENDSEC
  0
EOF
"""
    filepath.write_text(dxf_content, encoding="ascii")


def create_minimal_dwg(filepath: Path, version_header: bytes = b"AC1032") -> None:
    """Writes a synthetic DWG file with authentic AutoCAD magic header and binary payload."""
    padding = os.urandom(2048)  # 2KB binary payload
    filepath.write_bytes(version_header + b"\x00" * 10 + padding)


def create_minimal_pdf(filepath: Path, title: str, pages: int = 1) -> None:
    """Creates a compact multi-page PDF with raster/text content using Pillow."""
    imgs = []
    for p in range(pages):
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 580, 780], outline=(40, 40, 40), width=3)
        draw.text((40, 50), f"Laser Cut Catalogue Design: {title} - Page {p+1}", fill=(0, 0, 0))
        draw.text((40, 90), f"Category: Architectural Lattice Grills & Screens", fill=(50, 50, 50))
        # Draw some decorative shapes
        for y in range(150, 700, 100):
            draw.rectangle([60, y, 540, y + 60], outline=(100, 100, 100), width=2)
            draw.text((80, y + 20), f"Code: LATTICE-{p}-{y}", fill=(0, 0, 120))
        imgs.append(img)
    imgs[0].save(str(filepath), save_all=True, append_images=imgs[1:])


def generate_benchmark_folder(target_dir: Path | str, total_files: int = 1000) -> Path:
    out = Path(target_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    categories = [
        "Architectural_Panels",
        "Balcony_Railings",
        "Decorative_Screens",
        "Gates_And_Fences",
        "Interior_Partitions",
        "Jali_Cutting",
        "Ventilation_Grilles",
        "Window_Security",
    ]

    for cat in categories:
        (out / cat / "Sub_A").mkdir(parents=True, exist_ok=True)
        (out / cat / "Sub_B").mkdir(parents=True, exist_ok=True)

    folders = [out] + [out / cat for cat in categories] + [
        out / cat / sub for cat in categories for sub in ("Sub_A", "Sub_B")
    ]

    # Target: 500 DXF, 300 DWG, 200 PDF = 1000 files
    # + 50 temp files
    dxf_count = 500
    dwg_count = 300
    pdf_count = 200

    print(f"Generating {dxf_count} DXF, {dwg_count} DWG, {pdf_count} PDF files in {out}...")

    # 1. DXFs
    for i in range(dxf_count):
        folder = folders[i % len(folders)]
        fname = f"panel_design_{i:04d}.dxf"
        create_minimal_dxf(folder / fname, f"panel_{i:04d}", i)

    # 2. DWGs
    dwg_headers = [b"AC1032", b"AC1027", b"AC1024", b"AC1021", b"AC1018"]
    for i in range(dwg_count):
        folder = folders[(i + 3) % len(folders)]
        fname = f"cad_drawing_{i:04d}.dwg"
        header = dwg_headers[i % len(dwg_headers)]
        create_minimal_dwg(folder / fname, version_header=header)

    # 3. PDFs
    for i in range(pdf_count):
        folder = folders[(i + 7) % len(folders)]
        fname = f"screen_catalog_vol_{i:04d}.pdf"
        pages = 1 if (i % 3 != 0) else 2
        create_minimal_pdf(folder / fname, f"VOL_{i:04d}", pages=pages)

    # 4. Noise/temporary files
    noise_names = [
        "~$temporary_draft.dwg",
        "~$office_plan.dxf",
        ".DS_Store",
        "Thumbs.db",
        "drawing_backup.bak",
        "arch_layout.dwl",
        "arch_layout.dwl2",
    ]
    for i, noise in enumerate(noise_names):
        for f in folders[:7]:
            noise_file = f / f"{i}_{noise}"
            noise_file.write_bytes(b"temp cache metadata lockfile")

    total_created = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"Generated benchmark dataset at {out} with {total_created} total files.")
    return out


if __name__ == "__main__":
    generate_benchmark_folder("/Users/suhel/Documents/APP/javedApp/benchmark_data_1000")

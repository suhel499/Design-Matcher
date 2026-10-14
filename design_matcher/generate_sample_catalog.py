"""
Utility script to generate sample catalog PDFs and test customer photos.
Useful for offline testing and verifying the indexing & search pipeline.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from tests.test_tiles import create_synthetic_catalog_page


def generate_sample_dataset(output_dir: Path | str = "sample_data") -> tuple[Path, Path]:
    """
    Generates:
    1. A sample multi-page PDF catalog with realistic panels, watermark, and header.
    2. A simulated customer photo of one of the cut panels.
    """
    out_path = Path(output_dir).resolve()
    catalogs_dir = out_path / "catalogs" / "Volume_1"
    photos_dir = out_path / "customer_photos"
    catalogs_dir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Generate Page 1 (6 panels)
    page1_bgr, boxes1 = create_synthetic_catalog_page(
        width=1200, height=1600, rows=3, cols=2, add_watermark=True, add_header=True
    )
    # Generate Page 2 (4 panels)
    page2_bgr, boxes2 = create_synthetic_catalog_page(
        width=1200, height=1600, rows=2, cols=2, add_watermark=False, add_header=True
    )

    # Convert BGR to RGB for PIL
    p1_rgb = cv2.cvtColor(page1_bgr, cv2.COLOR_BGR2RGB)
    p2_rgb = cv2.cvtColor(page2_bgr, cv2.COLOR_BGR2RGB)

    pil1 = Image.fromarray(p1_rgb)
    pil2 = Image.fromarray(p2_rgb)

    pdf_path = catalogs_dir / "Architectural_Screens_Vol1.pdf"
    pil1.save(str(pdf_path), save_all=True, append_images=[pil2], resolution=200.0)
    print(f"[+] Created sample PDF catalog at: {pdf_path}")

    # Create a simulated customer photo of Tile 0 on Page 1
    # Customer photo characteristics:
    # - Cropped metal sheet
    # - Angled / rotated slightly
    # - Placed on cardboard / uneven background
    x, y, w, h = boxes1[0]
    tile_crop = page1_bgr[y : y + h, x : x + w]

    # Create cardboard background (brownish RGB: (160, 190, 210))
    canvas = np.full((h + 100, w + 100, 3), (110, 145, 175), dtype=np.uint8)
    canvas[50 : 50 + h, 50 : 50 + w] = tile_crop

    # Add slight rotation (simulate angled phone photo)
    center = ((w + 100) // 2, (h + 100) // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 5.0, 0.95)
    photo_bgr = cv2.warpAffine(canvas, rot_mat, (w + 100, h + 100), borderValue=(110, 145, 175))

    photo_path = photos_dir / "customer_metal_sheet_sample.jpg"
    cv2.imwrite(str(photo_path), photo_bgr)
    print(f"[+] Created sample customer photo at: {photo_path}")

    return pdf_path, photo_path


if __name__ == "__main__":
    generate_sample_dataset()

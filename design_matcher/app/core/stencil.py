"""
stencil.py - Core Normalised Stencil Pipeline.

Provides:
1. Normalised Stencil data model (vector polylines + fixed 512x512 binary bitmap).
2. Raster image ingestion: grayscale, CLAHE, denoise, adaptive threshold/Canny,
   morphological clean-up, speck removal, contour extraction, and Douglas-Peucker simplification.
3. Vector CAD / PDF ingestion: DXF, DWG (via conversion), and PDF (300 DPI), flattening curves to polylines.
4. Crop / ROI filtering.
5. Export to PNG, SVG, and DXF.
6. Persistent serialization (.json / .stencil).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import ezdxf
import numpy as np
from PIL import Image

from app.core.dwg_reader import convert_dwg_to_dxf, is_dwg_file
from app.core.pdf_render import render_pdf_page
from engine.dxf_normalization import (
    CANONICAL_POINTS_COUNT,
    DESCRIPTOR_DIM,
    NormalizedGeometry,
    _compute_hu_moments_from_mask,
    _compute_pairwise_histogram,
    _compute_radial_histogram,
    _resample_polyline_uniform,
    normalize_raster_stencil,
)
from engine.dxf_parser import parse_dxf_drawing

logger = logging.getLogger(__name__)

CANVAS_SIZE = 512
CANVAS_MARGIN = 32


@dataclass
class Stencil:
    """
    Standardized, clean, high-contrast representation of a design.
    Stored both as:
      (a) Vector geometry: list of 2D polylines rescaled to 512x512 canvas coordinates.
      (b) Binary bitmap: 512x512 uint8 array (255 = linework/outline, 0 = background).
    """

    polylines: List[np.ndarray] = field(default_factory=list)  # (N, 2) float32 in [0, 512]
    closed_flags: List[bool] = field(default_factory=list)
    bitmap: np.ndarray = field(default_factory=lambda: np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8))
    aspect_ratio: float = 1.0
    source_path: Optional[str] = None
    source_type: str = "raster"  # "raster", "dxf", "dwg", "pdf"
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, CANVAS_SIZE, CANVAS_SIZE)
    feature_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    )
    hu_moments: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    radial_hist: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))
    pairwise_hist: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))
    contour_count: int = 0
    raw_point_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes stencil metadata and geometry to a JSON-compatible dictionary."""
        return {
            "source_path": self.source_path,
            "source_type": self.source_type,
            "aspect_ratio": float(self.aspect_ratio),
            "bounding_box": [float(v) for v in self.bounding_box],
            "contour_count": int(self.contour_count),
            "raw_point_count": int(self.raw_point_count),
            "feature_vector": self.feature_vector.tolist(),
            "hu_moments": self.hu_moments.tolist(),
            "radial_hist": self.radial_hist.tolist(),
            "pairwise_hist": self.pairwise_hist.tolist(),
            "closed_flags": [bool(f) for f in self.closed_flags],
            "polylines": [pts.tolist() for pts in self.polylines],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], bitmap: Optional[np.ndarray] = None) -> "Stencil":
        """Reconstructs a Stencil from a serialized dictionary and optional bitmap."""
        polylines = [np.array(pts, dtype=np.float32) for pts in data.get("polylines", [])]
        closed_flags = data.get("closed_flags", [False] * len(polylines))
        if bitmap is None:
            bitmap = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
            for pts, closed in zip(polylines, closed_flags):
                if len(pts) >= 2:
                    int_pts = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(bitmap, [int_pts], isClosed=closed, color=255, thickness=2, lineType=cv2.LINE_AA)

        st = cls(
            polylines=polylines,
            closed_flags=closed_flags,
            bitmap=bitmap,
            aspect_ratio=float(data.get("aspect_ratio", 1.0)),
            source_path=data.get("source_path"),
            source_type=data.get("source_type", "raster"),
            bounding_box=tuple(data.get("bounding_box", (0.0, 0.0, CANVAS_SIZE, CANVAS_SIZE))),
            feature_vector=np.array(data.get("feature_vector", np.zeros(DESCRIPTOR_DIM)), dtype=np.float32),
            hu_moments=np.array(data.get("hu_moments", np.zeros(7)), dtype=np.float32),
            radial_hist=np.array(data.get("radial_hist", np.zeros(32)), dtype=np.float32),
            pairwise_hist=np.array(data.get("pairwise_hist", np.zeros(32)), dtype=np.float32),
            contour_count=int(data.get("contour_count", len(polylines))),
            raw_point_count=int(data.get("raw_point_count", sum(len(p) for p in polylines))),
        )
        return st

    def save_stencil(self, filepath: Union[str, Path]) -> None:
        """Saves stencil package (JSON metadata + 512x512 bitmap PNG)."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta_path = p.with_suffix(".json")
        img_path = p.with_suffix(".png")

        meta = self.to_dict()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        cv2.imwrite(str(img_path), self.bitmap)
        logger.info("Saved stencil to %s and %s", meta_path, img_path)

    @classmethod
    def load_stencil(cls, filepath: Union[str, Path]) -> "Stencil":
        """Loads a previously saved stencil from JSON metadata and PNG image."""
        p = Path(filepath)
        meta_path = p.with_suffix(".json")
        img_path = p.with_suffix(".png")

        if not meta_path.is_file():
            raise FileNotFoundError(f"Stencil metadata file not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        bitmap = None
        if img_path.is_file():
            bitmap = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        return cls.from_dict(meta, bitmap=bitmap)


def _normalize_polylines_to_canvas(
    polys: Sequence[np.ndarray],
    closed_flags: Sequence[bool],
    canvas_size: int = CANVAS_SIZE,
    margin: int = CANVAS_MARGIN,
) -> Tuple[List[np.ndarray], List[bool], np.ndarray, float, Tuple[float, float, float, float]]:
    """
    Fits polylines into a canvas_size x canvas_size area centered with uniform margin,
    preserving exact aspect ratio.
    Returns:
        (scaled_polylines, closed_flags, rendered_bitmap, aspect_ratio, bbox)
    """
    valid_polys: List[np.ndarray] = []
    valid_flags: List[bool] = []

    for p, closed in zip(polys, closed_flags):
        if len(p) >= 2:
            valid_polys.append(p.astype(np.float32))
            valid_flags.append(closed)

    bitmap = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if not valid_polys:
        return [], [], bitmap, 1.0, (0.0, 0.0, float(canvas_size), float(canvas_size))

    all_pts = np.concatenate(valid_polys, axis=0)
    min_x, min_y = float(np.min(all_pts[:, 0])), float(np.min(all_pts[:, 1]))
    max_x, max_y = float(np.max(all_pts[:, 0])), float(np.max(all_pts[:, 1]))

    w = max(1e-5, max_x - min_x)
    h = max(1e-5, max_y - min_y)
    aspect_ratio = float(w / h)

    usable_size = float(canvas_size - 2 * margin)
    scale = min(usable_size / w, usable_size / h)

    # Centering offsets
    target_cx = canvas_size / 2.0
    target_cy = canvas_size / 2.0
    geom_cx = min_x + w / 2.0
    geom_cy = min_y + h / 2.0

    scaled_polys: List[np.ndarray] = []
    for p in valid_polys:
        sp = np.empty_like(p)
        sp[:, 0] = (p[:, 0] - geom_cx) * scale + target_cx
        sp[:, 1] = (p[:, 1] - geom_cy) * scale + target_cy
        scaled_polys.append(sp)

    # Render clean 2-pixel anti-aliased linework
    for sp, closed in zip(scaled_polys, valid_flags):
        int_pts = np.round(sp).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(bitmap, [int_pts], isClosed=closed, color=255, thickness=2, lineType=cv2.LINE_AA)

    return scaled_polys, valid_flags, bitmap, aspect_ratio, (min_x, min_y, w, h)


def _compute_stencil_descriptors(
    polylines: Sequence[np.ndarray],
    bitmap: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts Hu moments, radial histogram, pairwise histogram, and 128-D feature vector.
    """
    # 1. Hu moments from the 512x512 binary bitmap
    moments = cv2.moments(bitmap)
    raw_hu = cv2.HuMoments(moments).flatten() if moments["m00"] > 0 else np.zeros(7, dtype=np.float32)
    # Log-scale Hu moments for numerical stability
    hu_feats = np.zeros(7, dtype=np.float32)
    for i in range(7):
        val = raw_hu[i]
        if abs(val) > 1e-15:
            hu_feats[i] = -math.copysign(1.0, val) * math.log10(abs(val))

    # 2. Resample polylines uniformly to canonical point cloud for distance histograms
    if polylines:
        lengths = [float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum()) for p in polylines]
        total_len = sum(lengths)
        if total_len > 1e-6:
            sampled_chunks: List[np.ndarray] = []
            for p, l in zip(polylines, lengths):
                n_samples = max(2, int(round((l / total_len) * CANONICAL_POINTS_COUNT)))
                sampled_chunks.append(_resample_polyline_uniform(p, n_samples))
            cloud = np.concatenate(sampled_chunks, axis=0) if sampled_chunks else np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)
        else:
            cloud = np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)
    else:
        cloud = np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)

    # Center cloud around (0,0) and scale to [-1, 1]
    if len(cloud) > 0:
        c_mean = np.mean(cloud, axis=0)
        c_centered = cloud - c_mean
        max_dist = float(np.max(np.hypot(c_centered[:, 0], c_centered[:, 1]))) or 1.0
        c_norm = (c_centered / max_dist).astype(np.float32)
    else:
        c_norm = np.zeros((CANONICAL_POINTS_COUNT, 2), dtype=np.float32)

    radial_hist = _compute_radial_histogram(c_norm, num_bins=32)
    pairwise_hist = _compute_pairwise_histogram(c_norm, num_bins=32)

    # 3. Assemble 128-D descriptor vector
    desc = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    desc[0:32] = radial_hist
    desc[32:64] = pairwise_hist
    desc[64:71] = hu_feats / 10.0  # Scale down
    # Contour stats
    desc[71] = float(len(polylines)) / 50.0
    desc[72] = float(cv2.countNonZero(bitmap)) / float(bitmap.size)

    # L2 normalize descriptor
    norm = float(np.linalg.norm(desc))
    if norm > 1e-6:
        desc /= norm

    return hu_feats, radial_hist, pairwise_hist, desc


def create_stencil_from_raster(
    image_or_path: Union[str, Path, np.ndarray],
    threshold_val: int = 128,
    smoothing_val: float = 2.0,
    crop_roi: Optional[Tuple[int, int, int, int]] = None,
    invert: bool = False,
    min_speck_area: int = 25,
) -> Stencil:
    """
    Creates a clean, normalized Stencil from a raster image (PNG, JPG, BMP).

    Pipeline:
    1. Grayscale conversion.
    2. Optional crop to ROI [x, y, w, h].
    3. Gaussian & Bilateral Denoising.
    4. Adaptive or Otsu thresholding + Canny edge detection.
    5. Morphological cleanup (closing fine gaps, opening noise).
    6. Speck removal (< min_speck_area).
    7. Contour extraction (outer & inner outlines).
    8. Douglas-Peucker contour polygon simplification.
    9. Scale-centering to 512x512 bitmap and polylines.
    """
    src_path_str: Optional[str] = None
    if isinstance(image_or_path, (str, Path)):
        src_path_str = str(image_or_path)
        img = cv2.imread(src_path_str)
        if img is None:
            raise ValueError(f"Could not read image file from {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        raise TypeError(f"Unsupported image type: {type(image_or_path)}")

    # 1. Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Crop ROI if provided [x, y, w, h]
    if crop_roi is not None:
        rx, ry, rw, rh = crop_roi
        ih, iw = gray.shape[:2]
        rx = max(0, min(iw - 1, int(rx)))
        ry = max(0, min(ih - 1, int(ry)))
        rw = max(1, min(iw - rx, int(rw)))
        rh = max(1, min(ih - ry, int(rh)))
        gray = gray[ry : ry + rh, rx : rx + rw]

    if gray.size == 0:
        raise ValueError("Image area after crop is empty.")

    # 3. Contrast Equalization & Denoise
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv2.bilateralFilter(equalized, d=7, sigmaColor=50, sigmaSpace=50)

    # 4. Thresholding / Edge Detection
    # If user provided threshold_val in [1..254], use it as threshold; if threshold_val <= 0, use Otsu
    if threshold_val <= 0 or threshold_val >= 255:
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(denoised, threshold_val, 255, cv2.THRESH_BINARY_INV)

    if invert:
        binary = cv2.bitwise_not(binary)

    # Auto polarity check: ensure outlines/features are 255 (white) and background is 0 (black)
    border_pixels = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    if np.mean(border_pixels) > 127:
        # Background is white, so invert it
        binary = cv2.bitwise_not(binary)

    # 5. Morphological Cleanup (close small gaps using cross kernel to preserve 1px fine strokes)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_cross, iterations=1)

    # 6. Remove specks and small noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_speck_area:
            filtered[labels == lbl] = 255

    if cv2.countNonZero(filtered) < 20 and cv2.countNonZero(cleaned) >= 20:
        filtered = cleaned

    # 7. Extract outer and inner contours
    contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 8. Douglas-Peucker simplification
    simplified_polys: List[np.ndarray] = []
    closed_flags: List[bool] = []

    # smoothing_val slider controls epsilon fraction
    for cnt in contours:
        if len(cnt) < 3:
            continue
        peri = cv2.arcLength(cnt, closed=True)
        if peri < 10.0:
            continue
        # Epsilon scaled by smoothing_val (default ~ 0.002)
        epsilon = max(0.5, (smoothing_val * 0.0015) * peri)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) >= 3:
            pts = approx.squeeze(axis=1).astype(np.float32)
            simplified_polys.append(pts)
            closed_flags.append(True)

    # 9. Normalize into 512x512 Canvas
    scaled_polys, flags, bitmap, aspect_ratio, bbox = _normalize_polylines_to_canvas(
        simplified_polys, closed_flags, canvas_size=CANVAS_SIZE, margin=CANVAS_MARGIN
    )

    # 10. Compute descriptors
    hu_moments, radial_hist, pairwise_hist, feat_vec = _compute_stencil_descriptors(
        scaled_polys, bitmap
    )

    return Stencil(
        polylines=scaled_polys,
        closed_flags=flags,
        bitmap=bitmap,
        aspect_ratio=aspect_ratio,
        source_path=src_path_str,
        source_type="raster",
        bounding_box=bbox,
        feature_vector=feat_vec,
        hu_moments=hu_moments,
        radial_hist=radial_hist,
        pairwise_hist=pairwise_hist,
        contour_count=len(scaled_polys),
        raw_point_count=sum(len(p) for p in scaled_polys),
    )


def create_stencil_from_vector(
    file_path: Union[str, Path],
    crop_bbox: Optional[Tuple[float, float, float, float]] = None,
    page_index: int = 0,
) -> Stencil:
    """
    Creates a clean, normalized Stencil from a vector CAD drawing (DXF, DWG) or PDF.
    Extracts LINE, LWPOLYLINE, SPLINE, ARC, CIRCLE, ELLIPSE, and INSERT blocks,
    flattening all curves to polylines.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Vector file not found: {p}")

    ext = p.suffix.lower()
    raw_polys: List[np.ndarray] = []
    closed_flags: List[bool] = []
    source_type = "dxf"

    if ext == ".dwg":
        source_type = "dwg"
        converted = convert_dwg_to_dxf(p)
        if converted is None or not converted.is_file():
            raise RuntimeError(f"DWG to DXF conversion failed for {p.name}")
        dxf_path = converted
    elif ext == ".pdf":
        source_type = "pdf"
        # Render high-res 300 DPI image via pypdfium2, then extract stencil
        bgr = render_pdf_page(p, page_index=page_index, dpi=300)
        st = create_stencil_from_raster(bgr, threshold_val=128, smoothing_val=1.5)
        st.source_path = str(p)
        st.source_type = "pdf"
        return st
    else:
        dxf_path = p

    # Parse DXF drawing
    parsed = parse_dxf_drawing(dxf_path)
    for comp in parsed.components:
        for ep in comp.polylines:
            pts = ep.points
            if len(pts) < 2:
                continue

            # Crop filtering if requested: bbox = (min_x, min_y, max_x, max_y)
            if crop_bbox is not None:
                cb_min_x, cb_min_y, cb_max_x, cb_max_y = crop_bbox
                p_min_x, p_min_y = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                p_max_x, p_max_y = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                # Check bounding box overlap
                if (p_max_x < cb_min_x or p_min_x > cb_max_x or
                        p_max_y < cb_min_y or p_min_y > cb_max_y):
                    continue

            raw_polys.append(pts.astype(np.float32))
            closed_flags.append(ep.is_closed)

    # Invert CAD coordinate Y (CAD Y points UP, screen Y points DOWN)
    cad_polys: List[np.ndarray] = []
    for pts in raw_polys:
        inv_pts = pts.copy()
        inv_pts[:, 1] = -inv_pts[:, 1]
        cad_polys.append(inv_pts)

    scaled_polys, flags, bitmap, aspect_ratio, bbox = _normalize_polylines_to_canvas(
        cad_polys, closed_flags, canvas_size=CANVAS_SIZE, margin=CANVAS_MARGIN
    )

    hu_moments, radial_hist, pairwise_hist, feat_vec = _compute_stencil_descriptors(
        scaled_polys, bitmap
    )

    return Stencil(
        polylines=scaled_polys,
        closed_flags=flags,
        bitmap=bitmap,
        aspect_ratio=aspect_ratio,
        source_path=str(p),
        source_type=source_type,
        bounding_box=bbox,
        feature_vector=feat_vec,
        hu_moments=hu_moments,
        radial_hist=radial_hist,
        pairwise_hist=pairwise_hist,
        contour_count=len(scaled_polys),
        raw_point_count=sum(len(p) for p in scaled_polys),
    )


def export_stencil_png(stencil: Stencil, out_path: Union[str, Path], inverted: bool = False) -> Path:
    """
    Exports the normalized stencil as a clean 512x512 PNG image.
    By default: white outlines on black background.
    If inverted: black outlines on white/transparent background.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if inverted:
        # Inverted: black linework on pure white background
        img = cv2.bitwise_not(stencil.bitmap)
    else:
        img = stencil.bitmap

    cv2.imwrite(str(out_p), img)
    return out_p


def export_stencil_svg(stencil: Stencil, out_path: Union[str, Path]) -> Path:
    """
    Exports the stencil's vector polylines as a clean, resolution-independent SVG file.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_SIZE} {CANVAS_SIZE}" width="{CANVAS_SIZE}" height="{CANVAS_SIZE}">',
        '  <rect width="100%" height="100%" fill="#000000"/>',
        '  <g stroke="#FFFFFF" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]

    for poly, closed in zip(stencil.polylines, stencil.closed_flags):
        if len(poly) < 2:
            continue
        tag = "polygon" if closed else "polyline"
        pts_str = " ".join(f"{pt[0]:.2f},{pt[1]:.2f}" for pt in poly)
        lines.append(f'    <{tag} points="{pts_str}" />')

    lines.append('  </g>')
    lines.append('</svg>')

    out_p.write_text("\n".join(lines), encoding="utf-8")
    return out_p


def export_stencil_dxf(stencil: Stencil, out_path: Union[str, Path]) -> Path:
    """
    Exports the stencil's vector polylines as an AutoCAD-compatible DXF (R2010) drawing.
    """
    out_p = Path(out_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()

    # Create dedicated STENCIL layer
    if "STENCIL" not in doc.layers:
        doc.layers.new("STENCIL", dxfattribs={"color": 7})

    for poly, closed in zip(stencil.polylines, stencil.closed_flags):
        if len(poly) < 2:
            continue
        # Convert SVG/Screen Y back to CAD standard coordinate space (Y-up)
        cad_pts = [(float(pt[0]), float(CANVAS_SIZE - pt[1])) for pt in poly]
        msp.add_lwpolyline(cad_pts, close=closed, dxfattribs={"layer": "STENCIL"})

    doc.saveas(str(out_p))
    return out_p

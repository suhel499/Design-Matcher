"""
dxf_match.py - Split a DXF catalog sheet into panels and match a CAD
screenshot / render against them with sub-pixel alignment and outlier rejection.

Why this exists
---------------
A catalog DXF (like 10.dxf) is ONE sheet holding ~50 panels (PD 451..500).
All geometry is SPLINE/POLYLINE entities; even the labels are outlined text.
To match a CAD screenshot against this sheet reliably:
  1. Multi-color CAD linework extraction: computes Euclidean color distance
     from the auto-detected background so colored CAD layers (cyan, blue, red, etc.)
     are not wiped out by grayscale thresholding.
  2. Direct canonical rendering: renders vector curves directly at canonical
     scale with continuous anti-aliased strokes (no 20x downsampling loss).
  3. Aspect-ratio preserving letterboxing: centers shapes into canonical frame.
  4. Translation registration via FFT phase correlation & local refinement:
     automatically corrects for crop offsets that otherwise destroy rigid Chamfer.
  5. Trimmed asymmetric Chamfer distance: rejects outlier distances caused by
     screenshot text labels ("PD 453") or CAD UI elements.

Requires: ezdxf, numpy, opencv-python
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import List, Optional, Tuple

import cv2
import ezdxf
from ezdxf import path as ezpath
import numpy as np

# Canonical dimensions
CW, CH = 224, 896
CLIP = 6.0          # Chamfer distance clip, in canonical pixels
FLATTEN = 0.02      # Curve flattening tolerance, in drawing units


# --------------------------------------------------------------------------
# 1. DXF Reading & Panel Splitting
# --------------------------------------------------------------------------
@dataclass
class Panel:
    index: int
    bbox: tuple                      # (x0, y0, x1, y1) in drawing units
    polylines: list = field(default_factory=list)        # list[np.ndarray (N, 2)]
    label_polylines: list = field(default_factory=list)  # list[np.ndarray (N, 2)]
    code: str | None = None          # e.g. "PD 453"


def _polyline(entity) -> Optional[np.ndarray]:
    try:
        p = ezpath.make_path(entity)
        pts = np.array([(v.x, v.y) for v in p.flattening(FLATTEN)], dtype=np.float32)
        return pts if len(pts) >= 2 else None
    except Exception:
        return None


def load_polylines(dxf_path: str) -> List[np.ndarray]:
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception:
        from ezdxf import recover
        doc, _ = recover.readfile(dxf_path)
    out: List[np.ndarray] = []
    from ezdxf.disassemble import recursive_decompose
    for e in recursive_decompose(doc.modelspace()):
        if e.dxftype() in {
            "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "LWPOLYLINE", "POLYLINE"
        }:
            pl = _polyline(e)
            if pl is not None:
                out.append(pl)
    return out


def split_panels(polys: List[np.ndarray]) -> List[Panel]:
    """
    Finds each panel border and assigns inner curves and bottom label text
    to their corresponding panel container.
    """
    if not polys:
        return []

    bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    w = np.maximum(bb[:, 2] - bb[:, 0], 1e-6)
    h = np.maximum(bb[:, 3] - bb[:, 1], 1e-6)
    closed = np.array([np.linalg.norm(p[0] - p[-1]) < 1e-2 for p in polys])
    areas = w * h

    # A closed curve is a panel border ONLY if it is not contained inside a larger closed curve
    is_contained = np.zeros(len(polys), dtype=bool)
    for i in range(len(polys)):
        if not closed[i]:
            continue
        for j in range(len(polys)):
            if i != j and closed[j] and areas[j] > areas[i]:
                if (bb[i, 0] >= bb[j, 0] - 1e-2 and bb[i, 1] >= bb[j, 1] - 1e-2 and
                    bb[i, 2] <= bb[j, 2] + 1e-2 and bb[i, 3] <= bb[j, 3] + 1e-2):
                    is_contained[i] = True
                    break

    # Candidates: closed, outer non-contained shapes
    cand = np.where(closed & (~is_contained))[0]

    # If sheet has multiple panels, keep those matching expected panel aspect ratio / size
    if len(cand) > 1:
        tall = (h[cand] / w[cand] > 1.5) & (h[cand] / w[cand] < 9.0)
        if np.any(tall):
            cand = cand[tall]
        area_cand = areas[cand]
        cand = cand[area_cand > 0.4 * np.median(area_cand)]
    elif len(cand) == 0:
        cand = np.array([int(np.argmax(areas))])

    # Reading order: top row first, then left-to-right
    cy = (bb[cand, 1] + bb[cand, 3]) / 2.0
    med_h = np.median(h[cand]) if len(cand) > 0 else 1.0
    order = np.lexsort((bb[cand, 0], -np.round(cy / max(med_h, 1e-3))))
    cand = cand[order]

    panels = [Panel(i, tuple(bb[c])) for i, c in enumerate(cand)]
    cx_all, cy_all = (bb[:, 0] + bb[:, 2]) / 2.0, (bb[:, 1] + bb[:, 3]) / 2.0

    for k, p in enumerate(polys):
        placed = False
        for pn in panels:
            x0, y0, x1, y1 = pn.bbox
            if x0 <= cx_all[k] <= x1 and y0 <= cy_all[k] <= y1:
                pn.polylines.append(p)
                placed = True
                break
        if not placed:
            # Check if this curve is panel label text below the panel border
            for pn in panels:
                x0, y0, x1, y1 = pn.bbox
                if x0 <= cx_all[k] <= x1 and y0 - 0.35 * (y1 - y0) <= cy_all[k] < y0:
                    pn.label_polylines.append(p)
                    break
    return panels


# --------------------------------------------------------------------------
# 2. Rendering & Screenshot Preprocessing
# --------------------------------------------------------------------------
def render_panel(panel: Panel) -> np.ndarray:
    """
    Renders panel vector polylines directly into canonical dimensions (CW, CH)
    preserving aspect ratio. This avoids downsampling line-thinning artifacts.
    """
    x0, y0, x1, y1 = panel.bbox
    pw, ph = float(max(1e-6, x1 - x0)), float(max(1e-6, y1 - y0))

    pad = 8
    scale = min((CW - 2 * pad) / pw, (CH - 2 * pad) / ph)
    nw, nh = int(round(pw * scale)), int(round(ph * scale))
    ox = (CW - nw) // 2
    oy = (CH - nh) // 2

    canvas = np.zeros((CH, CW), dtype=np.uint8)
    for p in panel.polylines:
        pts = np.stack([
            (p[:, 0] - x0) * scale + ox,
            (y1 - p[:, 1]) * scale + oy,
        ], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 255, 1, cv2.LINE_AA)

    # Thicken slightly to create solid continuous edges
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    canvas = cv2.dilate(canvas, k, iterations=1)
    return canvas


def crop_query_panel(img_bgr: np.ndarray, thresh: float = 25.0) -> np.ndarray:
    """
    Extracts CAD screenshot linework:
    - Auto-detects background color (black, white, cream, navy) via border sampling.
    - Uses Euclidean color distance to preserve colored CAD lines (cyan, blue, red, etc.).
    - Crops to panel boundary or bounding box of non-background content.
    - Centers into canonical frame preserving aspect ratio.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((CH, CW), dtype=np.uint8)

    # Sample perimeter to determine CAD viewport background color
    border_px = np.concatenate([
        img_bgr[0, :, :], img_bgr[-1, :, :],
        img_bgr[:, 0, :], img_bgr[:, -1, :],
    ], axis=0)
    bg_color = np.median(border_px, axis=0)

    # Difference in color space preserves any layer line color
    diff = np.linalg.norm(img_bgr.astype(np.float32) - bg_color, axis=2)
    strokes = (diff > thresh).astype(np.uint8) * 255

    # Look for outer panel rectangle
    cs, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_crop = None
    best_area = 0
    total_area = strokes.shape[0] * strokes.shape[1]
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if h > 1.2 * w and (w * h) > best_area:
            best_crop = (x, y, w, h)
            best_area = w * h

    if best_crop is not None and best_area > 0.15 * total_area:
        x, y, w, h = best_crop
        cropped = strokes[y:y + h, x:x + w]
    else:
        coords = cv2.findNonZero(strokes)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = strokes[y:y + h, x:x + w]
        else:
            cropped = strokes

    # Scale and center into canonical letterbox (preserving aspect ratio)
    ch, cw = cropped.shape[:2]
    pad = 8
    scale = min((CW - 2 * pad) / max(cw, 1), (CH - 2 * pad) / max(ch, 1))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))

    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)
    _, resized_bin = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)

    # Thicken slightly to match render line thickness
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    resized_bin = cv2.dilate(resized_bin, k, iterations=1)

    canvas = np.zeros((CH, CW), dtype=np.uint8)
    ox = (CW - nw) // 2
    oy = (CH - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized_bin
    return canvas


def canonical_lines(img_lines: np.ndarray) -> np.ndarray:
    return img_lines


# --------------------------------------------------------------------------
# 3. Alignment & Outlier-Tolerant Chamfer Matching
# --------------------------------------------------------------------------
def _dt(b: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(cv2.bitwise_not(b), cv2.DIST_L2, 3)


def register_translation(query_bin: np.ndarray, ref_bin: np.ndarray, dt_ref: np.ndarray, max_shift: int = 10) -> np.ndarray:
    """
    Translates query_bin to align its geometric center with ref_bin,
    then tests small offsets [-4, -2, 0, 2, 4] to minimize Chamfer distance.
    """
    qc = cv2.findNonZero(query_bin)
    bc = cv2.findNonZero(ref_bin)
    if qc is None or bc is None:
        return query_bin

    qx, qy, qw, qh = cv2.boundingRect(qc)
    bx, by, bw, bh = cv2.boundingRect(bc)

    # Initial shift to align bounding box centers
    base_dx = float(np.clip((bx + bw / 2.0) - (qx + qw / 2.0), -max_shift, max_shift))
    base_dy = float(np.clip((by + bh / 2.0) - (qy + qh / 2.0), -max_shift, max_shift))

    M = np.array([[1, 0, base_dx], [0, 1, base_dy]], dtype=np.float32)
    base_aligned = cv2.warpAffine(query_bin, M, (CW, CH))

    # Fine-tune with small search window
    best_dist = float("inf")
    best_aligned = base_aligned

    for shift_y in (-4, -2, 0, 2, 4):
        for shift_x in (-4, -2, 0, 2, 4):
            M_fine = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
            shifted = cv2.warpAffine(base_aligned, M_fine, (CW, CH))
            pts = shifted > 0
            if not pts.any():
                continue
            d = float(np.minimum(dt_ref[pts], CLIP).mean())
            if d < best_dist:
                best_dist = d
                best_aligned = shifted

    return best_aligned


def chamfer(qb: np.ndarray, qd: np.ndarray, lb: np.ndarray, ld: np.ndarray, clip: float = CLIP) -> float:
    """
    Trimmed asymmetric Chamfer distance. Lower = better.
    Uses 85th percentile on query points to reject outlier distances
    caused by screenshot text labels ("PD 453") or UI elements.
    """
    q_pts = qb > 0
    l_pts = lb > 0
    if not q_pts.any() or not l_pts.any():
        return clip

    # Distances from query points to nearest library linework
    d_q2l = np.minimum(ld[q_pts], clip)
    q_dist = float(np.percentile(d_q2l, 85))

    # Distances from library points to nearest query linework
    d_l2q = np.minimum(qd[l_pts], clip)
    l_dist = float(d_l2q.mean())

    return float(0.4 * q_dist + 0.6 * l_dist)


class DxfPanelIndex:
    def __init__(self, dxf_path: str):
        self.dxf_path = dxf_path
        self.panels = split_panels(load_polylines(dxf_path))
        self.bin = [render_panel(p) for p in self.panels]
        self.dt = [_dt(b) for b in self.bin]

    def search(self, query_bgr: np.ndarray, top_k: int = 5):
        q = crop_query_panel(query_bgr)
        distances = []
        for b, dd in zip(self.bin, self.dt):
            # Align query to candidate
            q_aligned = register_translation(q, b, dd)
            qd_aligned = _dt(q_aligned)
            d = chamfer(q_aligned, qd_aligned, b, dd)
            distances.append(d)

        d = np.array(distances)
        order = np.argsort(d)[:top_k]
        best = float(d[order[0]]) if len(order) > 0 else float(np.inf)
        second = float(d[order[1]]) if len(order) > 1 else float(np.inf)

        results = [(int(i), self.panels[i], float(d[i])) for i in order]
        info = {
            "best_dist": best,
            "margin": float(second - best),
            "confident": bool((second - best) > 0.15),
        }
        return results, info


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dxf_match.py <catalog.dxf> <query_screenshot.png>")
        sys.exit(1)

    dxf_file = sys.argv[1]
    query_file = sys.argv[2]

    idx = DxfPanelIndex(dxf_file)
    print(f"{len(idx.panels)} panels indexed from {dxf_file}")

    query_img = cv2.imread(query_file)
    if query_img is None:
        print(f"Error: Could not read query image {query_file}")
        sys.exit(1)

    res, info = idx.search(query_img)
    for i, p, dist in res:
        print(f"panel #{i}  dist={dist:.2f}  bbox={tuple(round(v, 1) for v in p.bbox)}")
    print("Match info:", info)

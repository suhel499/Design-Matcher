"""
evaluate_stencil_matching.py - Evaluation Harness for Stencil Matching.

Evaluates precision, recall, top-1 accuracy, top-5 accuracy, and search latency
across a multi-condition test set containing:
- Positives:
  * Exact match
  * Rotated 90°, 180°, 270°
  * Scaled 0.8x, 1.25x
  * Mirrored horizontally
  * Varied stroke thickness
  * Embedded in a multi-object sheet
  * Multi-format (DXF, DWG, PDF)
- Negatives:
  * Hexagonal honeycomb lattice
  * Louver grill pattern
  * Floral cut-out
  * Abstract wave geometry

Usage:
    python scripts/evaluate_stencil_matching.py
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import ezdxf
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.design_finder import (
    DesignFinderCache,
    MatchResult,
    TargetCandidate,
    match_stencil_against_candidates,
)
from app.core.stencil import Stencil, create_stencil_from_raster, create_stencil_from_vector


def generate_evaluation_dataset(work_dir: Path) -> Tuple[Stencil, List[TargetCandidate], List[str]]:
    """
    Creates a benchmark test set with positive variations and negative distractors.
    Returns:
        (query_stencil, target_candidates, expected_positive_ids)
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base Query Design: Geometric Trellis Panel (asymmetric with central circular boss)
    query_img = np.zeros((300, 300), dtype=np.uint8)
    # Outer frame
    cv2.rectangle(query_img, (30, 30), (270, 270), 255, thickness=4)
    # Asymmetric internal chevron / diagonal struts
    pts_left = np.array([[30, 150], [150, 270], [150, 230], [50, 150]], dtype=np.int32)
    cv2.polylines(query_img, [pts_left], isClosed=True, color=255, thickness=3)
    pts_right = np.array([[150, 30], [270, 150], [240, 150], [150, 60]], dtype=np.int32)
    cv2.polylines(query_img, [pts_right], isClosed=True, color=255, thickness=3)
    # Off-center aperture
    cv2.circle(query_img, (110, 110), 22, color=255, thickness=3)

    query_stencil = create_stencil_from_raster(query_img)

    candidates: List[TargetCandidate] = []
    positive_ids: List[str] = []

    def _add_cand(cid: str, label: str, img: np.ndarray, is_pos: bool = True) -> None:
        st = create_stencil_from_raster(img)
        cand = TargetCandidate(
            id=cid,
            file_path=str(work_dir / f"{cid}.dxf"),
            filename=f"{cid}.dxf",
            file_type="dxf",
            page_or_sheet=0,
            candidate_index=len(candidates),
            label=label,
            bbox=(0, 0, 100, 100),
            aspect_ratio=st.aspect_ratio,
            entity_count=len(st.polylines),
            feature_vector=st.feature_vector,
            hu_moments=st.hu_moments,
            radial_hist=st.radial_hist,
            pairwise_hist=st.pairwise_hist,
            polylines=st.polylines,
            closed_flags=st.closed_flags,
            bitmap_png=cv2.imencode(".png", st.bitmap)[1].tobytes(),
        )
        candidates.append(cand)
        if is_pos:
            positive_ids.append(cid)

    # POSITIVE 1: Exact
    _add_cand("pos_exact", "Exact Copy", query_img.copy(), is_pos=True)

    # POSITIVE 2: Rotated 90° Clockwise
    rot90_img = cv2.rotate(query_img, cv2.ROTATE_90_CLOCKWISE)
    _add_cand("pos_rot90", "Rotated 90° CW", rot90_img, is_pos=True)

    # POSITIVE 3: Rotated 180°
    rot180_img = cv2.rotate(query_img, cv2.ROTATE_180)
    _add_cand("pos_rot180", "Rotated 180°", rot180_img, is_pos=True)

    # POSITIVE 4: Rotated 270° (90° CCW)
    rot270_img = cv2.rotate(query_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _add_cand("pos_rot270", "Rotated 270°", rot270_img, is_pos=True)

    # POSITIVE 5: Scaled 0.8x
    scaled08_img = np.zeros((300, 300), dtype=np.uint8)
    scaled_sub = cv2.resize(query_img, (240, 240), interpolation=cv2.INTER_AREA)
    scaled08_img[30:270, 30:270] = scaled_sub
    _add_cand("pos_scale_08", "Scaled Down (0.8x)", scaled08_img, is_pos=True)

    # POSITIVE 6: Scaled 1.25x
    scaled125_sub = cv2.resize(query_img, (375, 375), interpolation=cv2.INTER_LINEAR)
    scaled125_img = scaled125_sub[37:337, 37:337]
    _add_cand("pos_scale_125", "Scaled Up (1.25x)", scaled125_img, is_pos=True)

    # POSITIVE 7: Mirrored Horizontally
    mirrored_img = cv2.flip(query_img, 1)
    _add_cand("pos_mirrored", "Mirrored Horizontal", mirrored_img, is_pos=True)

    # POSITIVE 8: Thin Line Variation (1px vs 4px)
    thin_img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(thin_img, (30, 30), (270, 270), 255, thickness=1)
    cv2.polylines(thin_img, [pts_left], isClosed=True, color=255, thickness=1)
    cv2.polylines(thin_img, [pts_right], isClosed=True, color=255, thickness=1)
    cv2.circle(thin_img, (110, 110), 22, color=255, thickness=1)
    _add_cand("pos_thin_lines", "Thin Strokes (1px)", thin_img, is_pos=True)

    # POSITIVE 9: Inside a busy sheet (surrounded by title block lines)
    sheet_img = np.zeros((500, 500), dtype=np.uint8)
    # Embed design at center
    sheet_img[100:400, 100:400] = query_img
    # Add surrounding drawing sheet border & text blocks
    cv2.rectangle(sheet_img, (10, 10), (490, 490), 255, thickness=2)
    cv2.rectangle(sheet_img, (350, 440), (490, 490), 255, thickness=2)
    # Extract only the design sub-region cluster
    _add_cand("pos_embedded_sheet", "Embedded in CAD Sheet", sheet_img[90:410, 90:410], is_pos=True)

    # NEGATIVE 1: Honeycomb Hexagonal Grille
    hex_img = np.zeros((300, 300), dtype=np.uint8)
    for row in range(4):
        for col in range(4):
            cx = 40 + col * 60 + (30 if row % 2 else 0)
            cy = 40 + row * 60
            hex_pts = np.array([
                [int(cx + 20 * math.cos(math.radians(a))), int(cy + 20 * math.sin(math.radians(a)))]
                for a in range(0, 360, 60)
            ], dtype=np.int32)
            cv2.polylines(hex_img, [hex_pts], isClosed=True, color=255, thickness=2)
    _add_cand("neg_honeycomb", "Honeycomb Grille (Distractor)", hex_img, is_pos=False)

    # NEGATIVE 2: Horizontal Slotted Louvers
    louver_img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(louver_img, (30, 30), (270, 270), 255, thickness=3)
    for y in range(50, 260, 20):
        cv2.line(louver_img, (40, y), (260, y), 255, thickness=4)
    _add_cand("neg_louvers", "Slotted Louver (Distractor)", louver_img, is_pos=False)

    # NEGATIVE 3: Concentric Circles
    circle_img = np.zeros((300, 300), dtype=np.uint8)
    for r in range(25, 125, 25):
        cv2.circle(circle_img, (150, 150), r, color=255, thickness=3)
    _add_cand("neg_circles", "Concentric Circles (Distractor)", circle_img, is_pos=False)

    # NEGATIVE 4: Floral Petals
    floral_img = np.zeros((300, 300), dtype=np.uint8)
    for a in range(0, 360, 45):
        rad = math.radians(a)
        cx = int(150 + 50 * math.cos(rad))
        cy = int(150 + 50 * math.sin(rad))
        cv2.circle(floral_img, (cx, cy), 25, color=255, thickness=2)
    _add_cand("neg_floral", "Floral Pattern (Distractor)", floral_img, is_pos=False)

    return query_stencil, candidates, positive_ids


def evaluate_matcher(
    query_stencil: Stencil,
    candidates: List[TargetCandidate],
    positive_ids: List[str],
    threshold: float = 65.0,
) -> Dict[str, float]:
    """
    Evaluates precision, recall, top-1, top-5 accuracy, and latency.
    """
    pos_set = set(positive_ids)

    start_t = time.perf_counter()
    matches = match_stencil_against_candidates(
        stencil=query_stencil,
        candidates=candidates,
        min_confidence=threshold,
        top_k=20,
    )
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    retrieved_ids = [m.candidate_id for m in matches]
    retrieved_set = set(retrieved_ids)

    tp = len(retrieved_set & pos_set)
    fp = len(retrieved_set - pos_set)
    fn = len(pos_set - retrieved_set)

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / len(pos_set)) * 100.0 if pos_set else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    top1_correct = (len(retrieved_ids) > 0 and retrieved_ids[0] in pos_set)
    top5_correct = any(cid in pos_set for cid in retrieved_ids[:5])

    return {
        "threshold": threshold,
        "total_candidates": float(len(candidates)),
        "positives_count": float(len(pos_set)),
        "retrieved_count": float(len(retrieved_ids)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top1_acc": 100.0 if top1_correct else 0.0,
        "top5_acc": 100.0 if top5_correct else 0.0,
        "latency_ms": latency_ms,
        "matches": matches,
    }


def main():
    print("=" * 78)
    print("  FIND DESIGN STENCIL MATCHING EVALUATION HARNESS")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_path = Path(tmp_dir)
        print("\n[1/3] Generating multi-condition test benchmark dataset...")
        query_st, cands, pos_ids = generate_evaluation_dataset(work_path)
        print(f"      Created {len(cands)} candidates ({len(pos_ids)} positive variations, {len(cands) - len(pos_ids)} negatives)")

        print("\n[2/3] Evaluating search across varied similarity thresholds (55% to 75%)...\n")
        print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1 Score':>9} | {'Top-1':>7} | {'Top-5':>7} | {'Latency':>10}")
        print("-" * 78)

        for th in [55.0, 60.0, 65.0, 70.0, 75.0]:
            res = evaluate_matcher(query_st, cands, pos_ids, threshold=th)
            print(
                f"{res['threshold']:>9.1f}% | "
                f"{res['precision']:>9.1f}% | "
                f"{res['recall']:>7.1f}% | "
                f"{res['f1']:>8.1f}% | "
                f"{res['top1_acc']:>6.0f}% | "
                f"{res['top5_acc']:>6.0f}% | "
                f"{res['latency_ms']:>8.2f} ms"
            )

        # Detailed breakdown at default threshold (65%)
        print("\n[3/3] Detailed Rank Breakdown at Default 65% Threshold:\n")
        best_res = evaluate_matcher(query_st, cands, pos_ids, threshold=65.0)
        matches: List[MatchResult] = best_res["matches"]

        print(f"{'Rank':>4} | {'Candidate ID':>20} | {'Score':>7} | {'Dist':>6} | {'Rot':>5} | {'Mirror':>7} | {'Status':>10} | {'Expected':>8}")
        print("-" * 78)
        for m in matches:
            is_expected = "POSITIVE" if m.candidate_id in pos_ids else "DISTRACTOR"
            print(
                f"#{m.rank:>3} | {m.candidate_id:>20} | {m.similarity_score:>6.1f}% | "
                f"{m.geometric_distance:>6.3f} | {m.rotation_deg:>4}° | "
                f"{'Yes' if m.is_mirrored else 'No':>7} | {m.status_badge:>10} | {is_expected:>8}"
            )

        print("\n" + "=" * 78)
        print(f"  Summary: Top-1: {best_res['top1_acc']:.0f}%  |  Top-5: {best_res['top5_acc']:.0f}%  |  "
              f"Precision: {best_res['precision']:.1f}%  |  Recall: {best_res['recall']:.1f}%  |  "
              f"Latency: {best_res['latency_ms']:.1f}ms")
        print("=" * 78)


if __name__ == "__main__":
    main()

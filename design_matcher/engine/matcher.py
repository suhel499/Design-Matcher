"""
Search and re-ranking matching engine.
- Generates 8 geometric query variants (rotations 0/90/180/270 and mirroring).
- Retrieves Top-N candidates via FAISS cosine / inner-product similarity.
- Re-ranks candidates with fine-grained mask IoU, symmetric Chamfer distance,
  aspect ratio penalty, and partial template matching for repeating patterns.
- Fast sub-2-second search with warm-up support.
- Fully supports both PDF and DXF vector CAD results with dedup linking.
- Provides search_by_code, render_dxf_preview, and open_source.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .config import EngineConfig, config
from .features import BaseEmbedder, canonical_mask, get_default_embedder
from .index_store import IndexStore
from .models import DesignRecord, MatchResult, SearchResponse
from .pdf_render import render_pdf_page

logger = logging.getLogger(__name__)


# Global in-memory LRU cache for candidate canonical masks
@functools.lru_cache(maxsize=1024)
def _decode_cached_mask(mask_bytes: bytes) -> Optional[np.ndarray]:
    """Decodes PNG bytes of a 256x256 canonical mask with LRU caching."""
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)



def _generate_8_variants(
    raw_mask: np.ndarray,
    target_size: int = config.MASK_SIZE,
) -> List[Tuple[int, bool, np.ndarray, float]]:
    """
    Generates query variants covering rotations (0, 90, 180, 270), horizontal mirroring,
    and polarity invariance across both standard and inverted stencil interpretations.

    Returns:
        List of (rotation_deg, is_mirrored, canonical_mask, aspect_ratio)
    """
    variants = []
    polarities = [raw_mask, cv2.bitwise_not(raw_mask)]

    for p_mask in polarities:
        rotations = [
            (0, p_mask),
            (90, cv2.rotate(p_mask, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(p_mask, cv2.ROTATE_180)),
            (270, cv2.rotate(p_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        for angle, rot_img in rotations:
            c_norm, ar_norm = canonical_mask(rot_img, target_size=target_size)
            variants.append((angle, False, c_norm, ar_norm))

            mirrored_img = cv2.flip(rot_img, 1)
            c_mirr, ar_mirr = canonical_mask(mirrored_img, target_size=target_size)
            variants.append((angle, True, c_mirr, ar_mirr))

    return variants


def _align_and_score_pair(
    query_mask: np.ndarray,
    cand_mask: np.ndarray,
    query_ar: float,
    cand_ar: float,
    emb_sim: float,
    cfg: EngineConfig,
) -> Tuple[float, bool]:
    """
    Computes precise alignment, cutout IoU, symmetric Chamfer distance,
    and partial template match score.

    Returns:
        (final_score, is_partial) where score is in range [0, 100].
    """
    best_iou = 0.0
    best_aligned_q = query_mask

    shifts = [(-4, 0), (4, 0), (0, -4), (0, 4), (0, 0)]
    h, w = query_mask.shape[:2]

    c_cut = cand_mask == 0
    c_cut_sum = c_cut.sum()

    for dx, dy in shifts:
        if dx == 0 and dy == 0:
            shifted = query_mask
        else:
            M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            shifted = cv2.warpAffine(query_mask, M, (w, h), borderValue=255)

        q_cut = shifted == 0
        intersect = (q_cut & c_cut).sum()
        union = (q_cut | c_cut).sum()
        iou = float(intersect) / float(max(1, union))

        if iou > best_iou:
            best_iou = iou
            best_aligned_q = shifted

    q_cut = best_aligned_q == 0

    # Symmetric Chamfer distance using Distance Transforms
    dt_c = cv2.distanceTransform((cand_mask != 0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_q = cv2.distanceTransform((best_aligned_q != 0).astype(np.uint8), cv2.DIST_L2, 5)

    dist_q_to_c = float(dt_c[q_cut].mean()) if q_cut.any() else 48.0
    dist_c_to_q = float(dt_q[c_cut].mean()) if c_cut.any() else 48.0
    chamfer = (dist_q_to_c + dist_c_to_q) / 2.0
    norm_chamfer = min(1.0, chamfer / 40.0)
    chamfer_sim = 1.0 - norm_chamfer

    # Aspect ratio penalty
    ar_diff = abs(query_ar - cand_ar)
    ar_penalty = min(1.0, ar_diff / 1.5)

    # Partial template matching check
    is_partial = False
    best_match_val = 0.0

    if c_cut_sum > 100 and q_cut.sum() > 100:
        if (cand_ar > 2.0 and query_ar < 1.2) or (query_ar > 2.0 and cand_ar < 1.2):
            q_pts = np.argwhere(query_mask == 0)
            if len(q_pts) > 50:
                ymin, xmin = q_pts.min(axis=0)
                ymax, xmax = q_pts.max(axis=0)
                q_patch = query_mask[ymin : ymax + 1, xmin : xmax + 1]
                ph, pw = q_patch.shape[:2]
                if ph < cand_mask.shape[0] and pw < cand_mask.shape[1]:
                    try:
                        tm_res = cv2.matchTemplate(cand_mask, q_patch, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(tm_res)
                        if max_val > 0.70:
                            is_partial = True
                            best_match_val = float(max_val)
                    except Exception:
                        pass

    if is_partial:
        score = (
            cfg.WEIGHT_EMBEDDING * emb_sim
            + 0.40 * best_match_val
            + 0.15 * chamfer_sim
        ) * 100.0
    else:
        score = (
            cfg.WEIGHT_EMBEDDING * emb_sim
            + cfg.WEIGHT_IOU * best_iou
            + cfg.WEIGHT_CHAMFER * chamfer_sim
            - cfg.WEIGHT_ASPECT_PENALTY * ar_penalty
        ) * 100.0

    return max(0.0, min(100.0, score)), is_partial


def render_dxf_preview(
    dxf_path: Union[str, Path],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    target_size: int = 800,
) -> np.ndarray:
    """
    Renders a clean CAD preview from a DXF file.
    If bbox is provided (x, y, w, h in drawing units), highlights the matched design with a gold border.
    """
    from .dxf_reader import read_dxf
    from .linework_to_mask import linework_to_mask

    dxf_data = read_dxf(dxf_path)
    mask, _ = linework_to_mask(dxf_data.polylines, bbox=dxf_data.bbox)

    h, w = mask.shape[:2]
    # Brushed silver on navy background
    preview = np.full((h, w, 3), (25, 20, 15), dtype=np.uint8)
    preview[mask == 255] = (218, 218, 222)
    edges = cv2.Canny(mask, 100, 200)
    preview[edges > 0] = (70, 70, 75)

    if bbox is not None:
        bx, by, bw, bh = bbox
        min_x, min_y, max_x, max_y = dxf_data.bbox
        dw = max(1e-4, max_x - min_x)
        dh = max(1e-4, max_y - min_y)
        scale_x = float(w) / dw
        scale_y = float(h) / dh

        px0 = int(round((bx - min_x) * scale_x))
        py0 = int(round((max_y - (by + bh)) * scale_y))
        px1 = int(round((bx + bw - min_x) * scale_x))
        py1 = int(round((max_y - by) * scale_y))

        px0 = max(0, min(w - 1, px0))
        py0 = max(0, min(h - 1, py0))
        px1 = max(px0 + 1, min(w, px1))
        py1 = max(py0 + 1, min(h, py1))

        # Draw highlight rect in yellow/gold
        cv2.rectangle(preview, (px0, py0), (px1, py1), (0, 215, 255), 3)

    return preview


def open_source(result: MatchResult) -> None:
    """
    Opens the source file in the user's default CAD viewer (for DXF) or PDF viewer.
    Uses os.startfile on Windows with fallback to system opener on macOS/Linux.
    """
    target_path = result.dxf_path if result.source_type == "dxf" else result.pdf_path
    if not target_path:
        logger.warning("No target path in MatchResult to open.")
        return

    p = Path(target_path).resolve()
    if not p.exists():
        logger.warning("Source file does not exist: %s", target_path)
        return

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(p))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(p)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as exc:
        logger.error("Failed to open source file %s: %s", p, exc)


class Matcher:
    """Performs visual retrieval and geometric re-ranking against catalog index."""

    def __init__(
        self,
        store: Optional[IndexStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        cfg: EngineConfig = config,
    ) -> None:
        self.cfg = cfg
        self.store = store if store is not None else IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
        self.embedder = (
            embedder
            if embedder is not None
            else get_default_embedder(cfg.EMBEDDER_TYPE, cfg)
        )

    def warm_up(self) -> None:
        """Warms up the embedder model and loads index into memory."""
        dummy_mask = np.full((self.cfg.MASK_SIZE, self.cfg.MASK_SIZE), 255, dtype=np.uint8)
        self.embedder.embed(dummy_mask, 1.0)
        _ = self.store.index.ntotal

    def search_by_code(self, text: str, limit: int = 20) -> List[MatchResult]:
        """Performs case-insensitive, space-insensitive search on design labels."""
        records = self.store.search_by_code(text, limit=limit)
        results: List[MatchResult] = []
        for rec in records:
            res = MatchResult(
                folder=rec.folder,
                pdf_path=rec.pdf_path,
                page_no=rec.page_no,
                bbox=(rec.x, rec.y, rec.w, rec.h),
                score=100.0,
                rotation=0,
                mirrored=False,
                partial=False,
                thumb_png=rec.thumb_png,
                design_id=rec.id,
                file_type=rec.source_type,
                source_type=rec.source_type,
                dxf_path=rec.pdf_path if rec.source_type == "dxf" else None,
                label=rec.label,
                bbox_dxf=rec.bbox_dxf,
                design_code=rec.design_code or rec.label,
                polarity_variant=rec.polarity_variant,
                entity_count=rec.entity_count,
                design_index_info=rec.design_index_info,
            )
            if rec.same_design_group:
                related_recs = self.store.get_related_designs(rec.same_design_group, exclude_id=rec.id)
                res.related = [
                    MatchResult(
                        folder=r.folder,
                        pdf_path=r.pdf_path,
                        page_no=r.page_no,
                        bbox=(r.x, r.y, r.w, r.h),
                        score=100.0,
                        rotation=0,
                        mirrored=False,
                        partial=False,
                        thumb_png=r.thumb_png,
                        design_id=r.id,
                        file_type=r.source_type,
                        source_type=r.source_type,
                        dxf_path=r.pdf_path if r.source_type == "dxf" else None,
                        label=r.label,
                        bbox_dxf=r.bbox_dxf,
                        design_code=r.design_code or r.label,
                        polarity_variant=r.polarity_variant,
                        entity_count=r.entity_count,
                        design_index_info=r.design_index_info,
                    )
                    for r in related_recs
                ]
            results.append(res)
        return results

    def search(
        self,
        photo_bgr: np.ndarray,
        top_k: int = config.TOP_K,
        already_cropped: bool = False,
    ) -> SearchResponse:
        """
        Executes complete search pipeline:
        1. Preprocess & extract stencil mask with normalize_stencil.
        2. Generate geometric query variants (rotations 0/90/180/270 and mirroring).
        3. FAISS candidate retrieval.
        4. Re-ranking on mask IoU, Chamfer, and aspect ratio.
        5. Return top matches partitioned into DXF and PDF results with dedup linking.
        """
        if photo_bgr is None or photo_bgr.size == 0:
            return SearchResponse([], no_confident_match=True, reason="Input photo is empty.")

        # Step 1: Preprocess query photo
        from preprocess import normalize_stencil

        raw_mask, _ = normalize_stencil(photo_bgr, crop=not already_cropped)
        if raw_mask is None or raw_mask.size == 0:
            return SearchResponse([], no_confident_match=True, reason="Failed to produce stencil mask.")

        cutout_ratio = float((raw_mask == 0).mean())
        if cutout_ratio < 0.005 or cutout_ratio > 0.995:
            return SearchResponse(
                [],
                no_confident_match=True,
                reason="No distinct laser cut-out detected in input photo.",
            )

        # Step 2: Generate query variants
        variants = _generate_8_variants(raw_mask, target_size=self.cfg.MASK_SIZE)

        # Batch embed variants
        var_vectors = []
        for _, _, v_mask, v_ar in variants:
            var_vectors.append(self.embedder.embed(v_mask, v_ar))
        var_matrix = np.stack(var_vectors, axis=0)

        # Query FAISS index for all variants
        sim_scores, indices = self.store.search_vectors(var_matrix, top_k=self.cfg.RERANK_N)

        # Aggregate unique candidate design IDs with best embedding similarity
        candidate_best: Dict[int, dict] = {}
        for var_idx, (angle, is_mirr, v_mask, v_ar) in enumerate(variants):
            for score, d_id in zip(sim_scores[var_idx], indices[var_idx]):
                if d_id == -1:
                    continue
                score = float(score)

                if d_id not in candidate_best or score > candidate_best[d_id]["emb_sim"]:
                    candidate_best[d_id] = {
                        "design_id": d_id,
                        "emb_sim": score,
                        "rotation": angle,
                        "mirrored": is_mirr,
                        "variant_mask": v_mask,
                        "variant_ar": v_ar,
                    }

        if not candidate_best:
            return SearchResponse([], no_confident_match=True, reason="Catalog index is empty.")

        # Step 3: Fetch design metadata
        design_records = self.store.get_designs_batch(list(candidate_best.keys()))

        # Step 4: Re-rank candidates on actual canonical masks
        candidate_results: List[MatchResult] = []

        for d_id, cand_info in candidate_best.items():
            record = design_records.get(d_id)
            if not record:
                continue

            cand_mask = None
            if record.canonical_mask_png:
                try:
                    cand_mask = _decode_cached_mask(record.canonical_mask_png)
                except Exception:
                    pass

            # Fallback: re-render from file if not stored
            if cand_mask is None:
                try:
                    if record.source_type == "dxf" or record.pdf_path.lower().endswith(".dxf"):
                        from .dxf_reader import read_dxf
                        from .linework_to_mask import linework_to_mask
                        dxf_data = read_dxf(record.pdf_path)
                        dxf_stencil, _ = linework_to_mask(dxf_data.polylines, bbox=dxf_data.bbox)
                        cand_mask, _ = canonical_mask(dxf_stencil, target_size=self.cfg.MASK_SIZE)
                    else:
                        page_no = record.page_no if record.page_no is not None else 0
                        page_img = render_pdf_page(record.pdf_path, page_no, dpi=self.cfg.RENDER_DPI)
                        crop_bgr = page_img[record.y : record.y + record.h, record.x : record.x + record.w]
                        crop_mask, _ = normalize_stencil(crop_bgr, crop=False)
                        cand_mask, _ = canonical_mask(crop_mask, target_size=self.cfg.MASK_SIZE)
                except Exception as exc:
                    logger.warning("Failed to render candidate mask for %d: %s", d_id, exc)
                    continue

            final_score, is_partial = _align_and_score_pair(
                query_mask=cand_info["variant_mask"],
                cand_mask=cand_mask,
                query_ar=cand_info["variant_ar"],
                cand_ar=record.aspect_ratio,
                emb_sim=cand_info["emb_sim"],
                cfg=self.cfg,
            )

            file_type = "dxf" if record.source_type == "dxf" or record.pdf_path.lower().endswith(".dxf") else "pdf"
            dxf_p = record.pdf_path if file_type == "dxf" else None

            res_obj = MatchResult(
                folder=record.folder,
                pdf_path=record.pdf_path,
                page_no=record.page_no,
                bbox=(record.x, record.y, record.w, record.h),
                score=round(final_score, 1),
                rotation=cand_info["rotation"],
                mirrored=cand_info["mirrored"],
                partial=is_partial,
                thumb_png=record.thumb_png,
                design_id=record.id,
                file_type=file_type,
                source_type=record.source_type,
                dxf_path=dxf_p,
                label=record.label,
                bbox_dxf=record.bbox_dxf,
                design_code=record.design_code or record.label,
                polarity_variant=record.polarity_variant,
                entity_count=record.entity_count,
                design_index_info=record.design_index_info,
            )

            # Attach linked related designs from same_design_group
            if record.same_design_group:
                related_recs = self.store.get_related_designs(record.same_design_group, exclude_id=record.id)
                res_obj.related = [
                    MatchResult(
                        folder=r.folder,
                        pdf_path=r.pdf_path,
                        page_no=r.page_no,
                        bbox=(r.x, r.y, r.w, r.h),
                        score=round(final_score, 1),
                        rotation=cand_info["rotation"],
                        mirrored=cand_info["mirrored"],
                        partial=is_partial,
                        thumb_png=r.thumb_png,
                        design_id=r.id,
                        file_type=r.source_type,
                        source_type=r.source_type,
                        dxf_path=r.pdf_path if r.source_type == "dxf" else None,
                        label=r.label,
                        bbox_dxf=r.bbox_dxf,
                        design_code=r.design_code or r.label,
                        polarity_variant=r.polarity_variant,
                        entity_count=r.entity_count,
                        design_index_info=r.design_index_info,
                    )
                    for r in related_recs
                ]

            candidate_results.append(res_obj)

        # Sort by final score descending
        candidate_results.sort(key=lambda r: r.score, reverse=True)

        dxf_matches = [r for r in candidate_results if r.file_type == "dxf"][:5]
        pdf_matches = [r for r in candidate_results if r.file_type == "pdf"][:5]

        if dxf_matches and pdf_matches:
            combined = dxf_matches + pdf_matches
        elif dxf_matches:
            combined = candidate_results[:top_k]
        else:
            combined = candidate_results[:top_k]

        if not combined or combined[0].score < self.cfg.MIN_CONFIDENCE:
            return SearchResponse(
                results=[],
                dxf_results=[],
                pdf_results=[],
                no_confident_match=True,
                reason=f"Top match score ({combined[0].score if combined else 0}) below confidence threshold ({self.cfg.MIN_CONFIDENCE}).",
            )

        return SearchResponse(
            results=combined,
            dxf_results=dxf_matches,
            pdf_results=pdf_matches,
            no_confident_match=False,
        )


# Global convenience instances and functions
_default_matcher: Optional[Matcher] = None


def get_default_matcher() -> Matcher:
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = Matcher()
    return _default_matcher


def search(
    photo_bgr: np.ndarray,
    top_k: int = config.TOP_K,
    already_cropped: bool = False,
) -> SearchResponse:
    """Global convenience search function."""
    return get_default_matcher().search(photo_bgr, top_k=top_k, already_cropped=already_cropped)


def search_by_code(text: str, limit: int = 20) -> List[MatchResult]:
    """Global convenience search_by_code function."""
    return get_default_matcher().search_by_code(text, limit=limit)


def warm_up() -> None:
    """Global convenience warm-up function."""
    get_default_matcher().warm_up()

"""
Search engine for Design Matcher.
Processes customer photos into stencil masks, runs multi-rotation DINOv2 embedding queries,
searches FAISS vector index, aggregates best candidate matches, and retrieves catalog metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.core.embed import DINOv2Embedder
from app.core.image_crop import load_image_with_exif
from app.core.index import CatalogIndex
from app.core.preprocess import (
    convert_photo_to_stencil_mask,
    convert_tile_to_stencil_mask,
    detect_cutout_subpanels,
    letterbox_image,
    normalize_dxf_screenshot,
    normalize_stencil,
)
from app.core.rerank import compute_chamfer_similarity, compute_mask_iou

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a matched catalog design tile result."""
    rank: int
    score: float
    confidence_pct: float
    tile_id: int
    pdf_path: str
    pdf_name: str
    folder: str
    page_number: int
    tile_index: int
    crop_type: str
    bbox: List[int]
    design_number: Optional[str]
    thumbnail_path: Optional[str]
    best_rotation_deg: int
    iou: Optional[float] = None
    chamfer: Optional[float] = None
    matched_region: str = "Full"
    file_type: str = "pdf"
    design_code: Optional[str] = None
    dxf_bbox: Optional[Any] = None
    polarity_variant: int = 0
    entity_count: int = 0
    design_index_info: Optional[str] = None
    match_status: str = "Verified Match"
    geometric_distance: Optional[float] = None
    retrieval_similarity: Optional[float] = None
    is_mirrored: bool = False
    raw_dxf_match: Optional[Any] = None
    aspect_ratio: Optional[float] = None


def dxf_match_to_search_result(m: Any) -> SearchResult:
    """Converts an engine DxfMatch dataclass into a GUI SearchResult."""
    return SearchResult(
        rank=getattr(m, "rank", 1),
        score=getattr(m, "confidence_score", 0.0) / 100.0,
        confidence_pct=float(getattr(m, "confidence_score", 0.0)),
        tile_id=getattr(m, "component_index", 0),
        pdf_path=getattr(m, "file_path", ""),
        pdf_name=getattr(m, "filename", ""),
        folder=str(Path(getattr(m, "file_path", "")).parent),
        page_number=1,
        tile_index=getattr(m, "component_index", 0),
        crop_type="dxf",
        bbox=[0, 0, 100, 100],
        design_number=getattr(m, "component_name", ""),
        thumbnail_path=None,
        best_rotation_deg=getattr(m, "rotation_deg", 0),
        iou=getattr(m, "raster_similarity", None),
        chamfer=getattr(m, "geometric_distance", None),
        matched_region=getattr(m, "component_name", "Full"),
        file_type="dxf",
        design_code=getattr(m, "component_name", ""),
        dxf_bbox=None,
        polarity_variant=0,
        entity_count=getattr(m, "entity_count", 0),
        design_index_info=f"{getattr(m, 'component_name', '')} ({getattr(m, 'entity_count', 0)} entities)",
        match_status=getattr(m, "match_status", "Verified Match"),
        geometric_distance=getattr(m, "geometric_distance", None),
        retrieval_similarity=getattr(m, "retrieval_similarity", None),
        is_mirrored=getattr(m, "is_mirrored", False),
        raw_dxf_match=m,
        aspect_ratio=getattr(m, "aspect_ratio", None),
    )


def rotate_image(image: np.ndarray, angle_deg: int) -> np.ndarray:
    """Rotates an image by 0, 90, 180, or 270 degrees."""
    if angle_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


class DesignSearcher:
    """Coordinates photo preprocessing, multi-rotation vector search, and ranking."""

    def __init__(self, catalog_index: Optional[CatalogIndex] = None) -> None:
        self.index = catalog_index if catalog_index is not None else CatalogIndex()
        self.embedder = DINOv2Embedder.get_instance()

    def search_by_code(self, query: str, limit: int = 20) -> List[SearchResult]:
        """
        Searches the catalog for exact or prefix matches by design code immediately
        without performing vector image search.
        """
        import json

        matched_tiles = self.index.db.search_by_code(query, limit=limit)
        results: List[SearchResult] = []
        for rank, meta in enumerate(matched_tiles, start=1):
            is_dxf = (
                meta["pdf_path"].lower().endswith(".dxf")
                or meta.get("source_type") == "dxf"
                or meta.get("crop_type") in ("dxf", "dxf_panel")
            )
            ft = "dxf" if is_dxf else "pdf"
            d_code = meta.get("design_code") or meta.get("design_number")
            d_bbox = meta.get("dxf_bbox")
            if isinstance(d_bbox, str):
                try:
                    d_bbox = json.loads(d_bbox)
                except Exception:
                    pass

            results.append(
                SearchResult(
                    rank=rank,
                    score=1.0,
                    confidence_pct=100.0,
                    tile_id=meta["id"],
                    pdf_path=meta["pdf_path"],
                    pdf_name=Path(meta["pdf_path"]).name,
                    folder=meta["folder"],
                    page_number=meta["page_number"],
                    tile_index=meta["tile_index"],
                    crop_type=meta.get("crop_type", "full"),
                    bbox=meta["bbox"],
                    design_number=meta.get("design_number"),
                    thumbnail_path=meta.get("thumbnail_path"),
                    best_rotation_deg=0,
                    matched_region="Full",
                    file_type=ft,
                    design_code=d_code,
                    dxf_bbox=d_bbox,
                    polarity_variant=int(meta.get("polarity_variant", 0)),
                    entity_count=int(meta.get("entity_count", 0)),
                    design_index_info=meta.get("design_index_info"),
                )
            )
        return results

    def search_mask(
        self,
        mask: np.ndarray,
        top_k: int = 5,
        candidate_pool: int = 100,
        test_rotations: bool = True,
        rerank: bool = True,
        include_dxf: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Searches the catalog for the best matches given a binary stencil mask.
        Tests 4 rotations (0, 90, 180, 270 degrees) and horizontal flips across dual polarities
        and refines candidate matches using shape IoU and Chamfer distance.

        Args:
            mask: Binary stencil mask (uint8, 0/255).
            top_k: Number of final results to return per format (top 5 DXF + top 5 PDF).
            candidate_pool: Number of top FAISS neighbors to retrieve per query variant.
            test_rotations: If True, evaluates at 0, 90, 180, and 270 degrees.
            rerank: If True, computes IoU and Chamfer distance on top candidates.
            include_dxf: Whether to search DXF designs. Defaults to database setting 'include_dxf'.

        Returns:
            List of SearchResult items sorted by descending composite score.
        """
        if self.index.total_vectors == 0:
            logger.warning("Search called but FAISS index is empty.")
            return []

        if include_dxf is None:
            include_dxf = self.index.db.get_setting("include_dxf", "true").lower() in ("true", "1", "yes")

        # Load configurable matching criteria from settings
        allow_rot = True
        allow_mirr = True
        allow_pol = True
        rerank_strat = "Hybrid Vector + Linework (Recommended)"
        if hasattr(self.index, "db") and self.index.db:
            allow_rot = self.index.db.get_setting("allow_rotations", "1") == "1"
            allow_mirr = self.index.db.get_setting("allow_mirrored", "1") == "1"
            allow_pol = self.index.db.get_setting("allow_polarity", "1") == "1"
            rerank_strat = self.index.db.get_setting("rerank_mode", "Hybrid Vector + Linework (Recommended)")

        rotations = [0, 90, 180, 270] if (test_rotations and allow_rot) else [0]

        # Extract multi-scale query subregions (e.g. Full, Right Section, Center Section, Left Section)
        query_subregions = detect_cutout_subpanels(mask)
        if not query_subregions:
            query_subregions = [("Full", mask)]

        # Generate query variants across subregions, polarities, rotations, and flip mirroring
        # Each entry: (mask_224, angle, is_inverted, region_label, region_mask)
        query_variants: List[Tuple[np.ndarray, int, bool, str, np.ndarray]] = []
        for reg_idx, (reg_lbl, reg_m) in enumerate(query_subregions):
            polarities = [False, True] if (reg_idx == 0 and allow_pol) else [False]
            for is_inv in polarities:
                base_m = cv2.bitwise_not(reg_m) if is_inv else reg_m
                for angle in rotations:
                    rot_m = rotate_image(base_m, angle)
                    lbox_m = letterbox_image(rot_m, target_size=(224, 224), pad_value=0)
                    query_variants.append((lbox_m, angle, is_inv, reg_lbl, reg_m))

                    # Flip mirroring augmentation for full mask query if enabled
                    if reg_idx == 0 and allow_mirr:
                        mirrored_m = cv2.flip(rot_m, 1)
                        lbox_mirr = letterbox_image(mirrored_m, target_size=(224, 224), pad_value=0)
                        query_variants.append((lbox_mirr, angle, is_inv, f"{reg_lbl}_Mirrored", reg_m))

        query_masks = [v[0] for v in query_variants]

        # Batch embed all rotated & polarity-adjusted masks with DINOv2
        query_vectors = self.embedder.embed_masks(query_masks, batch_size=len(query_masks))

        # 1. FAISS search for general / PDF candidate pool
        pdf_scores_matrix, pdf_ids_matrix = self.index.search_vectors(query_vectors, top_k=candidate_pool)

        # 2. Dedicated DXF vector search if include_dxf is enabled
        if include_dxf:
            dxf_scores_matrix, dxf_ids_matrix = self.index.search_dxf_vectors(query_vectors, top_k=min(candidate_pool, 50))
        else:
            dxf_scores_matrix, dxf_ids_matrix = np.empty((len(query_vectors), 0)), np.empty((len(query_vectors), 0))

        # Collect unique tile IDs from both candidate pools
        all_retrieved_ids: set[int] = set()
        for row in pdf_ids_matrix:
            for tid in row:
                if tid >= 0:
                    all_retrieved_ids.add(int(tid))
        for row in dxf_ids_matrix:
            for tid in row:
                if tid >= 0:
                    all_retrieved_ids.add(int(tid))

        # Batch fetch metadata from SQLite
        tiles_meta = self.index.db.get_tiles_by_ids(list(all_retrieved_ids))

        # Group by candidate tile key: (pdf_path, page_number, tile_index)
        best_candidate_matches: dict[Tuple[str, int, int], dict[str, Any]] = {}

        # Aggregate PDF candidate matches
        for var_idx, (_, angle, is_inv, reg_lbl, reg_m) in enumerate(query_variants):
            if var_idx < len(pdf_scores_matrix):
                for score, tid in zip(pdf_scores_matrix[var_idx], pdf_ids_matrix[var_idx]):
                    if tid < 0:
                        continue
                    meta = tiles_meta.get(int(tid))
                    if not meta:
                        continue
                    cand_key = (meta["pdf_path"], meta["page_number"], meta["tile_index"])
                    curr_score = float(score)
                    if cand_key not in best_candidate_matches or curr_score > best_candidate_matches[cand_key]["cos_score"]:
                        best_candidate_matches[cand_key] = {
                            "cos_score": curr_score,
                            "tile_id": int(tid),
                            "meta": meta,
                            "rotation": angle,
                            "inverted": is_inv,
                            "var_idx": var_idx,
                            "matched_region": reg_lbl,
                            "query_submask": reg_m,
                        }

        # Aggregate DXF candidate matches
        for var_idx, (_, angle, is_inv, reg_lbl, reg_m) in enumerate(query_variants):
            if var_idx < len(dxf_scores_matrix):
                for score, tid in zip(dxf_scores_matrix[var_idx], dxf_ids_matrix[var_idx]):
                    if tid < 0:
                        continue
                    meta = tiles_meta.get(int(tid))
                    if not meta:
                        continue
                    cand_key = (meta["pdf_path"], meta["page_number"], meta["tile_index"])
                    curr_score = float(score)
                    if cand_key not in best_candidate_matches or curr_score > best_candidate_matches[cand_key]["cos_score"]:
                        best_candidate_matches[cand_key] = {
                            "cos_score": curr_score,
                            "tile_id": int(tid),
                            "meta": meta,
                            "rotation": angle,
                            "inverted": is_inv,
                            "var_idx": var_idx,
                            "matched_region": reg_lbl,
                            "query_submask": reg_m,
                        }

        # Partition candidates by format so both DXF and PDF are guaranteed deep reranking
        all_cands = list(best_candidate_matches.values())
        dxf_cos = [
            c for c in all_cands
            if c["meta"]["pdf_path"].lower().endswith(".dxf") or c["meta"].get("crop_type") in ("dxf", "dxf_panel")
        ]
        dxf_cos.sort(key=lambda x: x["cos_score"], reverse=True)
        dxf_cos = dxf_cos[: max(top_k * 5, 25)]

        pdf_cos = [
            c for c in all_cands
            if not (c["meta"]["pdf_path"].lower().endswith(".dxf") or c["meta"].get("crop_type") in ("dxf", "dxf_panel"))
        ]
        pdf_cos.sort(key=lambda x: x["cos_score"], reverse=True)
        pdf_cos = pdf_cos[: max(top_k * 5, 25)]

        candidates_to_evaluate = dxf_cos + pdf_cos
        if not candidates_to_evaluate:
            candidates_to_evaluate = sorted(all_cands, key=lambda x: x["cos_score"], reverse=True)[: max(top_k * 8, 40)]

        for cand in candidates_to_evaluate:
            meta = cand["meta"]
            cos_score = cand["cos_score"]
            var_idx = cand["var_idx"]
            query_aligned = query_variants[var_idx][0]

            iou_score = None
            chamfer_score = None
            composite_score = cos_score

            if rerank and meta.get("thumbnail_path"):
                thumb_rel = meta["thumbnail_path"]
                thumb_full = self.index.data_dir / thumb_rel
                mask_full = thumb_full.with_name(thumb_full.stem + "_mask.png")

                cand_mask_224 = None
                if mask_full.is_file():
                    cand_mask_224 = cv2.imread(str(mask_full), cv2.IMREAD_GRAYSCALE)
                if cand_mask_224 is None and thumb_full.is_file():
                    thumb_bgr = cv2.imread(str(thumb_full))
                    if thumb_bgr is not None:
                        cand_mask = convert_tile_to_stencil_mask(thumb_bgr)
                        cand_mask_224 = letterbox_image(cand_mask, target_size=(224, 224), pad_value=0)

                if cand_mask_224 is not None:
                    if cand_mask_224.shape != query_aligned.shape:
                        cand_mask_224 = letterbox_image(
                            cand_mask_224,
                            target_size=(query_aligned.shape[1], query_aligned.shape[0]),
                            pad_value=0,
                        )
                    iou_val = compute_mask_iou(query_aligned, cand_mask_224)
                    chamfer_val = compute_chamfer_similarity(query_aligned, cand_mask_224)

                    iou_score = round(iou_val, 4)
                    chamfer_score = round(chamfer_val, 4)

                    # Dynamic composite weighting based on user-configured reranking strategy
                    rerank_lower = rerank_strat.lower()
                    if "exact linework" in rerank_lower:
                        composite_score = 0.20 * cos_score + 0.20 * iou_val + 0.60 * chamfer_val
                    elif "fast vector" in rerank_lower:
                        composite_score = 0.80 * cos_score + 0.20 * iou_val
                    else:
                        # Balanced Hybrid: 50% feature embedding, 30% pixel IoU, 20% edge Chamfer
                        composite_score = 0.50 * cos_score + 0.30 * iou_val + 0.20 * chamfer_val

            cand["final_score"] = composite_score
            cand["iou"] = iou_score
            cand["chamfer"] = chamfer_score

        # Sort final candidates by composite score
        sorted_candidates = sorted(
            candidates_to_evaluate,
            key=lambda x: x["final_score"],
            reverse=True,
        )

        dxf_candidates = [
            c for c in sorted_candidates
            if c["meta"]["pdf_path"].lower().endswith(".dxf") or c["meta"].get("crop_type") in ("dxf", "dxf_panel")
        ]
        pdf_candidates = [
            c for c in sorted_candidates
            if not (c["meta"]["pdf_path"].lower().endswith(".dxf") or c["meta"].get("crop_type") in ("dxf", "dxf_panel"))
        ]

        top_dxf = dxf_candidates[:top_k]
        top_pdf = pdf_candidates[:top_k]

        results: List[SearchResult] = []

        import json

        def _make_result(rank_idx: int, cand_dict: dict, ft_override: Optional[str] = None) -> SearchResult:
            m = cand_dict["meta"]
            f_score = cand_dict["final_score"]
            c_pct = max(0.0, min(100.0, f_score * 100.0))
            is_d = (
                m["pdf_path"].lower().endswith(".dxf")
                or m.get("source_type") == "dxf"
                or m.get("crop_type") in ("dxf", "dxf_panel")
            )
            ft = ft_override or ("dxf" if is_d else "pdf")
            d_bbox = m.get("dxf_bbox")
            if isinstance(d_bbox, str):
                try:
                    d_bbox = json.loads(d_bbox)
                except Exception:
                    pass
            d_code = m.get("design_code") or m.get("design_number")

            # Calculate design aspect ratio (width / height)
            calc_ar: Optional[float] = None
            if m.get("bbox") and len(m["bbox"]) == 4:
                bh = max(1, m["bbox"][2] - m["bbox"][0])
                bw = max(1, m["bbox"][3] - m["bbox"][1])
                calc_ar = round(float(bw) / float(bh), 2)
            elif m.get("aspect_ratio"):
                try:
                    calc_ar = round(float(m["aspect_ratio"]), 2)
                except Exception:
                    pass

            return SearchResult(
                rank=rank_idx,
                score=round(f_score, 4),
                confidence_pct=round(c_pct, 1),
                tile_id=cand_dict["tile_id"],
                pdf_path=m["pdf_path"],
                pdf_name=Path(m["pdf_path"]).name,
                folder=m["folder"],
                page_number=m["page_number"],
                tile_index=m["tile_index"],
                crop_type=m["crop_type"],
                bbox=m["bbox"],
                design_number=m.get("design_number"),
                thumbnail_path=m.get("thumbnail_path"),
                best_rotation_deg=cand_dict["rotation"],
                iou=cand_dict.get("iou"),
                chamfer=cand_dict.get("chamfer"),
                matched_region=cand_dict.get("matched_region", "Full"),
                file_type=ft,
                design_code=d_code,
                dxf_bbox=d_bbox,
                polarity_variant=int(m.get("polarity_variant", 0)),
                entity_count=int(m.get("entity_count", 0)),
                design_index_info=m.get("design_index_info"),
                aspect_ratio=calc_ar,
            )

        # Add top DXF candidates
        for rank, cand in enumerate(top_dxf, start=1):
            results.append(_make_result(rank, cand, ft_override="dxf"))

        # Add top PDF candidates
        for rank, cand in enumerate(top_pdf, start=1):
            results.append(_make_result(rank, cand, ft_override="pdf"))

        # Fallback if both empty but sorted_candidates has entries
        if not results and sorted_candidates:
            for rank, cand in enumerate(sorted_candidates[:top_k], start=1):
                results.append(_make_result(rank, cand))

        return results

    def search_photo_file(self, photo_path: Path | str, top_k: int = 5) -> List[SearchResult]:
        """
        Loads an image file, detects sheet boundary and perspective corrects,
        converts to normalized stencil mask, and searches the catalog.
        Automatically uses normalize_dxf_screenshot for CAD viewport screenshots.

        Args:
            photo_path: Path to user image file.
            top_k: Number of results to return.

        Returns:
            List of top search results.
        """
        p = Path(photo_path)
        if not p.is_file():
            raise FileNotFoundError(f"Photo file not found: {photo_path}")

        try:
            img_bgr = load_image_with_exif(p)
        except Exception:
            img_bgr = None

        if img_bgr is None:
            raise ValueError(f"Unable to read image file: {photo_path}")

        mask, _ = normalize_stencil(img_bgr, crop=True)
        return self.search_mask(mask, top_k=top_k)

    def search_dxf_screenshot(
        self,
        screenshot_bgr: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Searches the catalog using a DXF viewport screenshot as the query.

        Unlike ``search_mask`` which expects a photo from a camera, this method
        uses ``normalize_dxf_screenshot`` to convert the wireframe screenshot
        into the same FILLED stencil convention used by the vector indexer
        (linework_to_mask → XOR polygon fill, white=metal, black=cut-outs).

        Use this whenever the query image is:
        - A screenshot / render exported from your DXF viewer
        - A programmatic render via dxf_viewer.py or cv2 DXF render
        - Any CAD viewport capture (uniform background, thin strokes)

        Args:
            screenshot_bgr: BGR numpy array of the DXF screenshot.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult items sorted by descending composite score.
        """
        if screenshot_bgr is None or screenshot_bgr.size == 0:
            logger.warning("search_dxf_screenshot: empty image provided.")
            return []

        mask, info = normalize_dxf_screenshot(screenshot_bgr)
        logger.debug(
            "search_dxf_screenshot: filled stencil from DXF screenshot, "
            "metal=%.1f%%, method=%s",
            100.0 * (mask == 255).mean(),
            info.get("method"),
        )

        if mask is None or mask.size == 0 or (mask == 255).mean() > 0.98:
            logger.warning(
                "search_dxf_screenshot: stencil mask is nearly all-white — "
                "screenshot may not contain a DXF design."
            )
            return []

        return self.search_mask(mask, top_k=top_k)

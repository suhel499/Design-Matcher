"""
dxf_matcher.py - Hybrid Matching Engine for DXF Designs.

Implements the multi-stage matching architecture:
1. Mode A: PNG Stencil -> DXF Search.
2. Mode B: Reference DXF -> DXF Search.

Stages:
- Stage 1: Candidate Retrieval using normalized feature descriptors (radial histograms,
  pairwise distance distributions, Hu moments, aspect ratio).
- Stage 2: Geometric Verification using scipy.spatial.cKDTree for bidirectional Chamfer distance,
  rotation-aware alignment (0, 90, 180, 270 deg and horizontal flip mirroring), fine angular
  refinement, and raster contour validation.
- Stage 3: Calibrated Confidence and Match Status ('Verified Match', 'Possible Match', 'No Reliable Match').
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .dxf_database import DxfDatabase
from .dxf_normalization import (
    DESCRIPTOR_DIM,
    NormalizedGeometry,
    normalize_dxf_component,
    normalize_raster_stencil,
)
from .dxf_parser import parse_dxf_drawing

logger = logging.getLogger(__name__)


@dataclass
class DxfMatch:
    """Represents a ranked, geometrically verified DXF matching result."""
    rank: int
    file_path: str
    filename: str
    component_name: str
    component_index: int
    confidence_score: float             # Calibrated confidence: 0.0 to 100.0
    match_status: str                   # 'Verified Match', 'Possible Match', or 'No Reliable Match'
    geometric_distance: float           # Bidirectional Chamfer distance in normalized units
    retrieval_similarity: float         # Stage 1 cosine similarity (0.0 to 1.0)
    raster_similarity: float            # Stage 2 mask / edge similarity (0.0 to 1.0)
    aspect_ratio: float
    entity_count: int
    rotation_deg: int                   # Best alignment angle: 0, 90, 180, or 270 degrees
    is_mirrored: bool                   # True if mirrored/flipped matched best
    preview_png: Optional[bytes] = None
    canonical_mask_png: Optional[bytes] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return self.match_status == "Verified Match"


@dataclass
class DxfSearchResponse:
    """Complete response from a DXF design search."""
    matches: List[DxfMatch]
    best_match: Optional[DxfMatch] = None
    query_type: str = "mode_b_dxf"     # 'mode_a_png' or 'mode_b_dxf'
    query_name: str = ""
    total_candidates_evaluated: int = 0
    search_latency_ms: float = 0.0
    status_summary: str = ""

    def __post_init__(self):
        if self.matches and self.best_match is None:
            self.best_match = self.matches[0]


def _rotate_points_2d(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates a 2D point cloud around (0, 0) by angle_deg degrees."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return pts @ rot_mat.T


def compute_bidirectional_chamfer(
    query_pts: np.ndarray,
    cand_pts: np.ndarray,
    clip_distance: float = 0.5,
) -> float:
    """
    Computes trimmed bidirectional Chamfer distance between two normalized point clouds using cKDTree.
    Robust against minor modifications, local cuts, and outliers.
    """
    if len(query_pts) == 0 or len(cand_pts) == 0:
        return clip_distance

    tree_cand = cKDTree(cand_pts)
    tree_query = cKDTree(query_pts)

    d_q2c, _ = tree_cand.query(query_pts)
    d_c2q, _ = tree_query.query(cand_pts)

    # Trim 90th percentile to reject spurious text/label outlier points
    q_trimmed = np.minimum(d_q2c, clip_distance)
    c_trimmed = np.minimum(d_c2q, clip_distance)

    q_dist = float(np.percentile(q_trimmed, 90))
    c_dist = float(c_trimmed.mean())

    return float(0.45 * q_dist + 0.55 * c_dist)


def align_and_verify_point_clouds(
    query_pts: np.ndarray,
    cand_pts: np.ndarray,
) -> Tuple[float, int, bool]:
    """
    Evaluates 8 primary orientations (0, 90, 180, 270 degrees x normal/mirrored)
    and tests fine angular refinement (±3, ±6 degrees) to find minimal Chamfer distance.

    Returns:
        (best_distance, best_rotation_deg, is_mirrored)
    """
    best_dist = float("inf")
    best_rot = 0
    best_mirr = False

    orientations = [
        (0, False, query_pts),
        (90, False, _rotate_points_2d(query_pts, 90)),
        (180, False, _rotate_points_2d(query_pts, 180)),
        (270, False, _rotate_points_2d(query_pts, 270)),
    ]

    # Mirrored variants
    query_flipped = query_pts.copy()
    query_flipped[:, 0] = -query_flipped[:, 0]
    orientations.extend([
        (0, True, query_flipped),
        (90, True, _rotate_points_2d(query_flipped, 90)),
        (180, True, _rotate_points_2d(query_flipped, 180)),
        (270, True, _rotate_points_2d(query_flipped, 270)),
    ])

    best_oriented_pts = query_pts

    for angle, mirr, pts_orient in orientations:
        d = compute_bidirectional_chamfer(pts_orient, cand_pts)
        if d < best_dist:
            best_dist = d
            best_rot = angle
            best_mirr = mirr
            best_oriented_pts = pts_orient

    # Fine angular refinement around best orientation
    if best_dist < 0.25:
        for delta_ang in (-6.0, -3.0, 3.0, 6.0):
            refined = _rotate_points_2d(best_oriented_pts, delta_ang)
            d_fine = compute_bidirectional_chamfer(refined, cand_pts)
            if d_fine < best_dist:
                best_dist = d_fine

    return best_dist, best_rot, best_mirr


def compute_raster_iou(query_mask: np.ndarray, cand_mask: np.ndarray) -> float:
    """Computes Intersection-over-Union (IoU) of cutout holes (0 pixels)."""
    if query_mask.shape != cand_mask.shape:
        cand_mask = cv2.resize(cand_mask, (query_mask.shape[1], query_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    q_cut = query_mask == 0
    c_cut = cand_mask == 0

    intersect = float((q_cut & c_cut).sum())
    union = float((q_cut | c_cut).sum())

    return (intersect / union) if union > 0 else 0.5


class DxfMatchingEngine:
    """
    Production-ready hybrid matching engine for DXF and stencil designs.
    """

    def __init__(self, database: DxfDatabase) -> None:
        self.db = database

    def search_by_normalized_geometry(
        self,
        query_norm: NormalizedGeometry,
        top_k: int = 10,
        candidate_pool: int = 40,
        min_confidence: float = 40.0,
    ) -> DxfSearchResponse:
        """
        Executes complete hybrid search pipeline:
        1. Stage 1: Candidate retrieval via vector similarity.
        2. Stage 2: Geometric verification (cKDTree bidirectional Chamfer distance & rotation-aware alignment).
        3. Stage 3: Calibrated confidence calculation and ranking.
        """
        start_time = time.perf_counter()

        # Stage 1: Fast candidate retrieval across primary orientations (0 and 90 deg)
        query_variants = [(0, query_norm.feature_vector, query_norm.aspect_ratio)]

        # 90-degree rotated variant: aspect ratio inverted
        inv_ar = float(1.0 / max(1e-4, query_norm.aspect_ratio))
        rot_90_pts = _rotate_points_2d(query_norm.point_cloud, 90.0)
        rot_90_mask = cv2.rotate(query_norm.canonical_mask, cv2.ROTATE_90_CLOCKWISE)
        grid_90 = cv2.resize(rot_90_mask, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
        g_norm = np.linalg.norm(grid_90)
        if g_norm > 1e-7:
            grid_90 = grid_90 / g_norm

        stats_90 = np.array([
            float(math.log(max(1e-2, min(100.0, inv_ar)))),
            float((rot_90_mask == 255).mean()),
            min(1.0, query_norm.contour_count / 50.0),
            min(1.0, query_norm.contour_count / 30.0),
            min(1.0, query_norm.hole_count / 30.0),
            1.0,
            float(rot_90_pts[:, 0].std()),
            float(rot_90_pts[:, 1].std()),
            float(np.hypot(rot_90_pts[:, 0], rot_90_pts[:, 1]).std()),
        ], dtype=np.float32)

        fv_90 = np.concatenate([query_norm.radial_hist, query_norm.pairwise_hist, query_norm.hu_moments, stats_90, grid_90[:48]]).astype(np.float32)
        fv_90_norm = np.linalg.norm(fv_90)
        if fv_90_norm > 1e-7:
            fv_90 = fv_90 / fv_90_norm
        query_variants.append((90, fv_90, inv_ar))

        # Query candidates for both orientations and merge
        cand_dict: Dict[int, Dict[str, Any]] = {}
        for _, fv_v, ar_v in query_variants:
            cands_v = self.db.retrieve_candidates(
                query_vector=fv_v,
                query_aspect_ratio=ar_v,
                top_k=candidate_pool,
            )
            for c in cands_v:
                cid = c["component_id"]
                if cid not in cand_dict or c["retrieval_score"] > cand_dict[cid]["retrieval_score"]:
                    cand_dict[cid] = c

        candidates = list(cand_dict.values())

        if not candidates:
            latency = (time.perf_counter() - start_time) * 1000.0
            return DxfSearchResponse(
                matches=[],
                query_name=query_norm.component_name,
                total_candidates_evaluated=0,
                search_latency_ms=latency,
                status_summary="No candidates found in index.",
            )

        # Stage 2: Geometric Verification on retrieved candidates
        verified_results: List[DxfMatch] = []

        for cand in candidates:
            comp_id = cand["component_id"]
            geom_data = self.db.get_component_geometry(comp_id)
            if not geom_data:
                continue

            cand_pts = geom_data["point_cloud"]
            chamfer_dist, best_rot, is_mirr = align_and_verify_point_clouds(query_norm.point_cloud, cand_pts)

            # Raster IoU similarity
            raster_sim = 0.5
            if geom_data["canonical_mask_png"]:
                try:
                    c_mask = cv2.imdecode(np.frombuffer(geom_data["canonical_mask_png"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if c_mask is not None:
                        # Align query mask with best rotation
                        rot_mask = query_norm.canonical_mask
                        if best_rot == 90:
                            rot_mask = cv2.rotate(rot_mask, cv2.ROTATE_90_CLOCKWISE)
                        elif best_rot == 180:
                            rot_mask = cv2.rotate(rot_mask, cv2.ROTATE_180)
                        elif best_rot == 270:
                            rot_mask = cv2.rotate(rot_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        if is_mirr:
                            rot_mask = cv2.flip(rot_mask, 1)

                        raster_sim = compute_raster_iou(rot_mask, c_mask)
                except Exception as e:
                    logger.debug("Error computing raster IoU: %s", e)

            # Stage 3: Calibrated Confidence Calculation
            if chamfer_dist <= 0.04:
                geom_sim = 1.0
            elif chamfer_dist <= 0.16:
                geom_sim = max(0.50, 1.0 - (chamfer_dist - 0.04) / 0.18)
            else:
                geom_sim = max(0.0, 0.50 * (1.0 - (chamfer_dist - 0.16) / 0.20))

            retrieval_sim = max(0.0, min(1.0, cand["retrieval_score"]))

            # Calibrated composite similarity (weighted 50% vector geometry, 30% raster cutout IoU, 20% retrieval)
            composite_similarity = 0.50 * geom_sim + 0.30 * raster_sim + 0.20 * retrieval_sim
            confidence_score = round(float(np.clip(composite_similarity * 100.0, 0.0, 100.0)), 1)

            # Determine rigorous match status
            if confidence_score >= 80.0 and chamfer_dist <= 0.14:
                match_status = "Verified Match"
            elif confidence_score >= 55.0 and chamfer_dist <= 0.26:
                match_status = "Possible Match"
            else:
                match_status = "No Reliable Match"

            match_item = DxfMatch(
                rank=0,
                file_path=cand["file_path"],
                filename=cand["filename"],
                component_name=cand["name"],
                component_index=cand["component_index"],
                confidence_score=confidence_score,
                match_status=match_status,
                geometric_distance=round(chamfer_dist, 4),
                retrieval_similarity=round(retrieval_sim, 4),
                raster_similarity=round(raster_sim, 4),
                aspect_ratio=cand["aspect_ratio"],
                entity_count=cand["entity_count"],
                rotation_deg=best_rot,
                is_mirrored=is_mirr,
                preview_png=geom_data["thumb_png"],
                canonical_mask_png=geom_data["canonical_mask_png"],
                bbox=cand.get("bbox"),
            )
            verified_results.append(match_item)

        # Sort by confidence score descending
        verified_results.sort(key=lambda m: m.confidence_score, reverse=True)

        # Assign ranks
        for idx, m in enumerate(verified_results, start=1):
            m.rank = idx

        latency = (time.perf_counter() - start_time) * 1000.0

        top_matches = verified_results[:top_k]
        best_m = top_matches[0] if top_matches else None

        # Filter out if top match is below min_confidence threshold
        status_summary = "Matches found."
        if not top_matches or (best_m and best_m.confidence_score < min_confidence):
            status_summary = "No reliable matches found above confidence threshold."

        # Record search history in database
        self.db.record_search(
            query_type="dxf_hybrid",
            query_path=query_norm.component_name,
            best_score=best_m.confidence_score if best_m else 0.0,
            best_status=best_m.match_status if best_m else "None",
            matched_file=best_m.file_path if best_m else None,
            latency_ms=latency,
        )

        return DxfSearchResponse(
            matches=top_matches,
            best_match=best_m,
            query_type="dxf_hybrid",
            query_name=query_norm.component_name,
            total_candidates_evaluated=len(candidates),
            search_latency_ms=round(latency, 1),
            status_summary=status_summary,
        )

    def search_by_reference_dxf(
        self,
        reference_dxf_path: Union[str, Path],
        top_k: int = 10,
        min_confidence: float = 40.0,
        candidate_pool: int = 40,
        candidate_limit: Optional[int] = None,
        chamfer_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> DxfSearchResponse:
        """
        Mode B: Reference DXF -> DXF Search.
        Takes a user's reference .dxf file, normalizes geometry, and searches the indexed DXF library.
        """
        p = Path(reference_dxf_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Reference DXF file not found: {p}")

        parsed = parse_dxf_drawing(p)
        if not parsed.components:
            return DxfSearchResponse(
                matches=[],
                query_type="mode_b_dxf",
                query_name=p.name,
                status_summary=f"Reference DXF '{p.name}' contains no readable geometry.",
            )

        # Use primary component
        primary_comp = parsed.components[0]
        norm_geo = normalize_dxf_component(primary_comp)
        norm_geo.component_name = f"{p.name} ({primary_comp.name})"

        pool = candidate_limit if candidate_limit is not None else candidate_pool
        resp = self.search_by_normalized_geometry(
            query_norm=norm_geo,
            top_k=top_k,
            candidate_pool=pool,
            min_confidence=min_confidence,
        )
        resp.query_type = "mode_b_dxf"
        resp.query_name = p.name
        return resp

    def search_by_png_stencil(
        self,
        image: Union[np.ndarray, str, Path],
        top_k: int = 10,
        min_confidence: float = 40.0,
        candidate_pool: int = 40,
        candidate_limit: Optional[int] = None,
        chamfer_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> DxfSearchResponse:
        """
        Mode A: PNG Stencil -> DXF Search.
        Takes a PNG image / stencil mask, normalizes it into compatible vector descriptors,
        and searches the indexed DXF library.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image).resolve()
            if not img_path.is_file():
                raise FileNotFoundError(f"Query image not found: {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Could not decode image at {img_path}")
            label = img_path.name
        else:
            bgr = image
            label = "Uploaded Photo / Stencil"

        # Stencil preprocessing
        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr

        # Normalize stencil
        norm_geo = normalize_raster_stencil(gray, label=label)

        pool = candidate_limit if candidate_limit is not None else candidate_pool
        resp = self.search_by_normalized_geometry(
            query_norm=norm_geo,
            top_k=top_k,
            candidate_pool=pool,
            min_confidence=min_confidence,
        )
        resp.query_type = "mode_a_png"
        resp.query_name = label
        return resp


# Convenient alias for DxfMatchingEngine
DxfMatcher = DxfMatchingEngine


def search_by_reference_dxf(
    arg1: Union[DxfDatabase, DxfMatchingEngine, str, Path],
    arg2: Optional[Union[str, Path, DxfDatabase, DxfMatchingEngine]] = None,
    top_k: int = 10,
    min_confidence: float = 40.0,
    candidate_limit: int = 40,
    chamfer_threshold: float = 0.12,
    **kwargs: Any,
) -> List[DxfMatch]:
    """
    Module-level convenience function for Mode B (Reference DXF -> DXF Search).
    Accepts either (database, ref_dxf_path) or (ref_dxf_path, database=db).
    Returns a list of ranked DxfMatch results.
    """
    if isinstance(arg1, (DxfDatabase, DxfMatchingEngine)):
        engine = arg1 if isinstance(arg1, DxfMatchingEngine) else DxfMatchingEngine(arg1)
        ref_path = arg2
    elif isinstance(arg2, (DxfDatabase, DxfMatchingEngine)):
        engine = arg2 if isinstance(arg2, DxfMatchingEngine) else DxfMatchingEngine(arg2)
        ref_path = arg1
    else:
        db = kwargs.get("db") or kwargs.get("database")
        if db is None:
            raise ValueError("A DxfDatabase or DxfMatchingEngine must be provided to search_by_reference_dxf.")
        engine = db if isinstance(db, DxfMatchingEngine) else DxfMatchingEngine(db)
        ref_path = arg1

    if ref_path is None:
        raise ValueError("reference_dxf_path must be provided to search_by_reference_dxf.")

    resp = engine.search_by_reference_dxf(
        reference_dxf_path=ref_path,
        top_k=top_k,
        min_confidence=min_confidence,
        candidate_limit=candidate_limit,
        chamfer_threshold=chamfer_threshold,
        **kwargs,
    )
    return resp.matches


def search_by_png_stencil(
    arg1: Union[DxfDatabase, DxfMatchingEngine, np.ndarray, str, Path],
    arg2: Optional[Union[np.ndarray, str, Path, DxfDatabase, DxfMatchingEngine]] = None,
    top_k: int = 10,
    min_confidence: float = 40.0,
    candidate_limit: int = 40,
    chamfer_threshold: float = 0.12,
    **kwargs: Any,
) -> List[DxfMatch]:
    """
    Module-level convenience function for Mode A (PNG Stencil -> DXF Search).
    Accepts either (database, image) or (image, database=db).
    Returns a list of ranked DxfMatch results.
    """
    if isinstance(arg1, (DxfDatabase, DxfMatchingEngine)):
        engine = arg1 if isinstance(arg1, DxfMatchingEngine) else DxfMatchingEngine(arg1)
        image = arg2
    elif isinstance(arg2, (DxfDatabase, DxfMatchingEngine)):
        engine = arg2 if isinstance(arg2, DxfMatchingEngine) else DxfMatchingEngine(arg2)
        image = arg1
    else:
        db = kwargs.get("db") or kwargs.get("database")
        if db is None:
            raise ValueError("A DxfDatabase or DxfMatchingEngine must be provided to search_by_png_stencil.")
        engine = db if isinstance(db, DxfMatchingEngine) else DxfMatchingEngine(db)
        image = arg1

    if image is None:
        raise ValueError("image must be provided to search_by_png_stencil.")

    resp = engine.search_by_png_stencil(
        image=image,
        top_k=top_k,
        min_confidence=min_confidence,
        candidate_limit=candidate_limit,
        chamfer_threshold=chamfer_threshold,
        **kwargs,
    )
    return resp.matches

"""
Engine configuration constants and settings.
All constants have sane defaults and can be configured per deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EngineConfig:
    """Configuration settings for PDF rendering, indexing, and matching."""

    # PDF Rendering
    RENDER_DPI: int = 200

    # Mask & Feature Constants
    MASK_SIZE: int = 256
    MIN_DESIGN_AREA: int = 2500  # min bounding box area (w * h) at 200 DPI
    MAX_ASPECT_RATIO: float = 8.0  # ignore banners, slivers, headers
    THUMB_SIZE: int = 120  # thumbnail size (px)

    # Retrieval and Re-ranking
    TOP_K: int = 10
    RERANK_N: int = 20
    MIN_CONFIDENCE: float = 40.0  # scores below this trigger no_confident_match

    # Scoring Weights
    WEIGHT_EMBEDDING: float = 0.45
    WEIGHT_IOU: float = 0.35
    WEIGHT_CHAMFER: float = 0.20
    WEIGHT_ASPECT_PENALTY: float = 0.15

    # Storage Paths
    DB_PATH: Path = Path(os.environ.get("DESIGN_MATCHER_DB", "catalogs.db"))
    INDEX_PATH: Path = Path(os.environ.get("DESIGN_MATCHER_INDEX", "vectors.index"))
    MODELS_DIR: Path = Path(os.environ.get("DESIGN_MATCHER_MODELS", "models"))

    # Hardware / Concurrency
    EMBEDDER_TYPE: str = os.environ.get("DESIGN_MATCHER_EMBEDDER", "classical")
    MAX_WORKERS: int = max(1, min(4, (os.cpu_count() or 2) - 1))

    # DXF Linework & Mask Configuration
    MASK_RENDER_SIZE: int = 1600
    DXF_IGNORE_LAYERS: Optional[list[str]] = None
    DXF_GAP_TOLERANCE_RATIO: float = 0.001  # 0.1% of design diagonal
    DXF_FLATTEN_DIST_RATIO: float = 0.0005  # 0.05% of drawing diagonal
    DXF_OPEN_STROKE_RATIO: float = 0.003   # 0.3% of long side
    DXF_MAX_BLOCK_DEPTH: int = 8

    def __post_init__(self):
        if self.DXF_IGNORE_LAYERS is None:
            self.DXF_IGNORE_LAYERS = [
                "dim",
                "defpoints",
                "text",
                "grid",
                "axis",
                "construction",
                "hatch_fill",
                "viewport",
            ]


# Global default configuration instance
config = EngineConfig()

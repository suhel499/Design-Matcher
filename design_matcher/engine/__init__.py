"""
Design Matcher Engine: High-performance, UI-independent catalog indexing and search engine.
"""

from __future__ import annotations

from .config import EngineConfig, config
from .dxf_reader import DxfDrawingData, DxfLabel, DxfReaderError, read_dxf
from .dxf_render import (
    generate_dxf_thumbnail,
    get_dxf_metadata,
    render_dxf_to_stencil,
)
from .dxf_segment import DxfDesign, segment_dxf_drawing
from .features import BaseEmbedder, ClassicalEmbedder, canonical_mask, get_default_embedder
from .index_store import IndexStore
from .indexer import IndexStats, find_catalog_files, find_pdf_files, index_folders
from .linework_to_mask import is_outline_only, linework_to_mask, outline_to_mask
from .matcher import Matcher, open_source, render_dxf_preview, search, search_by_code, warm_up
from .models import DesignRecord, MatchResult, SearchResponse
from .pdf_render import get_pdf_page_count, render_pdf_page
from .segment import segment_designs
from .utils import (
    compute_file_fingerprint,
    open_pdf_at_page,
    overlay_diff,
    record_feedback,
    render_page_preview,
)

from .dxf_database import DxfDatabase, IndexingProgress
from .dxf_matcher import (
    DxfMatch,
    DxfMatchingEngine,
    DxfSearchResponse,
    search_by_png_stencil,
    search_by_reference_dxf,
)
from .dxf_normalization import (
    NormalizedGeometry,
    normalize_dxf_component,
    normalize_raster_stencil,
)
from .dxf_parser import DxfParserError, ParsedDxfDrawing, parse_dxf_drawing

__all__ = [
    "BaseEmbedder",
    "ClassicalEmbedder",
    "DesignRecord",
    "DxfDatabase",
    "DxfDesign",
    "DxfDrawingData",
    "DxfLabel",
    "DxfMatch",
    "DxfMatchingEngine",
    "DxfParserError",
    "DxfReaderError",
    "DxfSearchResponse",
    "EngineConfig",
    "IndexStats",
    "IndexStore",
    "IndexingProgress",
    "MatchResult",
    "Matcher",
    "NormalizedGeometry",
    "ParsedDxfDrawing",
    "SearchResponse",
    "canonical_mask",
    "compute_file_fingerprint",
    "config",
    "find_catalog_files",
    "find_pdf_files",
    "generate_dxf_thumbnail",
    "get_default_embedder",
    "get_dxf_metadata",
    "get_pdf_page_count",
    "index_folders",
    "is_outline_only",
    "linework_to_mask",
    "normalize_dxf_component",
    "normalize_raster_stencil",
    "open_pdf_at_page",
    "open_source",
    "outline_to_mask",
    "overlay_diff",
    "parse_dxf_drawing",
    "record_feedback",
    "render_dxf_preview",
    "render_dxf_to_stencil",
    "render_page_preview",
    "render_pdf_page",
    "search",
    "search_by_code",
    "search_by_png_stencil",
    "search_by_reference_dxf",
    "segment_designs",
    "segment_dxf_drawing",
    "read_dxf",
    "warm_up",
]

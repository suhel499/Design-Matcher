"""
Data models and dataclasses for the matching engine.
UI-independent, lightweight, and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DesignRecord:
    """Represents an indexed catalog design item."""

    id: int
    pdf_id: Optional[int]
    pdf_path: str
    folder: str
    page_no: Optional[int]
    x: int
    y: int
    w: int
    h: int
    thumb_png: bytes
    mask_hash: str
    aspect_ratio: float = 1.0
    fill_ratio: float = 0.5
    canonical_mask_png: Optional[bytes] = None
    source_type: str = "pdf"              # "pdf" or "dxf"
    dxf_id: Optional[int] = None
    label: Optional[str] = None           # e.g. "PD 010"
    design_code: Optional[str] = None     # Alias for label
    bbox_dxf: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h) in drawing units
    same_design_group: Optional[int] = None
    polarity_variant: int = 0             # 0 = normal/standard, 1 = inverted
    entity_count: int = 0
    design_index_info: Optional[str] = None

    def __post_init__(self):
        if self.design_code and not self.label:
            self.label = self.design_code
        elif self.label and not self.design_code:
            self.design_code = self.label


@dataclass
class MatchResult:
    """Represents a candidate design matched against a query photo."""

    folder: str
    pdf_path: str                             # File path to matched PDF or DXF
    page_no: Optional[int]                    # Page number (None for DXF)
    bbox: Tuple[int, int, int, int]           # (x, y, w, h) in rendered coordinates
    score: float                              # 0.0 to 100.0
    rotation: int                             # 0, 90, 180, 270 degrees
    mirrored: bool                            # True if horizontally flipped query matched best
    partial: bool                             # True if template-matched as repeating sub-pattern
    thumb_png: bytes                          # PNG encoded thumbnail bytes
    design_id: int = 0
    file_type: str = "pdf"                    # "pdf" or "dxf"
    source_type: str = "pdf"                  # "pdf" or "dxf"
    dxf_path: Optional[str] = None            # File path if source_type == "dxf"
    label: Optional[str] = None               # e.g. "PD 010"
    design_code: Optional[str] = None         # Alias for label
    bbox_dxf: Optional[Tuple[float, float, float, float]] = None
    polarity_variant: int = 0
    entity_count: int = 0
    design_index_info: Optional[str] = None   # e.g. "design 3 of 12"
    related: List[MatchResult] = field(default_factory=list)

    def __post_init__(self):
        if self.design_code and not self.label:
            self.label = self.design_code
        elif self.label and not self.design_code:
            self.design_code = self.label
        if self.source_type == "dxf" and not self.dxf_path:
            self.dxf_path = self.pdf_path
        if self.dxf_path and not self.source_type:
            self.source_type = "dxf"
        if self.source_type == "dxf":
            self.file_type = "dxf"


@dataclass
class SearchResponse:
    """Encapsulates the response from a photo search query, returning both DXF and PDF matches."""

    results: List[MatchResult]
    dxf_results: List[MatchResult] = field(default_factory=list)
    pdf_results: List[MatchResult] = field(default_factory=list)
    no_confident_match: bool = False
    reason: str = ""

    def __post_init__(self):
        if not self.dxf_results:
            self.dxf_results = [r for r in self.results if r.file_type == "dxf" or r.source_type == "dxf"]
        if not self.pdf_results:
            self.pdf_results = [r for r in self.results if r.file_type != "dxf" and r.source_type != "dxf"]

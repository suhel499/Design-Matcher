# Laser Cut Design Matcher

A high-performance Python 3.11 Windows desktop application and standalone matching engine for CNC/laser/router businesses. It matches customer photos of fabricated metal cut-out panels (jali/stencils) against large catalog PDF libraries across multiple directories.

---

## Architecture & Module Layout

The matching engine is strictly **UI-independent** (no PySide6 imports in the core engine) so it can be called safely from background worker threads and standalone scripts:

```text
design_matcher/
├── engine/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Global constants & tunable thresholds
│   ├── models.py             # Dataclasses (DesignRecord, MatchResult, SearchResponse)
│   ├── pdf_render.py         # pypdfium2 PDF page renderer (Apache-2.0 / BSD-3)
│   ├── segment.py            # Morphological segmentation of catalog pages
│   ├── features.py           # Canonical 256x256 mask prep & L2 embedders
│   ├── index_store.py        # SQLite metadata & FAISS vector persistence
│   ├── indexer.py            # Incremental folder indexing with ProcessPoolExecutor
│   ├── matcher.py            # 8-variant search + IoU/Chamfer/template re-ranker
│   ├── utils.py              # Visual diffs, preview rendering, and system open
│   └── ui_workers.py         # PySide6 SearchWorker & IndexWorker (QThread)
├── evaluate.py               # Accuracy benchmark script for customer photos
├── preprocess.py             # Stencil normalization (CLAHE, Otsu, polarity)
├── image_crop.py             # Lossless crop rect & sub-pixel 4-corner warp
├── requirements.txt          # Pinned dependencies
├── build.spec                # PyInstaller packaging configuration
└── tests/
    ├── test_matching_engine.py  # 20 synthetic patterns, polarity, incremental tests
    ├── test_preprocess.py       # Stencil normalization & polarity tests
    ├── test_image_crop.py       # Zero-loss rectangle crop & warp tests
    ├── test_tiles.py            # Tile segmentation tests
    └── test_gui.py              # PySide6 desktop GUI tests
```

---

## 1. Quality & Design Principles

1. **Strict Stencil Convention**:
   - `White (255)` = Solid metal panel material
   - `Black (0)` = Cut-outs, apertures, voids
2. **Polarity Invariance**:
   - Both light panels on dark backdrops and dark panels on light backdrops produce the exact same standardized canonical mask.
   - Search evaluates all 8 geometric orientations (0°, 90°, 180°, 270° x normal/mirrored) across both standard and inverted polarities to handle ambiguous lighting.
3. **Lossless Cropping**:
   - Cropping slices the original full-resolution BGR array directly without lossy JPEG compression or pixmap resampling.
   - Draggable pins use sub-pixel float32 `INTER_CUBIC` perspective warp.
4. **Sub-2-Second Search**:
   - Candidate canonical masks (256x256 PNG) are cached in SQLite for instantaneous mask re-ranking without re-rendering PDFs from disk during search.

---

## 2. Configuration & What to Tune First

All constants are defined in `engine/config.py` and can be customized or overridden via environment variables:

### Key Parameters to Tune:
1. `MIN_CONFIDENCE` (Default: `40.0`):
   - Scores below this threshold trigger `no_confident_match = True`.
   - **Tune lower** (e.g. `30.0` - `35.0`) if customer photos are taken from extreme angles or with heavy motion blur.
   - **Tune higher** (e.g. `45.0` - `50.0`) to minimize false positives in very large catalogs.
2. **Re-Ranking Score Weights**:
   - `WEIGHT_EMBEDDING = 0.45`: Global shape & topological descriptor similarity.
   - `WEIGHT_IOU = 0.35`: Exact pixel-wise intersection over union of cutout holes.
   - `WEIGHT_CHAMFER = 0.20`: Symmetric edge distance between query and catalog cutouts.
   - `WEIGHT_ASPECT_PENALTY = 0.15`: Penalizes mismatched aspect ratios (e.g. square vs tall runner).
   - *Tuning tip*: If candidates are retrieved in Top-5 but ranked incorrectly at Rank 1, increase `WEIGHT_IOU` and `WEIGHT_CHAMFER`.
3. `RENDER_DPI` (Default: `200`):
   - Balance between rendering speed and fine cut-out resolution. 200 DPI offers optimal detail for fine laser slits while rendering in ~30 ms/page.
4. `TOP_K` (Default: `10`) & `RERANK_N` (Default: `20`):
   - Number of vector neighbors to pull from FAISS for fine mask alignment.

---

## 3. Running Accuracy Evaluation (`evaluate.py`)

To benchmark your catalog on real customer photos:

1. Place test photos in a directory named as `<expected_target>__anything.jpg` (where `<expected_target>` is the design ID, PDF filename, or design number).
2. Run the evaluation CLI:

```bash
python evaluate.py --photos /path/to/customer_photos --db catalogs.db --index vectors.index
```

The script prints:
- Top-1 and Top-5 accuracy percentages
- Average latency per query (ms)
- Detailed failure analysis with recommendations on weight adjustments

---

## 4. PyInstaller Packaging for Windows

The application packages into a standalone Windows executable (`.exe`) without needing external Python installations.

### Build Command:
```bash
pyinstaller build.spec --clean
```

### Critical PyInstaller Considerations:
1. **Shapely & GEOS DLLs**:
   - Shapely relies on native GEOS C libraries (`geos_c.dll` on Windows, `libgeos_c.dylib` on macOS, `libgeos_c.so` on Linux).
   - In `build.spec`, collect dynamic libraries and data files using `collect_dynamic_libs('shapely')` and `collect_data_files('shapely')`.
   - Include hidden imports: `['shapely', 'shapely.ops', 'shapely.geometry', 'shapely.strtree']`.
2. **ezdxf Hidden Imports & Data Files**:
   - Collect package data with `collect_data_files('ezdxf')`.
   - Include hidden imports: `['ezdxf', 'ezdxf.recover', 'ezdxf.path', 'ezdxf.bbox', 'ezdxf.math']`.
3. **FAISS Hidden Imports**:
   - Add to `hiddenimports`: `['faiss', 'faiss.swigfaiss']`.
4. **pypdfium2 Binaries**:
   - `pypdfium2` bundles native shared libraries (`pdfium.dll`). PyInstaller collects them via `collect_data_files('pypdfium2')`.
5. **Multiprocessing Support**:
   - Ensure `multiprocessing.freeze_support()` is called before spawning worker processes in Windows frozen executables.

---

## 5. DXF CAD Vector Ingestion Pipeline

Alongside PDF catalogs, the engine directly ingests CAD line drawing files (`*.dxf`) without converting them through raster PDF steps:

```text
CAD DXF File (.dxf)
       │
       ▼
dxf_reader.py ────► ezdxf.recover, block expansion (depth 8), layer filtering
       │
       ▼
dxf_segment.py ───► Border panel detection, multi-design splitting, label pairing
       │
       ▼
linework_to_mask.py ► Shapely endpoint snapping (tol 0.1%), polygonize, even-odd XOR
       │
       ▼
canonical_mask() ──► 256x256 uint8 stencil mask (255 = metal, 0 = cutouts, True Y-axis)
       │
       ▼
FAISS + SQLite ────► Incremental index, dedup group linking (same_design_group)
```

### DXF Configuration Values to Tune First:
All values can be customized in `engine/config.py`:
1. **`DXF_GAP_TOLERANCE_RATIO`** (Default: `0.001` / `0.1%` of drawing diagonal):
   - Tolerance used by `shapely.STRtree` to snap open polyline and line segment endpoints before node union.
   - **Tune higher** (e.g. `0.002` - `0.003`) if customer CAD files have messy line endpoints or gaps between arcs.
   - **Tune lower** (e.g. `0.0005`) for high-precision micro-cut CAD drawings.
2. **`DXF_IGNORE_LAYERS`**:
   - Case-insensitive substring list of helper/construction layers to skip: `["dim", "defpoints", "text", "grid", "axis", "construction", "hatch_fill", "viewport"]`.
   - If ignore rules would filter out all geometry, `dxf_reader` automatically falls back to filtering only OFF, FROZEN, dimension, and text layers.
3. **`MASK_RENDER_SIZE`** (Default: `1600` px):
   - Rasterization canvas dimension for DXF linework.
   - High resolution (1600 px) ensures fine laser slits and small cutouts are accurately preserved without antialiasing artifacts before downsampling into the canonical 256x256 mask.
4. **`DXF_FLATTEN_DIST_RATIO`** (Default: `0.0005` / `0.05%` of bbox diagonal):
   - Flattening chord distance for arcs, ellipses, and splines. Provides smooth curve discretization at any unit scale (mm, inches, or unitless).
5. **`DXF_OPEN_STROKE_RATIO`** (Default: `0.003` / `0.3%` of long side, min 2 px):
   - Stroke thickness for unclosed open lines/slits, rendering them as thin cut-out voids.

---

## 6. DXF Ingestion Module & Debug CLI (`dxf_ingest.py`)

The standalone ingestion module `dxf_ingest.py` plugs directly into the existing matcher pipeline:

1. **Auto-Recovery & R12-2018 Support**:
   - `load_dxf(path)` reads CAD files using `ezdxf.readfile` with automatic fallback to `ezdxf.recover.readfile` for corrupted or R12 files.
   - Graceful rejection of `.dwg` files with clear user notification to export to DXF.
2. **Multi-Panel Extraction & Block Explosion**:
   - `extract_designs(doc)` resolves nested `INSERT` blocks via virtual entities (`entity.virtual_entities()`).
   - Ignores helper layers (`dim`, `text`, `construction`, `defpoints`, `hatch-fill`, `frame-only`) and entity types `TEXT`/`MTEXT`/`DIMENSION` for geometry.
   - Clusters nearby entities into individual design panels, using the outer closed rectangle as the panel border.
   - Automatically associates the nearest `TEXT`/`MTEXT` label below each panel (e.g., "PD 010") as the design code.
3. **High-Performance Rasterization & Gap Closure**:
   - `render_design(design, px_height=1024)` flattens curves (arcs, ellipses, splines) into crisp grayscale line art.
   - `convert_line_art_to_stencil_mask(rendered)` applies morphological closing to seal stencil bridges and flood-fills to identify cut-out voids (black 0) vs solid metal (white 255).
   - Generates both standard and inverted polarity versions for dual-polarity FAISS indexing.
4. **Debug CLI**:
   Run the debug CLI from the repository root to inspect detected panels, labels, and masks:
   ```bash
   python dxf_ingest.py path/to/drawing.dxf --out debug/
   ```
   Outputs:
   - `overlay_detected_panels.png`: Full drawing with colored bounding boxes and labels.
   - `panel_X_rendered.png`: Rendered line art for each panel.
   - `panel_X_mask_normal.png` and `panel_X_mask_inverted.png`: Dual polarity stencil masks.
5. **Text Shortcut Search**:
   - Search box in the header allows typing design codes (e.g., "PD 010") to instantly retrieve exact matches case- and space-insensitively without image search.
6. **UI Integration**:
   - Result cards show DXF format badges, design code, and index info (e.g., "Design: PD 010 | design 3 of 12").
   - Action buttons: "Open DXF", "Open DXF Folder", "Stencil Mask Preview" (comparing query mask, matched mask, and CAD preview), and "Copy Path".
   - Settings dialog includes DXF directory picker and include/exclude DXF search toggle.

---

## 7. Running the Tests

To run the complete test suite:

```bash
pytest -v
```

All 63 unit and integration tests validate:
- Stencil normalization, coordinate transformations, 4-corner perspective warp.
- 20-pattern recognition under rotation, mirroring, blur, noise, lighting, and perspective warp.
- DXF reading with recovery, nested block expansion up to depth 8, and layer filtering.
- Multi-panel DXF extraction, block explosion, stencil gap closing, and arc/spline curve flattening.
- Debug CLI execution and artifact generation.
- End-to-end photo perturbation test (rotation, perspective skew, blur, noise -> top-3 match).
- Incremental indexing for both DXF and PDF files, skipping unchanged files and removing deleted ones.
- Graceful handling of unsupported `.dwg` files and corrupted DXF files without scan crashes.
- In-app vector DXF viewer scene building, bounds verification, layer visibility toggling, upright Y-flip view transforms, ACI 7 contrast adaptation, measure tool accuracy, and LRU scene caching.

---

## 8. In-App Vector DXF Viewer (`dxf_viewer.py`)

A hardware-accelerated, crisp vector graphics viewer integrated directly into Design Matcher's results panel and standalone windows:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ [Toolbar] 🔍 Zoom In | 🔍 Zoom Out | ⛶ Fit All | 🎯 Zoom to Match     │
│           📏 Measure | 🌙 Dark/Light | 🖼 Query Mask | 🚀 Open in App  │
├───────────────┬────────────────────────────────────────┬───────────────┤
│ 📐 DESIGNS    │              VECTOR CANVAS             │ 🖼 QUERY MASK │
│ ★ PD 010      │  ┌──────────────────────────────────┐  │ (Side-by-side │
│   (1200×2400) │  │       Matched Design Frame       │  │  Photo Stencil│
│ ★ PD 011      │  │        [★ PD 010 Badge]          │  │  Mask)        │
│               │  └──────────────────────────────────┘  │               │
│ 📑 LAYERS     │                                        │               │
│ ☑ CUTTING     │                                        │               │
│ ☑ ENGRAVING   │                                        │               │
│ ☑ BORDER      │                                        │               │
├───────────────┴────────────────────────────────────────┴───────────────┤
│ [Status Bar] X: 125.40 mm  Y: 340.20 mm | Zoom: 145% | 1,240 entities  │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Viewer Features:
1. **Crisp Vector Graphics (`QGraphicsView` / `QGraphicsScene`)**:
   - Entities converted to vector `QPainterPath` items: LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE, block INSERTs (expanded with nested scale/rotation via `virtual_entities()`), and TEXT/MTEXT labels.
   - Cosmetic pens (`setCosmetic(True)`) ensure consistent, crisp hairline rendering under infinite zoom without line thickening.
   - Upright CAD coordinates: view transformed with `scale(1.0, -1.0)` so $+Y$ points up on screen, with text items counter-transformed to stay upright.
2. **Navigation & Controls**:
   - Mouse wheel zooms smoothly anchored under the cursor.
   - Middle-mouse or left-mouse dragging pans the canvas.
   - "Fit All" button (or double-clicking empty canvas, or `0` key) fits the whole drawing.
   - "Zoom to Match" frames the matched design bounding box (`dxf_bbox`) with comfortable padding.
   - Keyboard navigation: `+` / `-` / `0` for zoom and fit; arrow keys to pan.
3. **Design Highlight & Interactive Selection**:
   - Matched design bounding box rendered with a thick orange highlight rectangle and design code badge (`★ PD 010`).
   - Clicking any design panel in the drawing selects it and shows its code and dimensions.
   - "Detected Designs" side list allows clicking any design to zoom straight to it.
4. **Collapsible CAD Layer Panel**:
   - Checkboxes to toggle visibility of individual layers.
   - Colors rendered using native AutoCAD Color Index (ACI) and True Color attributes.
   - "🌙 Dark" / "☀️ Light" toggle automatically adapts ACI 7 (white/black) lines for high-contrast visibility.
5. **Precision Measure Tool ("📏 Measure")**:
   - Click two points in the drawing to measure Euclidean distance, $\Delta X$, and $\Delta Y$ in drawing units ($INSUNITS: mm, in, or unitless).
6. **Side-by-Side Comparison Panel ("🖼 Query Mask")**:
   - Toggles a side comparison panel displaying the customer's photo stencil mask side-by-side with the CAD vector drawing for 1:1 visual verification.
7. **"Open in Application" Launcher**:
   - Opens the DXF file using Windows default associated program (`os.startfile`) or cross-platform fallback (`QDesktopServices.openUrl`).
   - "Choose program..." dropdown menu lets users pick a specific executable (e.g. AutoCAD, CypCut, LaserCAD), remembered across sessions in `QSettings`.
   - "Show in Folder" (`explorer /select`) and "Copy Path" buttons.
8. **Performance & Robustness**:
   - Background thread loader (`QThread` / `DxfLoaderWorker`) with loading overlay ensures UI never freezes.
   - LRU scene cache (capacity = 5) makes re-clicking designs within the same DXF instantaneous (0 ms).
   - High-entity simplification mode (>200k entities) batches paths into unified layer items.
   - Automatic fallback to `ezdxf.recover.readfile` for corrupted files with technical details expander.

---

## 9. Manual Testing Checklist

| # | Test Action | Expected Result |
|---|-------------|-----------------|
| 1 | Single-click a DXF result card in results | Embedded viewer immediately displays DXF vector preview in right pane, auto-zoomed to matched design with orange highlight frame. |
| 2 | Double-click a DXF card or click "Open Viewer" | Standalone enlarged `DxfViewerWindow` opens with full toolbars, restoring previous window size and position. |
| 3 | Single-click a PDF result card | Preview pane switches to PDF page preview; existing PDF workflow behaves as before. |
| 4 | Mouse wheel zoom & drag pan | Viewport zooms smoothly centered on cursor; drag pans without lag. |
| 5 | Double-click empty canvas or click "Fit All" | Drawing fits completely into view preserving aspect ratio and $+Y$ upright orientation. |
| 6 | Click "Zoom to Match" | Viewport smoothly centers and frames the matched design bounding box with padding. |
| 7 | Toggle layer checkboxes in side panel | Associated layer linework hides and shows instantly without redrawing scene. |
| 8 | Click "🌙 Dark" / "☀️ Light" toggle | Background toggles between dark slate and light canvas; ACI 7 lines adapt automatically for contrast. |
| 9 | Activate "📏 Measure" tool and click 2 points | Rubber-band cyan measurement line draws with distance readout ($\Delta X$, $\Delta Y$) in drawing units. |
| 10 | Click a panel in "Detected Designs" side list | Viewport zooms directly to that panel and displays its design code badge. |
| 11 | Click "Open in Application" | File opens in the default CAD program or user-configured executable without crashing. |
| 12 | Select "Show in Folder" | Windows Explorer opens with the target DXF file selected. |
| 13 | Load a missing or corrupt DXF file | Shows friendly error notification without crashing or freezing the UI. |

---

## 10. Production DXF Design Matching Engine

The application includes a local, production-ready vector matching engine for searching libraries of 10,000–100,000 `.dxf` CAD files without cloud dependencies.

### Dual Search Modes:
1. **Mode A: Photo / PNG Stencil → DXF Search**:
   - Accepts customer photos or pre-binarized stencil masks.
   - Extracts canonical vector contours and 128-dimensional topological descriptors.
   - Searches the indexed DXF library and retrieves matching designs ranked by verified similarity.
2. **Mode B: Reference DXF → DXF Search**:
   - Accepts a reference `.dxf` file.
   - Handles multi-panel sheets by letting the user choose which design panel to query.
   - Matches designs invariant to scale (0.1x–10x), translation, rotation (0°, 90°, 180°, 270° + fine angles), mirroring (horizontal reflection), and differing polyline curve segmentations.
   - Evaluates bidirectional Chamfer distance via `scipy.spatial.cKDTree` and raster mask IoU.

### Multi-Stage Matching Architecture:
- **Stage 1 (Candidate Retrieval)**: Fast cosine similarity over 128-d in-memory vector cache (< 3 ms for 100,000 items in < 50 MB RAM). Retrieves top candidate design components.
- **Stage 2 (Geometric Verification)**: Detailed rotation-aware alignment across 8 orientations + local angular refinement (±3°, ±6°), bidirectional Chamfer distance calculation via `cKDTree`, and optional raster mask IoU.
- **Stage 3 (Calibrated Ranking)**: Results categorized into calibrated match statuses:
  - `Verified Match` (Confidence ≥ 70%, Chamfer distance ≤ 0.12)
  - `Possible Match` (Confidence 40%–69%)
  - `No Reliable Match` (Confidence < 40%)

### SQLite Database Schema (`dxf_library.db`):
- `design_files`: Stores unique file ID, path, filename, size, mtime, SHA-256 content hash, indexing status.
- `design_components`: Extracted design panels/sub-components with bounding box, entity count, and layer info.
- `design_features`: 128-d normalized feature vectors (schema v1) stored as 32-bit float BLOBs.
- `design_previews`: High-resolution rendered preview PNG and canonical binary stencil mask PNG.
- `index_jobs`: Tracks background batch indexing runs (start time, progress, files scanned/indexed/skipped/failed).
- `index_errors`: Logs unsupported entities, corrupted files, and parsing exceptions.
- `search_history`: Records search queries, latencies, and best matching results.

### Command-Line Interface (CLI):
```bash
# 1. Index a DXF library incrementally
python -m app.cli index-dxf /path/to/dxf_library

# 2. Rebuild the index from scratch
python -m app.cli index-dxf /path/to/dxf_library --rebuild

# 3. Match using a reference DXF file (Mode B)
python -m app.cli match-dxf /path/to/sample.dxf --top-k 10 --min-conf 30.0

# 4. View system status (both PDF catalog and DXF library)
python -m app.cli status
```

### Running the Evaluation Benchmark:
```bash
python benchmark_dxf_matcher.py
```
This script tests:
- Initial indexing throughput (files/sec)
- Incremental indexing efficiency (mtime/size cached skips)
- Mode B geometric invariance accuracy (Exact, Rotated, Scaled, Mirrored, Modified)
- Mode A stencil accuracy
- 10k, 50k, 100k vector search latency and memory footprint



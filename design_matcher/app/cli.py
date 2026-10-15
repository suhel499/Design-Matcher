"""
Command-line interface for Design Matcher.
Enables indexing, searching, status reporting, and model caching from the terminal.

Usage:
    python -m app.cli index /path/to/catalogs [--dpi 200] [--force]
    python -m app.cli search /path/to/customer_photo.jpg [--top-k 5]
    python -m app.cli status
    python -m app.cli download-model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so app modules can always be imported
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from app.core.embed import DEFAULT_MODEL_NAME, DINOv2Embedder
from app.core.index import CatalogIndex, IndexProgress, get_default_data_dir
from app.core.licensing.manager import LicenseManager
from app.core.search import DesignSearcher


def setup_cli_logging(verbose: bool = False) -> None:
    """Configures console and file logging."""
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=log_format, datefmt="%H:%M:%S")


def cmd_index(args: argparse.Namespace) -> int:
    """Handles the 'index' command."""
    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_default_data_dir()
    lic_mgr = LicenseManager(data_dir=data_dir)
    allowed, lic_msg = lic_mgr.is_operation_allowed("index_pdf")
    if not allowed:
        print(f"\n[LICENSE ERROR] Operation denied: {lic_msg}", file=sys.stderr)
        print("Please activate your subscription or renew your license to continue.\n", file=sys.stderr)
        return 1
    print(f"\n=======================================================")
    print(f" Design Matcher - Catalog Indexing")
    print(f"=======================================================")
    print(f" Catalog Root: {root_dir}")
    print(f" Data Directory: {data_dir}")
    print(f" Rendering DPI: {args.dpi}")
    print(f" Sub-regions: {'Enabled' if not args.no_subregions else 'Disabled'}")
    print(f" Force Reindex: {'Yes' if args.force else 'No'}")
    print(f"=======================================================\n")

    catalog_index = CatalogIndex(data_dir=data_dir)

    def on_progress(p: IndexProgress) -> None:
        if p.phase == "scanning":
            print(f"[*] {p.message}")
        elif p.phase == "indexing":
            page_info = f"page {p.current_page}/{p.total_pages_in_file}" if p.total_pages_in_file else ""
            print(
                f"\r[{p.percent:5.1f}%] File {p.current_file_idx}/{p.total_files}: {p.current_pdf_path} {page_info} (tiles: {p.total_tiles_indexed})",
                end="",
                flush=True,
            )
        elif p.phase == "saving":
            print(f"\n[*] {p.message}")
        elif p.phase == "complete":
            print(f"\n[+] {p.message}")
        elif p.phase == "cancelled":
            print(f"\n[!] {p.message}")

    try:
        new_tiles = catalog_index.index_catalog(
            root_dir=root_dir,
            dpi=args.dpi,
            include_subregions=not args.no_subregions,
            run_ocr=not args.no_ocr,
            progress_cb=on_progress,
            force_reindex=args.force,
        )
        print(f"\nSummary:")
        print(f" - Newly indexed tile vectors: {new_tiles}")
        print(f" - Total vectors in FAISS index: {catalog_index.total_vectors}")
        print(f" - Total files in SQLite DB: {catalog_index.db.get_total_file_count()}")
        return 0
    except Exception as exc:
        logging.exception("Indexing failed: %s", exc)
        print(f"\n[ERROR] Indexing failed: {exc}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Handles the 'search' command."""
    photo_path = Path(args.photo).resolve()
    if not photo_path.is_file():
        print(f"Error: Photo file does not exist: {photo_path}", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_default_data_dir()
    lic_mgr = LicenseManager(data_dir=data_dir)
    allowed, lic_msg = lic_mgr.is_operation_allowed("search_pdf")
    if not allowed:
        print(f"\n[LICENSE ERROR] Operation denied: {lic_msg}", file=sys.stderr)
        print("Please activate your subscription or renew your license to continue.\n", file=sys.stderr)
        return 1

    catalog_index = CatalogIndex(data_dir=data_dir)

    if catalog_index.total_vectors == 0:
        print("[!] The vector index is empty. Please run 'python -m app.cli index <root>' first.", file=sys.stderr)
        return 1

    searcher = DesignSearcher(catalog_index)
    print(f"\nSearching for closest matches for '{photo_path.name}'...")
    print(f"Total catalog vectors: {catalog_index.total_vectors}")

    try:
        results = searcher.search_photo_file(photo_path, top_k=args.top_k)
        if not results:
            print("No matching designs found.")
            return 0

        dxf_results = [
            r for r in results
            if r.pdf_path.lower().endswith(".dxf") or getattr(r, "file_type", "pdf") == "dxf"
        ]
        pdf_results = [
            r for r in results
            if not (r.pdf_path.lower().endswith(".dxf") or getattr(r, "file_type", "pdf") == "dxf")
        ]

        if dxf_results:
            print(f"\n📐 Top {len(dxf_results)} DXF CAD Matches:")
            print(f"{'Rank':<5} {'Score':<8} {'Conf%':<7} {'DXF CAD File':<30} {'Design#':<15} {'Folder'}")
            print("-" * 85)
            for res in dxf_results:
                des_num = res.design_number or "-"
                print(
                    f"{res.rank:<5} {res.score:<8.4f} {res.confidence_pct:<7.1f}% {res.pdf_name:<30} "
                    f"{des_num:<15} {res.folder}"
                )
            print("-" * 85)

        if pdf_results:
            print(f"\n📄 Top {len(pdf_results)} PDF Catalog Matches:")
            print(f"{'Rank':<5} {'Score':<8} {'Conf%':<7} {'PDF File':<25} {'Page':<6} {'Tile#':<6} {'Design#':<10} {'Folder'}")
            print("-" * 90)
            for res in pdf_results:
                des_num = res.design_number or "-"
                print(
                    f"{res.rank:<5} {res.score:<8.4f} {res.confidence_pct:<7.1f}% {res.pdf_name:<25} "
                    f"{res.page_number:<6} {res.tile_index:<6} {des_num:<10} {res.folder}"
                )
            print("-" * 90)
        return 0
    except Exception as exc:
        logging.exception("Search failed: %s", exc)
        print(f"[ERROR] Search failed: {exc}", file=sys.stderr)
        return 1


def cmd_index_dxf(args: argparse.Namespace) -> int:
    """Handles the 'index-dxf' command for dedicated DXF library indexing."""
    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        print(f"Error: Directory does not exist: {root_dir}", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_default_data_dir()
    dxf_db_path = data_dir / "dxf_library.db"
    from engine.dxf_database import DxfDatabase

    print(f"\n=======================================================")
    print(f" Design Matcher - DXF Library Indexing")
    print(f"=======================================================")
    print(f" DXF Root Folder:  {root_dir}")
    print(f" DXF Database:     {dxf_db_path}")
    print(f" Rebuild Index:    {'Yes' if args.rebuild else 'No (Incremental)'}")
    print(f"=======================================================\n")

    db = DxfDatabase(dxf_db_path)

    def on_progress(p):
        print(f"\r[{p.percent:5.1f}%] {p.message}", end="", flush=True)

    try:
        if args.rebuild:
            print("[*] Rebuilding index from scratch...")
            prog = db.rebuild_index(root_dir, progress_callback=on_progress)
        else:
            prog = db.scan_and_index_directory(root_dir, progress_callback=on_progress)

        if prog is None:
            prog = db.scan_and_index_directory(root_dir, progress_callback=on_progress)

        print(f"\n\n[+] Indexing Complete:")
        print(f" - Scanned files:   {prog.total}")
        print(f" - Newly indexed:   {prog.indexed}")
        print(f" - Skipped (cached): {prog.skipped}")
        print(f" - Failed/errors:   {prog.failed}")
        stats = db.get_stats()
        print(f" - Total in library: {stats['total_files']} files ({stats['total_components']} components)")
        return 0
    except Exception as exc:
        logging.exception("DXF indexing failed: %s", exc)
        print(f"\n[ERROR] DXF indexing failed: {exc}", file=sys.stderr)
        return 1


def cmd_match_dxf(args: argparse.Namespace) -> int:
    """Handles the 'match-dxf' command (Mode B: Reference DXF -> DXF search)."""
    ref_path = Path(args.reference_dxf).resolve()
    if not ref_path.is_file():
        print(f"Error: Reference DXF file does not exist: {ref_path}", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_default_data_dir()
    dxf_db_path = data_dir / "dxf_library.db"
    if not dxf_db_path.exists():
        print(f"[!] DXF Database not found at {dxf_db_path}. Please run 'index-dxf' first.", file=sys.stderr)
        return 1

    from engine.dxf_database import DxfDatabase
    from engine.dxf_matcher import DxfMatchingEngine

    db = DxfDatabase(dxf_db_path)
    engine = DxfMatchingEngine(db)

    print(f"\nSearching DXF library for matches to '{ref_path.name}'...")
    try:
        resp = engine.search_by_reference_dxf(
            ref_path,
            top_k=args.top_k,
            min_confidence=args.min_conf,
        )

        if not resp.matches:
            print("No matching DXF designs found above confidence threshold.")
            return 0

        print(f"\n📐 Top {len(resp.matches)} Verified DXF Matches ({resp.search_latency_ms:.1f} ms):")
        print(f"{'Rank':<5} {'Score':<8} {'Status':<16} {'Dist':<7} {'Rot':<5} {'DXF Filename':<30} {'Folder'}")
        print("-" * 95)
        for m in resp.matches:
            mirr_tag = " (M)" if m.is_mirrored else ""
            print(
                f"{m.rank:<5} {m.confidence_score:5.1f}%  {m.match_status:<16} {m.geometric_distance:<7.3f} "
                f"{str(m.rotation_deg) + '°' + mirr_tag:<5} {m.filename:<30} {Path(m.file_path).parent}"
            )
        print("-" * 95)
        return 0
    except Exception as exc:
        logging.exception("DXF search failed: %s", exc)
        print(f"[ERROR] DXF search failed: {exc}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Handles the 'status' command."""
    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_default_data_dir()
    catalog_index = CatalogIndex(data_dir=data_dir)

    print(f"\n=======================================================")
    print(f" Design Matcher - System Status")
    print(f"=======================================================")
    print(f" Storage Directory: {data_dir}")
    print(f" PDF/Image SQLite:  {catalog_index.db_path} (exists: {catalog_index.db_path.exists()})")
    print(f" PDF/Image FAISS:   {catalog_index.faiss_path} (exists: {catalog_index.faiss_path.exists()})")
    print(f" Total PDF Files:   {catalog_index.db.get_total_file_count()}")
    print(f" Total Tile Meta:   {catalog_index.db.get_total_tile_count()}")
    print(f" Total FAISS Vecs:  {catalog_index.total_vectors}")

    dxf_db_path = data_dir / "dxf_library.db"
    if dxf_db_path.exists():
        from engine.dxf_database import DxfDatabase
        dxf_db = DxfDatabase(dxf_db_path)
        stats = dxf_db.get_stats()
        print(f" DXF Library DB:    {dxf_db_path} (exists: True)")
        print(f" Total DXF Files:   {stats['total_files']}")
        print(f" Total DXF Comps:   {stats['total_components']}")
    else:
        print(f" DXF Library DB:    {dxf_db_path} (not created yet)")
    print(f"=======================================================")

    files = catalog_index.db.get_all_indexed_files()
    if files:
        print(f"\nIndexed PDF/Catalog Files ({len(files)}):")
        print(f"{'File Name':<30} {'Pages':<7} {'Tiles':<7} {'Folder':<20} {'Indexed At'}")
        print("-" * 85)
        for f in files:
            fname = Path(f["pdf_path"]).name
            print(f"{fname:<30} {f['page_count']:<7} {f['tile_count']:<7} {f['folder']:<20} {f['indexed_at']}")
        print("-" * 85)
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    """Pre-downloads DINOv2 model weights for offline operation."""
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    print(f"Pre-downloading '{DEFAULT_MODEL_NAME}' to local cache...")
    embedder = DINOv2Embedder(cache_dir=cache_dir)
    embedder._load_model()
    print("Model successfully loaded and cached!")
    return 0


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="design_matcher",
        description="Design Matcher: Cut-design catalog indexing and search tool.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute")

    # index command
    parser_index = subparsers.add_parser("index", help="Recursively index PDF & DXF catalogs in a directory.")
    parser_index.add_argument("root", help="Root folder containing PDF & DXF catalogs.")
    parser_index.add_argument("--dpi", type=int, default=200, help="Rendering DPI (default 200).")
    parser_index.add_argument("--force", action="store_true", help="Force reindexing even if file hash is unchanged.")
    parser_index.add_argument("--no-subregions", action="store_true", help="Skip generating overlapping sub-regions.")
    parser_index.add_argument("--no-ocr", action="store_true", help="Disable OCR for design numbers.")
    parser_index.add_argument("--data-dir", default=None, help="Custom data directory for SQLite and FAISS.")

    # search command
    parser_search = subparsers.add_parser("search", help="Search catalog for closest design matching a photo.")
    parser_search.add_argument("photo", help="Path to customer photo (JPG, PNG, WEBP, BMP).")
    parser_search.add_argument("--top-k", type=int, default=5, help="Number of results to return (default 5).")
    parser_search.add_argument("--data-dir", default=None, help="Custom data directory for SQLite and FAISS.")

    # status command
    parser_status = subparsers.add_parser("status", help="Show current catalog index statistics.")
    parser_status.add_argument("--data-dir", default=None, help="Custom data directory for SQLite and FAISS.")

    # index-dxf command
    parser_index_dxf = subparsers.add_parser("index-dxf", help="Scan and index a dedicated DXF library folder.")
    parser_index_dxf.add_argument("root", help="Root folder containing .dxf files.")
    parser_index_dxf.add_argument("--rebuild", action="store_true", help="Rebuild entire DXF index from scratch.")
    parser_index_dxf.add_argument("--data-dir", default=None, help="Custom data directory for SQLite.")

    # match-dxf command
    parser_match_dxf = subparsers.add_parser("match-dxf", help="Search DXF library using a reference .dxf file.")
    parser_match_dxf.add_argument("reference_dxf", help="Path to reference .dxf query file.")
    parser_match_dxf.add_argument("--top-k", type=int, default=10, help="Number of results (default 10).")
    parser_match_dxf.add_argument("--min-conf", type=float, default=30.0, help="Minimum confidence threshold (0-100).")
    parser_match_dxf.add_argument("--data-dir", default=None, help="Custom data directory for SQLite.")

    # download-model command
    parser_dl = subparsers.add_parser("download-model", help="Download and cache DINOv2 model for offline use.")
    parser_dl.add_argument("--cache-dir", default=None, help="Custom model cache directory.")

    args = parser.parse_args()
    setup_cli_logging(args.verbose)

    if args.command == "index":
        sys.exit(cmd_index(args))
    elif args.command == "index-dxf":
        sys.exit(cmd_index_dxf(args))
    elif args.command == "match-dxf":
        sys.exit(cmd_match_dxf(args))
    elif args.command == "search":
        sys.exit(cmd_search(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "download-model":
        sys.exit(cmd_download_model(args))


if __name__ == "__main__":
    main()

"""
Profiling script for Step 1: Profile first (don't guess).
Uses cProfile and pstats to profile the current indexing code path on the 1000-file mixed dataset.
Measures:
- Disk walk (scan_directory_scandir, os.walk in scan_cad_files)
- File open/parse (ezdxf, pypdfium2, dwg_reader)
- File hashing (compute_fast_hash SHA-256)
- Database writes (catalogue_files upsert, FTS5 sync)
- UI folder tree & unique display label generation
Reports the Top 10 bottlenecks with precise timings before changing any code.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add design_matcher to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "design_matcher"))

from app.core.db import CatalogDatabase
from app.core.file_scanner import build_unique_display_labels, scan_cad_files
from app.core.indexing_engine import TwoStageIndexingEngine, TwoStageProgress
from app.core.stage_a_extractor import (
    compute_fast_hash,
    extract_dxf_metadata,
    extract_metadata_batch,
    extract_pdf_metadata,
    extract_single_file_metadata,
)
from dxf_viewer import build_tree_hierarchy


class MockCatalogIndex:
    def __init__(self, db_path: Path, data_dir: Path):
        self.db = CatalogDatabase(db_path, use_mysql=False)
        self.data_dir = data_dir
        self.thumbnails_dir = data_dir / "thumbnails"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = None

    def remove_vectors(self, ids):
        pass

    def add_vectors(self, embeddings, ids):
        pass

    def save_index(self):
        pass


def run_comprehensive_profile(catalog_dir: Path):
    temp_dir = Path(tempfile.mkdtemp(prefix="profile_run_"))
    try:
        db_path = temp_dir / "catalog.db"
        catalog_index = MockCatalogIndex(db_path, temp_dir)

        print("=" * 80)
        print("STEP 1: PROFILING CURRENT INDEXING CODE PATH ON 1,000+ MIXED FILES")
        print("=" * 80)
        print(f"Dataset target: {catalog_dir}")

        # -------------------------------------------------------------
        # Part A: cProfile on TwoStageIndexingEngine (Metadata / Stage A)
        # -------------------------------------------------------------
        engine = TwoStageIndexingEngine(catalog_index, max_workers=1)  # Profile in-process to capture full calltree

        profiler = cProfile.Profile()
        start_time = time.perf_counter()

        profiler.enable()
        summary = engine.run_two_stage_index(catalog_dir, mode="metadata_only")
        profiler.disable()

        total_duration = time.perf_counter() - start_time
        total_files = summary.get("items_processed", 0)
        files_per_sec = total_files / total_duration if total_duration > 0 else 0

        print(f"\n[+] Indexing Complete: {total_files} files processed in {total_duration:.3f}s ({files_per_sec:.1f} files/sec)")

        # -------------------------------------------------------------
        # Part B: Top 10 Bottlenecks Analysis using pstats
        # -------------------------------------------------------------
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(25)
        raw_pstats_cum = s.getvalue()

        s_tot = io.StringIO()
        ps_tot = pstats.Stats(profiler, stream=s_tot).sort_stats("tottime")
        ps_tot.print_stats(25)
        raw_pstats_tot = s_tot.getvalue()

        # -------------------------------------------------------------
        # Part C: Component-by-Component Micro-benchmarks
        # -------------------------------------------------------------
        # 1. Disk walk / discovery
        t0 = time.perf_counter()
        scanned = engine.scan_directory_scandir(catalog_dir)
        disk_walk_time = time.perf_counter() - t0

        # 2. File hashing (SHA256 of all files)
        t0 = time.perf_counter()
        for p, _, _ in scanned:
            compute_fast_hash(p)
        hash_time = time.perf_counter() - t0

        # 3. DXF Parsing
        dxf_files = [p for p, _, _ in scanned if p.lower().endswith(".dxf")]
        t0 = time.perf_counter()
        dxf_errors = 0
        for f in dxf_files:
            try:
                extract_dxf_metadata(f)
            except Exception:
                dxf_errors += 1
        dxf_parse_time = time.perf_counter() - t0

        # 4. PDF Parsing
        pdf_files = [p for p, _, _ in scanned if p.lower().endswith(".pdf")]
        t0 = time.perf_counter()
        pdf_errors = 0
        for f in pdf_files:
            try:
                extract_pdf_metadata(f)
            except Exception:
                pdf_errors += 1
        pdf_parse_time = time.perf_counter() - t0

        # 5. Database Writes
        t0 = time.perf_counter()
        db_path_bench = temp_dir / "bench_write.db"
        bench_db = CatalogDatabase(db_path_bench, use_mysql=False)
        dummy_records = [
            {
                "path": p,
                "filename": os.path.basename(p),
                "extension": os.path.splitext(p)[1],
                "file_size": sz,
                "modified_ns": mod,
                "content_hash": "dummyhash",
                "page_count": 1,
                "searchable_metadata": f"{os.path.basename(p)} test keywords layer cut frame",
                "metadata_status": "completed",
                "visual_status": "pending",
                "parser_version": "1.0",
                "error_message": "",
            }
            for p, mod, sz in scanned
        ]
        bench_db.upsert_catalogue_files_batch(dummy_records)
        db_write_time = time.perf_counter() - t0

        # 6. UI Folder Tree and Display Label Generation
        t0 = time.perf_counter()
        all_paths = [Path(p) for p, _, _ in scanned]
        cad_paths = [p for p in all_paths if p.suffix.lower() in (".dxf", ".dwg")]
        labels = build_unique_display_labels(cad_paths, root=catalog_dir)
        tree = build_tree_hierarchy(cad_paths, root=catalog_dir)
        ui_labels_time = time.perf_counter() - t0

        # 7. scan_cad_files (os.walk legacy scanner in dxf_viewer)
        t0 = time.perf_counter()
        matched_walk = scan_cad_files(catalog_dir, allowed_extensions=[".dxf", ".dwg"], recursive=True)
        os_walk_time = time.perf_counter() - t0

        # Output detailed report
        print("\n" + "=" * 80)
        print("TIMING BREAKDOWN BY COMPONENT (Where the time goes):")
        print("=" * 80)
        print(f"1. Disk Walk (scan_directory_scandir):         {disk_walk_time*1000:8.2f} ms ({disk_walk_time/total_duration*100:5.1f}%)")
        print(f"2. File SHA-256 Hashing (compute_fast_hash):   {hash_time*1000:8.2f} ms ({hash_time/total_duration*100:5.1f}%)")
        print(f"3. DXF Full File Parsing (ezdxf.readfile):      {dxf_parse_time*1000:8.2f} ms ({dxf_parse_time/total_duration*100:5.1f}%) [500 files]")
        print(f"4. PDF Text Parsing (pypdfium2 doc/textpage):   {pdf_parse_time*1000:8.2f} ms ({pdf_parse_time/total_duration*100:5.1f}%) [200 files]")
        print(f"5. Database Writes & FTS5 Sync:                {db_write_time*1000:8.2f} ms ({db_write_time/total_duration*100:5.1f}%)")
        print(f"6. UI Label & Tree Hierarchy Generation:       {ui_labels_time*1000:8.2f} ms")
        print(f"7. Legacy os.walk Scanner (scan_cad_files):    {os_walk_time*1000:8.2f} ms")

        print("\n" + "=" * 80)
        print("RAW cProfile (Cumulative Time):")
        print("=" * 80)
        print(raw_pstats_cum)

        print("\n" + "=" * 80)
        print("RAW cProfile (Internal Self-Time):")
        print("=" * 80)
        print(raw_pstats_tot)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    benchmark_dir = Path("/Users/suhel/Documents/APP/javedApp/benchmark_data_1000")
    run_comprehensive_profile(benchmark_dir)

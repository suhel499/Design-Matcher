"""
Evaluation CLI script for customer photo benchmark datasets.
Parses query filenames formatted as:
    <design_id_or_expected_target>__<description>.jpg
Computes Top-1 and Top-5 accuracy, benchmark latency, and failure analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from engine.config import EngineConfig, config
from engine.features import get_default_embedder
from engine.index_store import IndexStore
from engine.matcher import Matcher


def parse_expected_target(filename: str) -> str:
    """Extracts expected target token before the double-underscore '__' separator."""
    stem = Path(filename).stem
    if "__" in stem:
        return stem.split("__", 1)[0].strip()
    return stem.strip()


def run_evaluation(
    photos_dir: Path,
    db_path: Path,
    index_path: Path,
    min_confidence: float = 40.0,
    top_k: int = 5,
) -> None:
    if not photos_dir.is_dir():
        print(f"Error: Photos directory not found: {photos_dir}")
        sys.exit(1)

    cfg = EngineConfig(DB_PATH=db_path, INDEX_PATH=index_path, MIN_CONFIDENCE=min_confidence)
    store = IndexStore(db_path=db_path, index_path=index_path)
    embedder = get_default_embedder(cfg.EMBEDDER_TYPE, cfg)
    matcher = Matcher(store=store, embedder=embedder, cfg=cfg)

    # Pre-warm FAISS index
    matcher.warm_up()

    photo_files = [
        p for p in sorted(photos_dir.glob("*"))
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ]

    if not photo_files:
        print(f"No image files found in {photos_dir}")
        return

    print("=" * 70)
    print(f"EVALUATING {len(photo_files)} QUERIES AGAINST CATALOG INDEX")
    print(f"DB: {db_path} | Index: {index_path} | Min Confidence: {min_confidence}")
    print("=" * 70)

    top1_correct = 0
    top5_correct = 0
    latencies = []
    failures = []

    for idx, photo_p in enumerate(photo_files, 1):
        expected_token = parse_expected_target(photo_p.name).lower()

        # Load image via np.fromfile for Unicode path robustness
        data = np.fromfile(str(photo_p), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{idx}/{len(photo_files)}] Could not read image: {photo_p.name}")
            failures.append((photo_p.name, expected_token, "READ_ERROR", 0.0))
            continue

        t0 = time.perf_counter()
        response = matcher.search(bgr, top_k=top_k)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed_ms)

        results = response.results
        is_top1 = False
        is_top5 = False
        predicted_tokens = []

        for rank, match in enumerate(results):
            # Check target match against design_id or pdf filename/stem
            cand_tokens = [
                str(match.design_id).lower(),
                Path(match.pdf_path).stem.lower(),
                f"{Path(match.pdf_path).stem.lower()}-p{match.page_no}",
            ]
            predicted_tokens.append((cand_tokens[0], match.score))

            if expected_token in cand_tokens or any(expected_token in tok for tok in cand_tokens):
                if rank == 0:
                    is_top1 = True
                is_top5 = True
                break

        if is_top1:
            top1_correct += 1
            top5_correct += 1
            print(f"[{idx}/{len(photo_files)}] PASS (Top-1) - {photo_p.name} ({t_elapsed_ms:.1f}ms)")
        elif is_top5:
            top5_correct += 1
            print(f"[{idx}/{len(photo_files)}] PASS (Top-5) - {photo_p.name} ({t_elapsed_ms:.1f}ms)")
        else:
            top_pred = predicted_tokens[0] if predicted_tokens else ("NONE", 0.0)
            failures.append((photo_p.name, expected_token, top_pred[0], top_pred[1]))
            print(f"[{idx}/{len(photo_files)}] FAIL - {photo_p.name} -> Top: {top_pred[0]} ({top_pred[1]}%) ({t_elapsed_ms:.1f}ms)")

    n_total = len(photo_files)
    top1_pct = (top1_correct / n_total) * 100.0
    top5_pct = (top5_correct / n_total) * 100.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Queries Evaluated: {n_total}")
    print(f"Top-1 Accuracy:          {top1_correct}/{n_total} ({top1_pct:.1f}%)")
    print(f"Top-5 Accuracy:          {top5_correct}/{n_total} ({top5_pct:.1f}%)")
    print(f"Average Search Latency:  {avg_latency:.1f} ms")
    print("=" * 70)

    if failures:
        print("\nFAILURE CASES:")
        for fname, exp, got, score in failures:
            print(f"  • {fname}: expected '{exp}', got '{got}' with score {score:.1f}")
        print("\nTuning Recommendations:")
        print("  - If Top-1 fails but Top-5 passes: increase WEIGHT_IOU or WEIGHT_CHAMFER in engine/config.py.")
        print("  - If designs are partial crops: decrease final_score threshold for partial template matching.")
        print("  - If false negatives occur on blurry customer photos: tune MIN_CONFIDENCE.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Design Matcher against test photos.")
    parser.add_argument("--photos", type=Path, required=True, help="Directory containing query photos.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="Path to catalogs.db SQLite database.")
    parser.add_argument("--index", type=Path, default=config.INDEX_PATH, help="Path to vectors.index FAISS file.")
    parser.add_argument("--min-confidence", type=float, default=40.0, help="Confidence threshold.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to consider.")
    args = parser.parse_args()

    run_evaluation(
        photos_dir=args.photos,
        db_path=args.db,
        index_path=args.index,
        min_confidence=args.min_confidence,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

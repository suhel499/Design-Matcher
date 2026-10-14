"""
Unit tests for the matching engine:
1. 20 synthetic cut-out patterns evaluated across perturbations (rotated, mirrored, noisy,
   blurred, differently lit, perspective-warped) with >=90% Top-1 and >=98% Top-5 accuracy.
2. Polarity invariance: light panel on dark backdrop vs dark panel on light backdrop.
3. Incremental indexing (skips unchanged, re-indexes changed, removes deleted) and cancellation.
4. Edge cases & error handling.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
from PIL import Image

from engine.config import EngineConfig
from engine.features import get_default_embedder
from engine.index_store import IndexStore
from engine.indexer import index_folders
from engine.matcher import Matcher


def generate_distinct_pattern(idx: int, size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a distinct synthetic laser cut-out pattern."""
    img = np.full((size, size, 3), (210, 210, 215), dtype=np.uint8)  # Silver metal
    mask = np.full((size, size), 255, dtype=np.uint8)  # 255 = metal

    if idx == 0:
        for r in range(50, size - 40, 60):
            for c in range(50, size - 40, 60):
                cv2.circle(mask, (c, r), 18, 0, -1)
    elif idx == 1:
        for rad in range(30, size // 2 - 20, 30):
            cv2.circle(mask, (size // 2, size // 2), rad, 0, 10)
    elif idx == 2:
        for r in range(50, size - 40, 70):
            for c in range(50, size - 40, 70):
                pts = np.array([[c, r - 24], [c + 24, r], [c, r + 24], [c - 24, r]], np.int32)
                cv2.fillPoly(mask, [pts], 0)
    elif idx == 3:
        for y in range(40, size - 40, 50):
            shift = 25 if (y // 50) % 2 == 1 else 0
            for x in range(40 + shift, size - 40, 50):
                cv2.circle(mask, (x, y), 16, 0, -1)
    elif idx == 4:
        for d in range(-size, size * 2, 40):
            cv2.line(mask, (d, 0), (d + size, size), 0, 12)
    elif idx == 5:
        for d in range(-size, size * 2, 40):
            cv2.line(mask, (d, size), (d + size, 0), 0, 12)
    elif idx == 6:
        for y in range(50, size - 40, 65):
            for x in range(50, size - 40, 65):
                cv2.line(mask, (x - 15, y), (x + 15, y), 0, 8)
                cv2.line(mask, (x, y - 15), (x, y + 15), 0, 8)
    elif idx == 7:
        for y in range(50, size - 40, 45):
            pts = [(x, int(y + 12 * np.sin(x / 25.0))) for x in range(30, size - 30, 4)]
            for i in range(len(pts) - 1):
                cv2.line(mask, pts[i], pts[i + 1], 0, 8)
    elif idx == 8:
        for x in range(50, size - 40, 45):
            cv2.line(mask, (x, 35), (x, size - 35), 0, 14)
    elif idx == 9:
        for y in range(50, size - 40, 45):
            cv2.line(mask, (35, y), (size - 35, y), 0, 14)
    elif idx == 10:
        cx, cy = size // 2, size // 2
        for ang in range(0, 360, 20):
            rad = np.radians(ang)
            x2 = int(cx + (size // 2 - 40) * np.cos(rad))
            y2 = int(cy + (size // 2 - 40) * np.sin(rad))
            cv2.line(mask, (cx, cy), (x2, y2), 0, 8)
    elif idx == 11:
        for r in range(40, size - 40, 60):
            for c in range(40, size - 40, 60):
                if ((r // 60) + (c // 60)) % 2 == 0:
                    cv2.rectangle(mask, (c, r), (c + 35, r + 35), 0, -1)
    elif idx == 12:
        for y in range(40, size - 40, 40):
            pts = np.array([[40, y], [size // 2, y + 20], [size - 40, y]], np.int32)
            cv2.polylines(mask, [pts], False, 0, 10)
    elif idx == 13:
        for y in range(50, size - 40, 60):
            for x in range(50, size - 40, 60):
                r = 8 + (x % 20)
                cv2.circle(mask, (x, y), r, 0, -1)
    elif idx == 14:
        for y in range(50, size - 40, 60):
            for x in range(50, size - 40, 60):
                pts = np.array([[x, y - 18], [x + 18, y + 18], [x - 18, y + 18]], np.int32)
                cv2.fillPoly(mask, [pts], 0)
    elif idx == 15:
        for y in range(50, size - 40, 50):
            cv2.line(mask, (40, y), (size - 40, y), 0, 6)
        for x in range(50, size - 40, 50):
            cv2.line(mask, (x, 40), (x, size - 40), 0, 6)
    elif idx == 16:
        for s in range(30, size // 2 - 20, 25):
            cv2.rectangle(mask, (size // 2 - s, size // 2 - s), (size // 2 + s, size // 2 + s), 0, 8)
    elif idx == 17:
        for y in range(50, size - 40, 65):
            for x in range(50, size - 40, 65):
                cv2.ellipse(mask, (x, y), (20, 10), 0, 0, 360, 0, -1)
    elif idx == 18:
        for y in range(50, size - 40, 60):
            for x in range(50, size - 40, 60):
                cv2.line(mask, (x - 15, y - 15), (x + 15, y + 15), 0, 7)
                cv2.line(mask, (x - 15, y + 15), (x + 15, y - 15), 0, 7)
    else:
        rng = np.random.RandomState(42 + idx)
        for _ in range(35):
            x, y = rng.randint(40, size - 40, size=2)
            cv2.circle(mask, (int(x), int(y)), 14, 0, -1)

    # Solid panel border frame
    cv2.rectangle(mask, (0, 0), (size - 1, size - 1), 255, 18)
    img[mask == 0] = (20, 20, 30)  # Cutouts backfilled with dark void
    return img, mask


def test_synthetic_20_patterns_accuracy(tmp_path: Path):
    """
    Renders 20 distinct patterns into PDFs, indexes them, and tests against:
    rotated, mirrored, noisy, blurred, differently lit, and perspective-warped queries.
    Asserts >= 90% Top-1 accuracy and >= 98% Top-5 accuracy.
    """
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()

    # Create 20 catalog PDFs
    for i in range(20):
        img, _ = generate_distinct_pattern(i)
        page = np.full((550, 550, 3), (30, 25, 20), dtype=np.uint8)
        page[75:475, 75:475] = img
        pil = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        pdf_path = cat_dir / f"design_{i:02d}.pdf"
        pil.save(str(pdf_path), "PDF", resolution=200.0)

    cfg = EngineConfig(
        DB_PATH=tmp_path / "test.db",
        INDEX_PATH=tmp_path / "test.index",
        MIN_CONFIDENCE=25.0,
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)

    stats = index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)
    assert stats.indexed_files == 20
    assert stats.indexed_designs == 20

    matcher = Matcher(store=store, embedder=embedder, cfg=cfg)

    top1_hits = 0
    top5_hits = 0

    for i in range(20):
        base_img, _ = generate_distinct_pattern(i)
        mode = i % 6
        if mode == 0:
            # Rotated 90 deg
            q_img = cv2.rotate(base_img, cv2.ROTATE_90_CLOCKWISE)
        elif mode == 1:
            # Mirrored
            q_img = cv2.flip(base_img, 1)
        elif mode == 2:
            # Blurred
            q_img = cv2.GaussianBlur(base_img, (7, 7), 1.8)
        elif mode == 3:
            # Noisy
            noise = np.random.normal(0, 15, base_img.shape).astype(np.int16)
            q_img = np.clip(base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode == 4:
            # Differently lit (illumination gradient across panel)
            h, w = base_img.shape[:2]
            grad = np.linspace(0.7, 1.3, w).reshape(1, w, 1)
            q_img = np.clip(base_img.astype(float) * grad, 0, 255).astype(np.uint8)
        else:
            # Perspective warped (skewed 4-corners)
            h, w = base_img.shape[:2]
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([[15, 10], [w - 10, 20], [w - 18, h - 15], [8, h - 10]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            q_img = cv2.warpPerspective(base_img, M, (w, h), borderValue=(20, 20, 30))

        resp = matcher.search(q_img, top_k=5, already_cropped=True)
        expected_pdf = f"design_{i:02d}.pdf"
        matched_pdfs = [Path(m.pdf_path).name for m in resp.results]

        if matched_pdfs and matched_pdfs[0] == expected_pdf:
            top1_hits += 1
        if expected_pdf in matched_pdfs:
            top5_hits += 1

    top1_acc = top1_hits / 20.0
    top5_acc = top5_hits / 20.0

    assert top1_acc >= 0.90, f"Top-1 accuracy {top1_acc:.2%} was below 90%"
    assert top5_acc >= 0.98, f"Top-5 accuracy {top5_acc:.2%} was below 98%"


def test_polarity_invariance(tmp_path: Path):
    """
    Asserts that a light panel on a dark backdrop and a dark panel on a light backdrop
    both successfully match the identical catalog design with high confidence.
    """
    cat_dir = tmp_path / "cat_pol"
    cat_dir.mkdir()

    # Catalog design
    page = np.full((600, 600, 3), (45, 25, 15), dtype=np.uint8)
    page[100:500, 100:500] = (215, 215, 215)
    cv2.circle(page, (300, 300), 60, (45, 25, 15), -1)
    cv2.rectangle(page, (150, 150), (450, 180), (45, 25, 15), -1)
    cv2.rectangle(page, (150, 420), (450, 450), (45, 25, 15), -1)

    pil = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
    pdf_p = cat_dir / "target.pdf"
    pil.save(str(pdf_p), "PDF")

    cfg = EngineConfig(
        DB_PATH=tmp_path / "pol.db",
        INDEX_PATH=tmp_path / "pol.index",
        MIN_CONFIDENCE=30.0,
    )
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    embedder = get_default_embedder("classical", cfg)
    index_folders([cat_dir], store=store, embedder=embedder, cfg=cfg)

    matcher = Matcher(store=store, embedder=embedder, cfg=cfg)

    # Query 1: Light panel on dark backdrop (e.g. dark floor)
    q_light_on_dark = np.zeros((600, 600, 3), dtype=np.uint8)
    q_light_on_dark[100:500, 100:500] = (215, 215, 215)
    cv2.circle(q_light_on_dark, (300, 300), 60, (0, 0, 0), -1)
    cv2.rectangle(q_light_on_dark, (150, 150), (450, 180), (0, 0, 0), -1)
    cv2.rectangle(q_light_on_dark, (150, 420), (450, 450), (0, 0, 0), -1)

    # Query 2: Dark panel on light backdrop (e.g. white wall)
    q_dark_on_light = np.full((600, 600, 3), 245, dtype=np.uint8)
    q_dark_on_light[100:500, 100:500] = (30, 30, 30)
    cv2.circle(q_dark_on_light, (300, 300), 60, (245, 245, 245), -1)
    cv2.rectangle(q_dark_on_light, (150, 150), (450, 180), (245, 245, 245), -1)
    cv2.rectangle(q_dark_on_light, (150, 420), (450, 450), (245, 245, 245), -1)

    res_light = matcher.search(q_light_on_dark, top_k=1, already_cropped=False)
    res_dark = matcher.search(q_dark_on_light, top_k=1, already_cropped=False)

    assert not res_light.no_confident_match
    assert not res_dark.no_confident_match
    assert len(res_light.results) == 1
    assert len(res_dark.results) == 1
    assert Path(res_light.results[0].pdf_path).name == "target.pdf"
    assert Path(res_dark.results[0].pdf_path).name == "target.pdf"
    assert res_light.results[0].score >= 80.0
    assert res_dark.results[0].score >= 80.0


def test_incremental_indexing_and_cancellation(tmp_path: Path):
    """Tests incremental change detection, skipping unchanged files, deletion removal, and cancellation."""
    cat_dir = tmp_path / "incremental_cat"
    cat_dir.mkdir()

    # Step 1: Create 2 PDFs
    for name in ["doc1.pdf", "doc2.pdf"]:
        img, _ = generate_distinct_pattern(0)
        page = np.full((550, 550, 3), (30, 25, 20), dtype=np.uint8)
        page[75:475, 75:475] = img
        pil = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        pil.save(str(cat_dir / name), "PDF")

    cfg = EngineConfig(DB_PATH=tmp_path / "inc.db", INDEX_PATH=tmp_path / "inc.index")
    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)

    # Initial Indexing
    stats1 = index_folders([cat_dir], store=store, cfg=cfg)
    assert stats1.indexed_files == 2
    assert stats1.skipped_files == 0
    assert store.index.ntotal == 2

    # Re-scan without changes -> should skip both files
    stats2 = index_folders([cat_dir], store=store, cfg=cfg)
    assert stats2.indexed_files == 0
    assert stats2.skipped_files == 2
    assert store.index.ntotal == 2

    # Modify doc1.pdf
    img_mod, _ = generate_distinct_pattern(1)
    page_mod = np.full((550, 550, 3), (30, 25, 20), dtype=np.uint8)
    page_mod[75:475, 75:475] = img_mod
    pil_mod = Image.fromarray(cv2.cvtColor(page_mod, cv2.COLOR_BGR2RGB))
    pil_mod.save(str(cat_dir / "doc1.pdf"), "PDF")

    stats3 = index_folders([cat_dir], store=store, cfg=cfg)
    assert stats3.indexed_files == 1
    assert stats3.skipped_files == 1
    assert store.index.ntotal == 2

    # Delete doc2.pdf from disk
    (cat_dir / "doc2.pdf").unlink()
    stats4 = index_folders([cat_dir], store=store, cfg=cfg)
    assert stats4.deleted_files == 1
    assert store.index.ntotal == 1

    # Test Cancellation
    cancel_event = threading.Event()
    cancel_event.set()
    stats5 = index_folders([cat_dir], cancel_event=cancel_event, store=store, cfg=cfg)
    assert stats5.cancelled is True


def test_empty_or_corrupt_input_handling(tmp_path: Path):
    """Asserts that degenerate images and corrupt PDFs do not crash the engine."""
    cfg = EngineConfig(DB_PATH=tmp_path / "err.db", INDEX_PATH=tmp_path / "err.index")
    matcher = Matcher(cfg=cfg)

    # Completely blank image (no cutouts)
    blank_bgr = np.full((300, 300, 3), 255, dtype=np.uint8)
    res_blank = matcher.search(blank_bgr)
    assert res_blank.no_confident_match is True
    assert "no distinct laser cut-out" in res_blank.reason.lower()

    # Corrupt PDF file indexing
    cat_dir = tmp_path / "corrupt_cat"
    cat_dir.mkdir()
    (cat_dir / "bad.pdf").write_bytes(b"NOT A REAL PDF CONTENT AT ALL")

    store = IndexStore(db_path=cfg.DB_PATH, index_path=cfg.INDEX_PATH)
    stats = index_folders([cat_dir], store=store, cfg=cfg)
    assert stats.errors == 1
    assert stats.indexed_files == 0

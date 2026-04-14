"""
VEDA Pipeline Router.

Orchestrates the full document-processing pipeline by calling the same
service functions used by the individual API routers — no logic is duplicated.

Flow (per request):
  1. Upload & Identify    — ingest logic (reused from routers.ingest helpers)
  2. Layout Analysis      — page-by-page YOLO via services.layout_engine
  3. Spatial Sort         — XY-Cut via services.spatial_sort_engine
  4. OCR / Gemini         — PARALLEL per page:
                            · GEMINI_LABELS  → gemini_engine.describe_image
                            · everything else → OCR (PyMuPDF → Tesseract → Gemini fallback)
  5. Finalize             — write JSON to disk, clean Redis

Rollback:
  Any un-caught exception triggers _cleanup(), which removes the uploaded
  file, debug images, Redis keys, and any partial output JSON.

Parallelism strategy:
  Within each page, every region is processed concurrently using
  asyncio.gather() + loop.run_in_executor().  The blocking Gemini and
  Tesseract calls run in the default ThreadPoolExecutor whose max_workers
  is set at startup from os.cpu_count() so it is always tuned to this
  machine (detected: see MAX_WORKERS constant below).
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import filetype
import fitz
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

# ── Service imports (direct calls — no self-HTTP) ────────────────────────────
from services.layout_engine import analyze_layout, draw_layout_on_image, pdf_to_images
from services.ocr_engine import (
    extract_full_page_text,
    extract_text_from_pdf_region,
    extract_text_from_region,
)
from services.gemini_engine import describe_image as gemini_describe_image
from services.gemini_engine import extract_text_with_gemini
from services.redis_client import (
    delete_file_keys,
    get_all_pages,
    set_page,
    set_total_pages,
)
from services.spatial_sort_engine import process_spatial_sort

# ── Reuse the PDF classifier from the ingest router (single source of truth) ─
from routers.ingest import _classify_pdf

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────

STORAGE_DIR = "storage"
DEBUG_DIR = "storage/debug"
ICEBERG_DIR = "storage/iceberg_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(ICEBERG_DIR, exist_ok=True)

# Labels routed to Gemini describe-image (visual regions).
# Everything NOT in this set goes to Tesseract / PyMuPDF OCR.
GEMINI_LABELS: set[str] = {
    "figure", "table", "image", "picture",
    "isolate_formula",
}

# Thread pool — size derived from os.cpu_count() at startup.
# I/O-bound tasks (Gemini API + Tesseract) benefit from more threads than cores,
# so we cap at cpu_count * 4 (but no more than 32 to avoid overwhelming Gemini).
_CPU_COUNT: int = os.cpu_count() or 4
MAX_WORKERS: int = min(_CPU_COUNT * 4, 32)
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

logger.info(
    f"Pipeline thread pool initialised — CPUs={_CPU_COUNT}, MAX_WORKERS={MAX_WORKERS}"
)


# ── Rollback helper ───────────────────────────────────────────────────────────

def _cleanup(file_id: str, file_path: str | None) -> None:
    """
    Remove every artefact tied to *file_id*.
    Safe to call even when some artefacts don't exist yet.
    """
    logger.warning(f"🔄 PIPELINE ROLLBACK: cleaning artefacts for file_id={file_id}")

    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"   [ROLLBACK] Deleted uploaded file: {file_path}")

    for img_path in glob.glob(os.path.join(DEBUG_DIR, f"{file_id}_*")):
        os.remove(img_path)
        logger.info(f"   [ROLLBACK] Deleted debug image: {img_path}")

    deleted = delete_file_keys(file_id)
    logger.info(f"   [ROLLBACK] Deleted {deleted} Redis key(s)")

    final_json = os.path.join(ICEBERG_DIR, f"{file_id}_final.json")
    if os.path.exists(final_json):
        os.remove(final_json)
        logger.info(f"   [ROLLBACK] Deleted partial final JSON: {final_json}")


# ── Page image loader ─────────────────────────────────────────────────────────

def _load_page_image(file_path: str, page_index: int) -> np.ndarray:
    """
    Load a single page as an OpenCV BGR image.
    Works for both plain image files (page_index must be 0) and PDFs.
    """
    img = cv2.imread(file_path)
    if img is not None:
        return img

    doc = fitz.open(file_path)
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(
            f"Page index {page_index} out of range "
            f"(document has {len(doc)} pages)"
        )
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)


# ── Per-region async coroutines ───────────────────────────────────────────────
# Each coroutine wraps one blocking call in run_in_executor so all regions
# on a page can run concurrently in the thread pool.

async def _process_gemini_region(
    region: dict,
    file_id: str,
    page_num: int,
    file_path: str,
    page_image: np.ndarray,
    lock: threading.Lock,
    counters: dict[str, int],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Process one GEMINI_LABELS region: call describe_image, fall back to OCR.
    Mutates *region* in-place.  Thread-safe counter updates via *lock*.
    """
    label = region.get("label", "?")
    bbox = region.get("bbox", [])
    t0 = time.time()
    logger.info(
        f"   [PARALLEL][Gemini] START  page={page_num} label={label} bbox={bbox}"
    )

    try:
        result: dict = await loop.run_in_executor(
            _EXECUTOR,
            lambda: gemini_describe_image(
                file_id=file_id,
                page=page_num,
                bbox=bbox,
                top_k=8,
            ),
        )
        region["text"] = result.get("gemini_response", "")
        region["gemini_caption"] = result.get("caption")
        region["gemini_context_text"] = result.get("context_text")
        region["gemini_response"] = result.get("gemini_response")

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"   [PARALLEL][Gemini] DONE   page={page_num} label={label} "
            f"chars={len(region['text'])} elapsed={elapsed:.1f}ms"
        )
        logger.debug(
            f"   [PARALLEL][Gemini] caption={result.get('caption')!r} "
            f"context_snippet={str(result.get('context_text',''))[:80]!r}"
        )

        with lock:
            counters["gemini"] += 1

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        logger.warning(
            f"   [PARALLEL][Gemini] FAILED page={page_num} label={label} "
            f"elapsed={elapsed:.1f}ms — {exc}. Falling back to OCR."
        )
        region["error"] = str(exc)

        # Fallback chain: PyMuPDF → Tesseract → Gemini plain-OCR
        text = await loop.run_in_executor(
            _EXECUTOR,
            lambda: extract_text_from_pdf_region(file_path, page_num, bbox),
        )
        if not text.strip():
            logger.info(
                f"   [PARALLEL][Fallback→Tesseract] page={page_num} label={label} bbox={bbox}"
            )
            text = await loop.run_in_executor(
                _EXECUTOR,
                lambda: extract_text_from_region(page_image, bbox),
            )
        if not text.strip():
            logger.info(
                f"   [PARALLEL][Fallback→GeminiOCR] page={page_num} label={label} bbox={bbox}"
            )
            text = await loop.run_in_executor(
                _EXECUTOR,
                lambda: extract_text_with_gemini(page_image, bbox),
            )
        region["text"] = text

        with lock:
            counters["ocr"] += 1

        logger.info(
            f"   [PARALLEL][Fallback] DONE page={page_num} label={label} "
            f"chars={len(text)} elapsed={(time.time()-t0)*1000:.1f}ms"
        )


async def _process_ocr_region(
    region: dict,
    file_id: str,
    page_num: int,
    file_path: str,
    page_image: np.ndarray,
    lock: threading.Lock,
    counters: dict[str, int],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Process one text region: PyMuPDF → Tesseract → Gemini fallback.
    Mutates *region* in-place.  Thread-safe counter updates via *lock*.
    """
    label = region.get("label", "?")
    bbox = region.get("bbox", [])
    t0 = time.time()
    logger.info(
        f"   [PARALLEL][OCR]    START  page={page_num} label={label} bbox={bbox}"
    )

    try:
        # 1. PyMuPDF native (fast, works for digital PDFs)
        text: str = await loop.run_in_executor(
            _EXECUTOR,
            lambda: extract_text_from_pdf_region(file_path, page_num, bbox),
        )

        if text.strip():
            logger.debug(
                f"   [PARALLEL][OCR]    PyMuPDF HIT page={page_num} label={label} chars={len(text)}"
            )
            with lock:
                counters["pymupdf"] += 1
        else:
            # 2. Tesseract (scanned docs)
            logger.debug(
                f"   [PARALLEL][OCR]    PyMuPDF MISS → Tesseract page={page_num} label={label}"
            )
            text = await loop.run_in_executor(
                _EXECUTOR,
                lambda: extract_text_from_region(page_image, bbox),
            )

            if text.strip():
                with lock:
                    counters["ocr"] += 1
            else:
                # 3. Gemini OCR fallback (last resort)
                logger.info(
                    f"   [PARALLEL][OCR]    Tesseract MISS → GeminiOCR page={page_num} label={label}"
                )
                text = await loop.run_in_executor(
                    _EXECUTOR,
                    lambda: extract_text_with_gemini(page_image, bbox),
                )
                with lock:
                    counters["gemini"] += 1

        region["text"] = text
        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"   [PARALLEL][OCR]    DONE   page={page_num} label={label} "
            f"chars={len(text)} elapsed={elapsed:.1f}ms"
        )

    except Exception as exc:
        elapsed = (time.time() - t0) * 1000
        logger.error(
            f"   [PARALLEL][OCR]    ERROR  page={page_num} label={label} "
            f"elapsed={elapsed:.1f}ms — {exc}"
        )
        region["text"] = ""
        region["error"] = str(exc)


async def _process_page(
    page_data: dict,
    file_id: str,
    file_path: str,
    counters: dict[str, int],
    lock: threading.Lock,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Process all regions on one page in parallel, then write the result to Redis.

    Gemini regions and OCR regions are dispatched simultaneously.
    """
    page_num: int = page_data.get("page", 0)
    regions: list[dict] = page_data.get("regions", [])

    logger.info(
        f"📄 [PAGE {page_num}] Starting parallel region processing — "
        f"{len(regions)} region(s)"
    )
    page_t0 = time.time()

    # Load the page image once (shared read-only across all region coroutines)
    page_image: np.ndarray = await loop.run_in_executor(
        _EXECUTOR,
        lambda: _load_page_image(file_path, page_num - 1),
    )
    logger.debug(f"   [PAGE {page_num}] Image loaded: shape={page_image.shape}")

    # Build one coroutine per region
    tasks: list[asyncio.coroutine] = []
    for region in regions:
        bbox = region.get("bbox", [])
        if not bbox or len(bbox) != 4:
            logger.warning(
                f"   [PAGE {page_num}] Skipping region with invalid bbox: {bbox}"
            )
            continue

        label = region.get("label", "").lower().replace(" ", "_")

        if label in GEMINI_LABELS:
            tasks.append(
                _process_gemini_region(
                    region, file_id, page_num, file_path,
                    page_image, lock, counters, loop,
                )
            )
        else:
            tasks.append(
                _process_ocr_region(
                    region, file_id, page_num, file_path,
                    page_image, lock, counters, loop,
                )
            )

    gemini_task_count = sum(
        1 for r in regions
        if r.get("label", "").lower().replace(" ", "_") in GEMINI_LABELS
        and r.get("bbox") and len(r.get("bbox", [])) == 4
    )
    ocr_task_count = len(tasks) - gemini_task_count
    logger.info(
        f"   [PAGE {page_num}] Dispatching {len(tasks)} tasks in parallel — "
        f"Gemini={gemini_task_count} OCR={ocr_task_count}"
    )

    # Run all region tasks concurrently (return_exceptions keeps one failure
    # from cancelling other regions)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any gather-level exceptions (shouldn't happen since coroutines
    # catch their own, but defensive programming)
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(
                f"   [PAGE {page_num}] Uncaught exception in task {i}: {res}"
            )

    # Full-page text fallback: if no region produced any text, try PyMuPDF
    page_has_text = any(r.get("text", "").strip() for r in regions)
    if not page_has_text and file_path.lower().endswith(".pdf"):
        logger.info(
            f"   [PAGE {page_num}] Zero text from all regions — "
            f"attempting full-page PyMuPDF extraction…"
        )
        full_text: str = await loop.run_in_executor(
            _EXECUTOR,
            lambda: extract_full_page_text(file_path, page_num),
        )
        if full_text.strip():
            # Attach to first non-visual region or synthesise one
            text_region = next(
                (
                    r for r in regions
                    if r.get("label", "").lower().replace(" ", "_") not in GEMINI_LABELS
                ),
                None,
            )
            if text_region:
                text_region["text"] = full_text
            else:
                regions.append({
                    "label": "text",
                    "bbox": [0, 0, 100, 100],
                    "text": full_text,
                    "confidence": 1.0,
                    "id": "r_fullpage",
                    "reading_order": 0,
                })
            with lock:
                counters["pymupdf"] += 1
            logger.info(
                f"   [PAGE {page_num}] Full-page fallback extracted "
                f"{len(full_text)} chars."
            )
        else:
            logger.warning(
                f"   [PAGE {page_num}] Full-page fallback also returned no text."
            )

    # Write updated page back to Redis
    set_page(file_id, page_num, page_data)

    elapsed = (time.time() - page_t0) * 1000
    logger.info(
        f"📄 [PAGE {page_num}] Completed in {elapsed:.1f}ms — "
        f"regions processed={len(tasks)}"
    )


# ── Pipeline Endpoint ─────────────────────────────────────────────────────────

@router.post("/pipeline")
async def run_pipeline(
    file: UploadFile = File(...),
    start_page: int = Query(1, ge=1, description="1-indexed page to start from"),
) -> dict:
    """
    Run the full VEDA document-processing pipeline on an uploaded file.

    Steps:
      1. Upload & Identify
      2. Layout Analysis  (YOLO, page-by-page, from start_page)
      3. Spatial Sort     (XY-Cut reading-order)
      4. OCR & Description (parallel per page: Gemini for visuals, OCR for text)
      5. Finalize         (write JSON to disk, clean Redis)

    On failure, all artefacts are rolled back automatically.
    """
    pipeline_start = time.time()
    file_id: str | None = None
    file_path: str | None = None
    steps_log: list[dict] = []
    step: int = 0
    step_name: str = "Init"

    loop = asyncio.get_event_loop()

    try:
        # ── STEP 1: Upload & Identify ─────────────────────────────────────────
        step = 1
        step_name = "Upload & Identify"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(STORAGE_DIR, f"{file_id}{file_extension}")

        # Save to disk
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        logger.info(
            f"   [STEP 1] Saved to disk: {file_path} "
            f"({os.path.getsize(file_path)} bytes)"
        )

        # Detect file type via magic bytes (reuses _classify_pdf from ingest.py)
        kind = filetype.guess(file_path)
        category = "UNKNOWN"
        mime_type = "unknown/unknown"

        if kind:
            mime_type = kind.mime
            logger.debug(f"   [STEP 1] Detected MIME: {mime_type}")
            if mime_type.startswith("image/"):
                category = "IMAGE"
            elif mime_type == "application/pdf":
                category = _classify_pdf(file_path)   # ← reused, not redefined
            elif "word" in mime_type or "officedocument.wordprocessingml" in mime_type:
                category = "OFFICE_WORD"
            elif "presentation" in mime_type or "powerpoint" in mime_type:
                category = "OFFICE_PPT"

        if category == "UNKNOWN":
            if file_extension in [".doc", ".docx"]:
                category = "OFFICE_WORD"
            elif file_extension in [".ppt", ".pptx"]:
                category = "OFFICE_PPT"
            elif file_extension in [".txt"]:
                category = "TEXT_FILE"

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name,
            "status": "success", "time_ms": round(step_time, 2),
            "details": {"file_id": file_id, "category": category},
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — DONE in {step_time:.2f}ms "
            f"(file_id={file_id}, category={category})"
        )

        # ── STEP 2: Layout Analysis ───────────────────────────────────────────
        step = 2
        step_name = "Layout Analysis"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        images = pdf_to_images(file_bytes)
        total_pages = len(images)

        logger.info(f"   [STEP 2] Total pages in document: {total_pages}")

        if start_page > total_pages:
            raise ValueError(
                f"start_page={start_page} exceeds total pages ({total_pages})."
            )

        pages_to_process = list(range(start_page - 1, total_pages))  # 0-indexed
        layout_results: list[dict] = []

        for page_idx in pages_to_process:
            img = images[page_idx]
            page_num = page_idx + 1  # 1-indexed
            page_t0 = time.time()
            logger.info(
                f"   [STEP 2] Page {page_num}/{total_pages} — running YOLO…"
            )

            regions = analyze_layout(img)

            output_filename = f"{file_id}_page_{page_num}.jpg"
            output_path = os.path.join(DEBUG_DIR, output_filename)
            draw_layout_on_image(img, regions, output_path)

            page_result = {
                "page": page_num,
                "regions": regions,
                "meta": {
                    "process_time_ms": round((time.time() - page_t0) * 1000, 2),
                    "model": "doclayout_yolo",
                },
                "debug_image_url": f"/api/v1/layout/debug_image/{output_filename}",
            }
            layout_results.append(page_result)
            set_page(file_id, page_num, page_result)

            logger.info(
                f"   [STEP 2] Page {page_num}/{total_pages} — "
                f"{len(regions)} regions found "
                f"({page_result['meta']['process_time_ms']}ms)"
            )

        set_total_pages(file_id, total_pages)
        logger.info(f"   [STEP 2] Cached {len(layout_results)} pages in Redis")

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name,
            "status": "success", "time_ms": round(step_time, 2),
            "details": {
                "total_pages": total_pages,
                "pages_processed": len(pages_to_process),
                "start_page": start_page,
            },
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — DONE in {step_time:.2f}ms "
            f"({len(pages_to_process)}/{total_pages} pages from page {start_page})"
        )

        # ── STEP 3: Spatial Sort ──────────────────────────────────────────────
        step = 3
        step_name = "Spatial Sort"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        pages_from_redis = get_all_pages(file_id)
        if not pages_from_redis:
            raise RuntimeError("No pages found in Redis after layout analysis.")
        logger.info(
            f"   [STEP 3] Fetched {len(pages_from_redis)} pages from Redis for sorting"
        )

        # Reuse process_spatial_sort directly (same function the spatial_sort router uses)
        ordered_payload = process_spatial_sort({"layout_data": pages_from_redis})

        for page_data in ordered_payload.get("layout_data", []):
            page_num = page_data.get("page")
            if page_num is not None:
                set_page(file_id, page_num, page_data)
                logger.debug(
                    f"   [STEP 3] Wrote sorted page {page_num} back to Redis "
                    f"({len(page_data.get('regions', []))} regions)"
                )

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name,
            "status": "success", "time_ms": round(step_time, 2),
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — DONE in {step_time:.2f}ms"
        )

        # ── STEP 4: Parallel OCR / Gemini ─────────────────────────────────────
        step = 4
        step_name = "OCR & Description (Parallel)"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        logger.info(
            f"   [STEP 4] Thread pool: MAX_WORKERS={MAX_WORKERS} (CPUs={_CPU_COUNT})"
        )
        step_start = time.time()

        sorted_pages = get_all_pages(file_id)
        logger.info(
            f"   [STEP 4] Processing {len(sorted_pages)} pages — "
            f"one page at a time, all regions in parallel within each page"
        )

        # Shared counters updated thread-safely across region coroutines
        counters: dict[str, int] = {"ocr": 0, "gemini": 0, "pymupdf": 0}
        lock = threading.Lock()

        for page_data in sorted_pages:
            await _process_page(
                page_data=page_data,
                file_id=file_id,
                file_path=file_path,
                counters=counters,
                lock=lock,
                loop=loop,
            )

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name,
            "status": "success", "time_ms": round(step_time, 2),
            "details": {
                "ocr_regions": counters["ocr"],
                "gemini_regions": counters["gemini"],
                "pymupdf_regions": counters["pymupdf"],
                "max_workers": MAX_WORKERS,
            },
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — DONE in {step_time:.2f}ms "
            f"(OCR={counters['ocr']}, Gemini={counters['gemini']}, "
            f"PyMuPDF={counters['pymupdf']})"
        )

        # ── STEP 5: Finalize ──────────────────────────────────────────────────
        step = 5
        step_name = "Finalize"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        final_pages = get_all_pages(file_id)
        if not final_pages:
            raise RuntimeError("No pages found in Redis for finalization.")

        final_document: dict[str, Any] = {
            "file_id": file_id,
            "total_pages": len(final_pages),
            "pages": final_pages,
        }

        output_path = os.path.join(ICEBERG_DIR, f"{file_id}_final.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_document, f, indent=2, ensure_ascii=False)

        logger.info(
            f"   [STEP 5] Final JSON written to iceberg_storage: "
            f"{os.path.abspath(output_path)} "
            f"({os.path.getsize(output_path)} bytes)"
        )

        deleted_count = delete_file_keys(file_id)
        logger.info(f"   [STEP 5] Cleaned {deleted_count} Redis key(s)")

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name,
            "status": "success", "time_ms": round(step_time, 2),
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — DONE in {step_time:.2f}ms"
        )

        # ── SUCCESS ───────────────────────────────────────────────────────────
        total_time = (time.time() - pipeline_start) * 1000
        logger.info(
            f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}ms "
            f"for file_id={file_id}"
        )

        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename,
            "category": category,
            "total_pages": total_pages,
            "start_page": start_page,
            "pages_processed": len(pages_to_process),
            "steps": steps_log,
            "output_path": output_path,
            "final_document": final_document,
            "total_time_ms": round(total_time, 2),
        }

    except Exception as exc:
        # ── FAILURE — log, rollback, raise ───────────────────────────────────
        error_msg = traceback.format_exc()
        logger.error(
            f"❌ PIPELINE STEP {step} ({step_name}) — FAILED\n"
            f"   steps so far: {steps_log}\n"
            f"{error_msg}"
        )
        steps_log.append({
            "step": step, "name": step_name,
            "status": "failed", "error": str(exc),
        })

        if file_id:
            _cleanup(file_id, file_path)

        total_time = (time.time() - pipeline_start) * 1000
        logger.error(
            f"💀 PIPELINE ABORTED after {total_time:.2f}ms "
            f"at step {step} ({step_name})"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "failed_at_step": step,
                "failed_step_name": step_name,
                "error": str(exc),
                "steps": steps_log,
                "total_time_ms": round(total_time, 2),
            },
        )

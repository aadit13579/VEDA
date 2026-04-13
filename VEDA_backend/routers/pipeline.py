"""
VEDA Pipeline Router.

Single endpoint that orchestrates the full document-processing flow:
  1. Upload & Identify
  2. Layout Analysis  (page-by-page)
  3. Spatial Sort
  4. OCR  (Tesseract for text)  /  Describe-Image  (Gemini for visuals)
  5. Finalize  (write JSON to disk, clean Redis)

If any step fails the pipeline rolls back all artefacts
(uploaded file, debug images, Redis keys, partial output).
"""

import os
import glob
import shutil
import uuid
import time
import traceback
import json

import cv2
import fitz
import numpy as np
import filetype
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

# --- Service imports (direct calls — no self-HTTP) ---
from services.layout_engine import pdf_to_images, analyze_layout, draw_layout_on_image
from services.spatial_sort_engine import process_spatial_sort
from services.ocr_engine import extract_text_from_region, extract_text_from_pdf_region, extract_full_page_text
from services.gemini_engine import describe_image as gemini_describe_image, extract_text_with_gemini
from services.redis_client import (
    set_page, get_page, get_all_pages,
    set_total_pages, delete_file_keys,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ---------- Constants ----------

STORAGE_DIR = "storage"
DEBUG_DIR = "storage/debug"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Labels routed to Gemini describe-image (visual regions)
GEMINI_LABELS = {
    "figure", "table", "image", "picture",
    "isolate_formula", "figure_caption", "table_caption",
}

# Everything else goes to Tesseract OCR.
# We use the set above to decide; if a label is NOT in GEMINI_LABELS → Tesseract.


# ---------- Helpers ----------

def _classify_pdf(path: str) -> str:
    """Check whether a PDF is text-based (digital) or image-based (scanned)."""
    try:
        doc = fitz.open(path)
        text_length = 0
        for i in range(min(3, len(doc))):
            text_length += len(doc[i].get_text())
        return "PDF_DIGITAL" if text_length > 50 else "PDF_SCANNED"
    except Exception:
        return "PDF_SCANNED"


def _cleanup(file_id: str, file_path: str | None):
    """
    Rollback helper — remove every artefact associated with *file_id*.
    Safe to call even if some artefacts don't exist yet.
    """
    logger.info(f"🔄 PIPELINE ROLLBACK: Cleaning up artefacts for {file_id}")

    # 1. Uploaded file
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"   Deleted uploaded file: {file_path}")

    # 2. Debug images
    for img_path in glob.glob(os.path.join(DEBUG_DIR, f"{file_id}_*")):
        os.remove(img_path)
        logger.info(f"   Deleted debug image: {img_path}")

    # 3. Redis keys
    deleted = delete_file_keys(file_id)
    logger.info(f"   Deleted {deleted} Redis key(s)")

    # 4. Partial final JSON
    final_json = os.path.join(STORAGE_DIR, f"{file_id}_final.json")
    if os.path.exists(final_json):
        os.remove(final_json)
        logger.info(f"   Deleted partial final JSON: {final_json}")


def _load_page_image(file_path: str, page_index: int):
    """
    Load a single page as an OpenCV image.
    Works for both plain images (page_index must be 0) and PDFs.
    """
    img = cv2.imread(file_path)
    if img is not None:
        return img

    # PDF path
    doc = fitz.open(file_path)
    if page_index < 0 or page_index >= len(doc):
        raise ValueError(f"Page index {page_index} out of range (document has {len(doc)} pages)")
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)


# ---------- Pipeline Endpoint ----------

@router.post("/pipeline")
async def run_pipeline(
    file: UploadFile = File(...),
    start_page: int = Query(1, ge=1, description="1-indexed page to start processing from"),
):
    """
    Run the full VEDA document-processing pipeline on an uploaded file.

    Args:
      file:       The document to process.
      start_page: 1-indexed page number to start from (default 1).

    Steps:
      1. Upload & Identify
      2. Layout Analysis  (page-by-page YOLO inference, from start_page)
      3. Spatial Sort      (XY-Cut reading-order)
      4. OCR / Describe-Image  (Tesseract for text, Gemini for visuals)
      5. Finalize          (write JSON, clean Redis)

    On failure at any step the pipeline rolls back all artefacts.
    """
    pipeline_start = time.time()
    file_id: str | None = None
    file_path: str | None = None
    steps_log: list[dict] = []

    try:
        # ============================================================
        # STEP 1 — Upload & Identify
        # ============================================================
        step = 1
        step_name = "Upload & Identify"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(STORAGE_DIR, f"{file_id}{file_extension}")

        # Save to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect type
        kind = filetype.guess(file_path)
        category = "UNKNOWN"
        mime_type = "unknown/unknown"

        if kind:
            mime_type = kind.mime
            if mime_type.startswith("image/"):
                category = "IMAGE"
            elif mime_type == "application/pdf":
                category = _classify_pdf(file_path)
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
        steps_log.append({"step": step, "name": step_name, "status": "success", "time_ms": round(step_time, 2)})
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — COMPLETED in {step_time:.2f}ms "
            f"(file_id={file_id}, category={category})"
        )

        # ============================================================
        # STEP 2 — Layout Analysis  (page-by-page)
        # ============================================================
        step = 2
        step_name = "Layout Analysis"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        # Convert to images first
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        images = pdf_to_images(file_bytes)
        total_pages = len(images)

        # Validate start_page
        if start_page > total_pages:
            raise ValueError(
                f"start_page={start_page} exceeds total pages ({total_pages})."
            )

        layout_results: list[dict] = []
        pages_to_process = list(range(start_page - 1, total_pages))  # 0-indexed

        for page_idx in pages_to_process:
            img = images[page_idx]
            page_num = page_idx + 1  # 1-indexed
            page_start = time.time()
            logger.info(f"   📄 Page {page_num}/{total_pages} — running YOLO layout analysis…")

            regions = analyze_layout(img)

            # Draw debug image
            output_filename = f"{file_id}_page_{page_num}.jpg"
            output_path = os.path.join(DEBUG_DIR, output_filename)
            draw_layout_on_image(img, regions, output_path)

            page_result = {
                "page": page_num,
                "regions": regions,
                "meta": {
                    "process_time_ms": round((time.time() - page_start) * 1000, 2),
                    "model": "doclayout_yolo",
                },
                "debug_image_url": f"/api/v1/layout/debug_image/{output_filename}",
            }
            layout_results.append(page_result)

            # Cache this page in Redis immediately
            set_page(file_id, page_num, page_result)
            logger.info(f"   📄 Page {page_num}/{total_pages} — done ({page_result['meta']['process_time_ms']}ms)")

        set_total_pages(file_id, total_pages)

        step_time = (time.time() - step_start) * 1000
        pages_processed = len(pages_to_process)
        steps_log.append({"step": step, "name": step_name, "status": "success", "time_ms": round(step_time, 2),
                          "details": {"total_pages": total_pages, "pages_processed": pages_processed, "start_page": start_page}})
        logger.info(f"✅ PIPELINE STEP {step}: {step_name} — COMPLETED in {step_time:.2f}ms ({pages_processed}/{total_pages} pages, from page {start_page})")

        # ============================================================
        # STEP 3 — Spatial Sort
        # ============================================================
        step = 3
        step_name = "Spatial Sort"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        pages_from_redis = get_all_pages(file_id)
        if not pages_from_redis:
            raise RuntimeError("No pages found in Redis after layout analysis.")

        payload = {"layout_data": pages_from_redis}
        ordered_payload = process_spatial_sort(payload)

        # Write sorted pages back to Redis
        for page_data in ordered_payload.get("layout_data", []):
            page_num = page_data.get("page")
            if page_num is not None:
                set_page(file_id, page_num, page_data)

        step_time = (time.time() - step_start) * 1000
        steps_log.append({"step": step, "name": step_name, "status": "success", "time_ms": round(step_time, 2)})
        logger.info(f"✅ PIPELINE STEP {step}: {step_name} — COMPLETED in {step_time:.2f}ms")

        # ============================================================
        # STEP 4 — OCR  /  Describe-Image
        # ============================================================
        step = 4
        step_name = "OCR & Description"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        ocr_count = 0
        gemini_count = 0
        pymupdf_count = 0

        # Re-fetch sorted pages from Redis
        sorted_pages = get_all_pages(file_id)

        for page_data in sorted_pages:
            page_num = page_data.get("page", 1)
            regions = page_data.get("regions", [])

            # Load the page image once per page (only if Tesseract or Gemini needs it)
            page_image = None

            # Track whether any region on this page got text
            page_has_text = False

            for region in regions:
                label = region.get("label", "").lower().replace(" ", "_")
                bbox = region.get("bbox", [])

                if not bbox or len(bbox) != 4:
                    continue

                if label in GEMINI_LABELS:
                    # --- Gemini Describe-Image ---
                    try:
                        logger.info(
                            f"   🤖 Gemini: page={page_num}, label={label}, bbox={bbox}"
                        )
                        result = gemini_describe_image(
                            file_id=file_id,
                            page=page_num,
                            bbox=bbox,
                            top_k=8,
                        )
                        region["text"] = result.get("gemini_response", "")
                        region["gemini_caption"] = result.get("caption")
                        region["gemini_context_text"] = result.get("context_text")
                        region["gemini_response"] = result.get("gemini_response")
                        gemini_count += 1
                        if region["text"].strip():
                            page_has_text = True
                    except Exception as e:
                        logger.warning(
                            f"   ⚠️ Gemini failed for page={page_num}, bbox={bbox}: {e}. "
                            f"Falling back to text extraction."
                        )
                        # Fallback: try PyMuPDF first, then Tesseract, then Gemini
                        text = extract_text_from_pdf_region(file_path, page_num, bbox)
                        if not text.strip():
                            if page_image is None:
                                page_image = _load_page_image(file_path, page_num - 1)
                            text = extract_text_from_region(page_image, bbox)
                            # FINAL FAILSAFE: Gemini OCR if Tesseract is missing/failed
                            if not text.strip():
                                logger.info(f"   🤖 Gemini OCR Fallback for label={label}, bbox={bbox}")
                                text = extract_text_with_gemini(page_image, bbox)
                        region["text"] = text
                        ocr_count += 1
                        if text.strip():
                            page_has_text = True
                else:
                    # --- Text Extraction: PyMuPDF first, Tesseract fallback ---
                    logger.info(
                        f"   📝 Text: page={page_num}, label={label}, bbox={bbox}"
                    )
                    # Try PyMuPDF native extraction first (works for digital PDFs)
                    text = extract_text_from_pdf_region(file_path, page_num, bbox)
                    if text.strip():
                        pymupdf_count += 1
                    else:
                        # Fall back to Tesseract OCR (for scanned docs)
                        if page_image is None:
                            page_image = _load_page_image(file_path, page_num - 1)
                        text = extract_text_from_region(page_image, bbox)
                        
                        # FINAL FAILSAFE: Gemini OCR if Tesseract is missing/failed
                        if not text.strip():
                            logger.info(f"   🤖 Gemini OCR Fallback for label={label}, bbox={bbox}")
                            text = extract_text_with_gemini(page_image, bbox)
                            if text.strip():
                                gemini_count += 1

                    region["text"] = text
                    ocr_count += 1
                    if text.strip():
                        page_has_text = True

            # FINAL FALLBACK: If no region on this page got any text,
            # try extracting full-page text with PyMuPDF and assign it
            # to the first text-type region (or create a synthetic one).
            if not page_has_text and file_path.lower().endswith(".pdf"):
                logger.info(
                    f"   ⚠️ No text extracted for page {page_num} via regions. "
                    f"Trying full-page PyMuPDF extraction…"
                )
                full_text = extract_full_page_text(file_path, page_num)
                if full_text.strip():
                    # Find first text-type region to attach it to
                    text_region = None
                    for region in regions:
                        lbl = region.get("label", "").lower().replace(" ", "_")
                        if lbl not in GEMINI_LABELS:
                            text_region = region
                            break

                    if text_region:
                        text_region["text"] = full_text
                    else:
                        # No text region exists — add a synthetic one
                        regions.append({
                            "label": "text",
                            "bbox": [0, 0, 100, 100],
                            "text": full_text,
                            "confidence": 1.0,
                            "id": "r_fullpage",
                            "reading_order": 0,
                        })
                    page_has_text = True
                    pymupdf_count += 1
                    logger.info(
                        f"   ✅ Full-page fallback extracted {len(full_text)} chars for page {page_num}"
                    )

            # Write updated page back to Redis
            set_page(file_id, page_num, page_data)

        step_time = (time.time() - step_start) * 1000
        steps_log.append({
            "step": step, "name": step_name, "status": "success",
            "time_ms": round(step_time, 2),
            "details": {
                "ocr_regions": ocr_count,
                "gemini_regions": gemini_count,
                "pymupdf_regions": pymupdf_count,
            },
        })
        logger.info(
            f"✅ PIPELINE STEP {step}: {step_name} — COMPLETED in {step_time:.2f}ms "
            f"(OCR={ocr_count}, Gemini={gemini_count}, PyMuPDF={pymupdf_count})"
        )

        # ============================================================
        # STEP 5 — Finalize
        # ============================================================
        step = 5
        step_name = "Finalize"
        logger.info(f"🔵 PIPELINE STEP {step}: {step_name} — STARTED")
        step_start = time.time()

        final_pages = get_all_pages(file_id)
        if not final_pages:
            raise RuntimeError("No pages found in Redis for finalization.")

        final_document = {
            "file_id": file_id,
            "total_pages": len(final_pages),
            "pages": final_pages,
        }

        output_path = os.path.join(STORAGE_DIR, f"{file_id}_final.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_document, f, indent=2, ensure_ascii=False)

        logger.info(f"   Final JSON written to {os.path.abspath(output_path)}")

        # Clean up Redis
        deleted_count = delete_file_keys(file_id)
        logger.info(f"   Cleaned {deleted_count} Redis key(s)")

        step_time = (time.time() - step_start) * 1000
        steps_log.append({"step": step, "name": step_name, "status": "success", "time_ms": round(step_time, 2)})
        logger.info(f"✅ PIPELINE STEP {step}: {step_name} — COMPLETED in {step_time:.2f}ms")

        # ============================================================
        # SUCCESS — Return full response
        # ============================================================
        total_time = (time.time() - pipeline_start) * 1000
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}ms for file_id={file_id}")

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

    except Exception as e:
        # ============================================================
        # FAILURE — Log, rollback, return error
        # ============================================================
        error_msg = traceback.format_exc()
        logger.error(
            f"❌ PIPELINE STEP {step}: {step_name} — FAILED\n{error_msg}"
        )

        # Record the failed step
        steps_log.append({"step": step, "name": step_name, "status": "failed", "error": str(e)})

        # Rollback all artefacts
        if file_id:
            _cleanup(file_id, file_path)

        total_time = (time.time() - pipeline_start) * 1000
        logger.error(f"💀 PIPELINE ABORTED after {total_time:.2f}ms at step {step} ({step_name})")

        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "failed_at_step": step,
                "failed_step_name": step_name,
                "error": str(e),
                "steps": steps_log,
                "total_time_ms": round(total_time, 2),
            },
        )

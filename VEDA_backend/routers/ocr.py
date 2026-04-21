"""
OCR Router.

Performs OCR on a single bounding box region.
After extraction, writes the text back into the matching region in Redis.
"""

import time
import cv2
import numpy as np
import os
import glob
import fitz
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.ocr_engine import extract_text_from_region
from services.redis_client import get_page, set_page, bbox_matches
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

STORAGE_DIR = "storage"


class OCRRequest(BaseModel):
    bbox: List[int]          # [x1, y1, x2, y2]
    page: Optional[int] = 1  # Page number (for multi-page PDFs)


@router.post("/layout/ocr/{file_id}")
async def perform_ocr(file_id: str, request: OCRRequest):
    """
    Performs OCR on a single bounding box region.

    - Accepts a file_id (image already on disk from /upload)
    - Accepts a single bbox [x1, y1, x2, y2] from the spatial sort output
    - Returns the extracted text for that region only
    - Writes the text back to the matching region in Redis
    """
    start_time = time.time()
    logger.info(f"OCR request for file_id={file_id}, page={request.page}, bbox={request.bbox}")

    # 1. Find the file on disk
    search_pattern = os.path.join(STORAGE_DIR, f"{file_id}.*")
    files = glob.glob(search_pattern)

    if not files:
        raise HTTPException(
            status_code=404,
            detail="File not found. Please upload via /ingest first."
        )

    file_path = files[0]

    # 2. Load the correct page image
    img = cv2.imread(file_path)

    if img is None:
        # File is likely a PDF — convert the requested page to an image
        try:
            doc = fitz.open(file_path)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read the file as an image or PDF."
            )

        page_index = request.page - 1
        if page_index < 0 or page_index >= len(doc):
            raise HTTPException(
                status_code=400,
                detail=f"Page {request.page} does not exist. Document has {len(doc)} page(s)."
            )

        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    # 3. Extract text from the single bounding box
    text = extract_text_from_region(img, request.bbox)

    # 4. Write text back to matching region in Redis
    _write_text_to_redis(file_id, request.page, request.bbox, text)

    process_time = (time.time() - start_time) * 1000
    logger.info(f"OCR completed for file_id={file_id} in {process_time:.2f}ms")

    return {
        "file_id": file_id,
        "page": request.page,
        "bbox": request.bbox,
        "text": text,
        "process_time_ms": round(process_time, 2)
    }


def _write_text_to_redis(file_id: str, page: int, bbox: List[int], text: str) -> None:
    """
    Find the region in Redis whose bbox matches and attach the OCR text.

    If the page or region is not found in Redis, this is a no-op
    (the OCR result is still returned to the caller).
    """
    page_data = get_page(file_id, page)
    if page_data is None:
        logger.debug(f"Redis: page {page} not cached for file {file_id}, skipping write-back.")
        return

    regions = page_data.get("regions", [])
    matched = False

    for region in regions:
        region_bbox = region.get("bbox", [])
        if bbox_matches(region_bbox, bbox):
            region["text"] = text
            matched = True
            break

    if matched:
        set_page(file_id, page, page_data)
        logger.info(f"Redis: wrote OCR text to region with bbox={bbox} on page {page}")
    else:
        logger.warning(
            f"Redis: no region with bbox={bbox} found on page {page} for file {file_id}. "
            f"Text not written back."
        )

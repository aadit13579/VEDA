"""
VEDA OCR Router.

Performs OCR on a single bounding box region of an uploaded document.
After text extraction, writes the result back into the matching region
in Redis so that downstream consumers see the OCR text without
re-extracting it.

This module exists to expose region-level OCR as a standalone API
endpoint, allowing the frontend or other tools to trigger OCR on
specific regions independently of the full pipeline.

Leverages: FastAPI, PyMuPDF (fitz), OpenCV (cv2), Tesseract (via ocr_engine),
           Redis client.
"""

import os
import time
import glob

import cv2
import fitz
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.ocr_engine import extract_text_from_region
from services.redis_client import get_page, set_page, bbox_matches
from utils.logger import get_logger

logger = get_logger(__name__)


class OCRRequest(BaseModel):
    """
    Request payload for the single-region OCR endpoint.

    Specifies the bounding box coordinates and the (optional) page
    number for multi-page PDF documents.

    Leverages: Pydantic BaseModel.
    """

    bbox: List[int]
    page: Optional[int] = 1


class OCRRouter:
    """
    Router for performing OCR on individual document regions.

    Loads the document image (or PDF page), crops the specified bounding
    box, runs Tesseract OCR, writes the result back to Redis, and
    returns the extracted text.

    Leverages: FastAPI APIRouter, ocr_engine, redis_client, fitz, cv2.
    """

    STORAGE_DIR = "storage"

    def __init__(self):
        """
        Initialize the router and register routes.

        Leverages: FastAPI APIRouter.
        """
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Bind endpoint methods to their HTTP routes.

        Leverages: FastAPI APIRouter.add_api_route.
        """
        self.router.add_api_route(
            "/layout/ocr/{file_id}",
            self.perform_ocr,
            methods=["POST"],
        )

    async def perform_ocr(self, file_id: str, request: OCRRequest):
        """
        Perform OCR on a single bounding box region.

        Finds the file on disk, loads the correct page image (handling
        both plain images and PDFs), extracts text using Tesseract, writes
        the result back to the matching Redis region, and returns the text.

        Leverages: glob, cv2.imread, fitz, extract_text_from_region.
        """
        start_time = time.time()
        logger.info(
            f"OCR request for file_id={file_id}, page={request.page}, bbox={request.bbox}"
        )

        search_pattern = os.path.join(self.STORAGE_DIR, f"{file_id}.*")
        files = glob.glob(search_pattern)

        if not files:
            raise HTTPException(
                status_code=404,
                detail="File not found. Please upload via /ingest first.",
            )

        file_path = files[0]

        img = cv2.imread(file_path)

        if img is None:
            try:
                doc = fitz.open(file_path)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read the file as an image or PDF.",
                )

            page_index = request.page - 1
            if page_index < 0 or page_index >= len(doc):
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {request.page} does not exist. "
                    f"Document has {len(doc)} page(s).",
                )

            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        text = extract_text_from_region(img, request.bbox)

        self._write_text_to_redis(file_id, request.page, request.bbox, text)

        process_time = (time.time() - start_time) * 1000
        logger.info(f"OCR completed for file_id={file_id} in {process_time:.2f}ms")

        return {
            "file_id": file_id,
            "page": request.page,
            "bbox": request.bbox,
            "text": text,
            "process_time_ms": round(process_time, 2),
        }

    def _write_text_to_redis(
        self, file_id: str, page: int, bbox: List[int], text: str
    ) -> None:
        """
        Find the matching region in Redis and attach the OCR text.

        If the page or region is not found in Redis, this is a no-op —
        the OCR result is still returned to the caller.

        Leverages: get_page, set_page, bbox_matches.
        """
        page_data = get_page(file_id, page)
        if page_data is None:
            logger.debug(
                f"Redis: page {page} not cached for file {file_id}, skipping write-back."
            )
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
            logger.info(
                f"Redis: wrote OCR text to region with bbox={bbox} on page {page}"
            )
        else:
            logger.warning(
                f"Redis: no region with bbox={bbox} found on page {page} "
                f"for file {file_id}. Text not written back."
            )


_instance = OCRRouter()
router = _instance.router

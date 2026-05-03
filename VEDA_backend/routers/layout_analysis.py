"""
VEDA Layout Analysis Router.

Provides endpoints for running YOLO-based document layout analysis
and serving the resulting debug images. Processes each page of an
uploaded document, detects text/table/figure regions, draws bounding
box overlays, and caches the results in Redis.

This module exists to expose layout analysis as a standalone API
endpoint, allowing the frontend or external tools to trigger analysis
independently of the full pipeline.

Leverages: FastAPI, DocLayout-YOLO (via layout_engine), Redis, OpenCV.
"""

import os
import time
import glob
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.layout_engine import pdf_to_images, analyze_layout, draw_layout_on_image
from services.redis_client import set_page, set_total_pages
from utils.logger import get_logger

logger = get_logger(__name__)


class LayoutAnalysisRouter:
    """
    Router for document layout analysis and debug image serving.

    Orchestrates the conversion of uploaded documents to images, runs
    YOLO layout detection on each page, saves debug images with
    bounding boxes, and caches results in Redis.

    Leverages: FastAPI APIRouter, layout_engine, redis_client.
    """

    STORAGE_DIR = "storage"
    DEBUG_DIR = "storage/debug"

    def __init__(self):
        """
        Initialize the router and ensure the debug directory exists.

        Leverages: os.makedirs, FastAPI APIRouter.
        """
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Bind endpoint methods to their HTTP routes.

        Leverages: FastAPI APIRouter.add_api_route.
        """
        self.router.add_api_route(
            "/analyze_layout/{file_id}",
            self.generate_bounding_boxes,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/layout/debug_image/{filename}",
            self.get_debug_image,
            methods=["GET"],
        )

    async def generate_bounding_boxes(self, file_id: str):
        """
        Run YOLO layout analysis on all pages of an uploaded document.

        Finds the file by file_id, converts PDF pages to images, runs
        layout detection, draws debug overlays, caches results in Redis,
        and returns the structured layout data with debug image URLs.

        Leverages: glob, pdf_to_images, analyze_layout, draw_layout_on_image, Redis.
        """
        logger.info(f"Received layout analysis request for file_id: {file_id}")
        start_time = time.time()

        search_pattern = os.path.join(self.STORAGE_DIR, f"{file_id}.*")
        files = glob.glob(search_pattern)

        if not files:
            raise HTTPException(
                status_code=404,
                detail="File not found. Please upload via /ingest first.",
            )

        file_path = files[0]

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            images = pdf_to_images(file_bytes)

            results = []

            for page_num, img in enumerate(images):
                page_start = time.time()
                regions = analyze_layout(img)
                page_time = (time.time() - page_start) * 1000

                output_filename = f"{file_id}_page_{page_num + 1}.jpg"
                output_path = os.path.join(self.DEBUG_DIR, output_filename)
                draw_layout_on_image(img, regions, output_path)

                results.append(
                    {
                        "page": page_num + 1,
                        "regions": regions,
                        "meta": {
                            "process_time_ms": round(page_time, 2),
                            "model": "doclayout_yolo",
                        },
                        "debug_image_url": f"/api/v1/layout/debug_image/{output_filename}",
                    }
                )

            for page_result in results:
                set_page(file_id, page_result["page"], page_result)
            set_total_pages(file_id, len(images))
            logger.info(f"Cached {len(results)} pages in Redis for file {file_id}")

            process_time = (time.time() - start_time) * 1000

            logger.info(
                f"Layout analysis completed for {file_id} in {process_time:.2f}ms"
            )

            return {
                "status": "success",
                "file_id": file_id,
                "pages_processed": len(images),
                "layout_data": results,
                "process_time_ms": round(process_time, 2),
            }

        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(
                f"❌ Error during layout analysis for {file_id}:\n{error_msg}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def get_debug_image(self, filename: str):
        """
        Serve a generated debug image so the frontend can display it.

        Returns the image file if it exists, or 404 if not found.

        Leverages: FastAPI FileResponse.
        """
        file_path = os.path.join(self.DEBUG_DIR, filename)
        logger.info(f"Requested debug image: {filename}")
        logger.info(f"Looking for file at: {os.path.abspath(file_path)}")

        if os.path.exists(file_path):
            return FileResponse(file_path)

        logger.error(f"❌ Image not found at: {os.path.abspath(file_path)}")
        raise HTTPException(status_code=404, detail="Image not found")


_instance = LayoutAnalysisRouter()
router = _instance.router

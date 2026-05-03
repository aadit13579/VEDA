"""
VEDA Streaming Pipeline Router.

Orchestrates the full document-processing pipeline by calling the same
service functions used by the individual API routers. Handles document
upload, layout analysis, spatial sort, parallel region OCR/Gemini, and
SSE streaming of results.

This module exists to provide a unified, asynchronous pipeline that
streams page-by-page progress to the frontend, managing background
tasks, concurrency, and rollback on failure.

Leverages: FastAPI, asyncio, ThreadPoolExecutor, YOLO (layout_engine),
           XY-Cut (spatial_sort_engine), OCR (ocr_engine), Gemini
           (gemini_engine), and Redis (redis_client).
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
from typing import Any, Optional, AsyncGenerator

import cv2
import filetype
import fitz
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

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
from routers.ingest import _classify_pdf
from utils.logger import get_logger
from utils.text_cleaner import clean_extracted_text

logger = get_logger(__name__)


class PipelineRouter:
    """
    Router for the VEDA streaming document pipeline.

    Manages the thread pool, active SSE queues, and orchestrates the
    multi-stage document processing flow including rollback and cleanup.

    Leverages: FastAPI APIRouter, ThreadPoolExecutor, asyncio.Queue.
    """

    STORAGE_DIR = "storage"
    DEBUG_DIR = "storage/debug"
    ICEBERG_DIR = "storage/iceberg_storage"

    GEMINI_LABELS: set[str] = {
        "figure", "table", "image", "picture", "isolate_formula",
    }

    def __init__(self):
        """
        Initialize directories, thread pool, and router endpoints.

        Leverages: os.makedirs, ThreadPoolExecutor, APIRouter.
        """
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        os.makedirs(self.ICEBERG_DIR, exist_ok=True)

        _cpu_count = os.cpu_count() or 4
        self.max_workers = min(_cpu_count * 4, 32)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Pipeline thread pool initialised: MAX_WORKERS={self.max_workers}")

        self.active_pipelines: dict[str, asyncio.Queue] = {}

        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Register API routes for the pipeline.

        Leverages: APIRouter.add_api_route.
        """
        self.router.add_api_route("/pipeline/start", self.start_pipeline, methods=["POST"])
        self.router.add_api_route("/pipeline/stream/{file_id}", self.stream_pipeline, methods=["GET"])
        self.router.add_api_route("/pipeline/history", self.get_history, methods=["GET"])
        self.router.add_api_route("/pipeline/history/{file_id}", self.get_history_file, methods=["GET"])

    def parse_segments_str(self, segments_str: str, total_pages: int) -> list[int]:
        """
        Parse voice command segments into a deduplicated list of page numbers.

        Leverages: Python string splitting and int parsing.
        """
        pages: list[int] = []
        seen: set[int] = set()

        for part in segments_str.split(","):
            part = part.strip()
            if not part:
                continue

            if "-" in part:
                left, right = part.split("-", 1)
                try:
                    from_p = int(left.strip())
                except ValueError:
                    continue

                if right.strip().lower() == "end":
                    to_p = total_pages
                else:
                    try:
                        to_p = int(right.strip())
                    except ValueError:
                        continue

                for p in range(from_p, to_p + 1):
                    if 1 <= p <= total_pages and p not in seen:
                        pages.append(p)
                        seen.add(p)
            else:
                try:
                    p = int(part)
                except ValueError:
                    continue
                if 1 <= p <= total_pages and p not in seen:
                    pages.append(p)
                    seen.add(p)

        return pages

    def sse_event(self, event_type: str, data: dict) -> str:
        """
        Format a Server-Sent Event frame.

        Leverages: json.dumps.
        """
        payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event_type}\ndata: {payload}\n\n"

    def cleanup(self, file_id: str, file_path: str | None) -> None:
        """
        Remove all artefacts associated with a file_id on rollback.

        Leverages: os.remove, glob, delete_file_keys.
        """
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        for img_path in glob.glob(os.path.join(self.DEBUG_DIR, f"{file_id}_*")):
            os.remove(img_path)

        delete_file_keys(file_id)

        final_json = os.path.join(self.ICEBERG_DIR, f"{file_id}_final.json")
        if os.path.exists(final_json):
            os.remove(final_json)

    def load_page_image(self, file_path: str, page_index: int) -> np.ndarray:
        """
        Load a specific page as an OpenCV BGR image from PDF or image file.

        Leverages: cv2.imread, fitz.open.
        """
        img = cv2.imread(file_path)
        if img is not None:
            return img

        doc = fitz.open(file_path)
        if page_index < 0 or page_index >= len(doc):
            raise ValueError("Page index out of range")
        
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    async def process_gemini_region(
        self,
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
        Process a visual region using Gemini with OCR fallback.

        Leverages: asyncio.run_in_executor, gemini_describe_image.
        """
        bbox = region.get("bbox", [])
        
        try:
            result: dict = await loop.run_in_executor(
                self.executor,
                lambda: gemini_describe_image(file_id=file_id, page=page_num, bbox=bbox, top_k=8),
            )
            raw_resp = result.get("gemini_response", "")
            region["text"] = clean_extracted_text(raw_resp)
            region["gemini_caption"] = result.get("caption")
            region["gemini_context_text"] = result.get("context_text")
            region["gemini_response"] = clean_extracted_text(raw_resp)

            with lock:
                counters["gemini"] += 1

        except Exception as exc:
            region["error"] = str(exc)

            text = await loop.run_in_executor(
                self.executor,
                lambda: extract_text_from_pdf_region(file_path, page_num, bbox),
            )
            if not text.strip():
                text = await loop.run_in_executor(
                    self.executor,
                    lambda: extract_text_from_region(page_image, bbox),
                )
            if not text.strip():
                text = await loop.run_in_executor(
                    self.executor,
                    lambda: extract_text_with_gemini(page_image, bbox),
                )
            region["text"] = clean_extracted_text(text)

            with lock:
                counters["ocr"] += 1

    async def process_ocr_region(
        self,
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
        Process a text region using PyMuPDF, Tesseract, or Gemini fallback.

        Leverages: asyncio.run_in_executor, extract_text_from_pdf_region.
        """
        bbox = region.get("bbox", [])
        
        try:
            text: str = await loop.run_in_executor(
                self.executor,
                lambda: extract_text_from_pdf_region(file_path, page_num, bbox),
            )

            if text.strip():
                with lock:
                    counters["pymupdf"] += 1
            else:
                text = await loop.run_in_executor(
                    self.executor,
                    lambda: extract_text_from_region(page_image, bbox),
                )

                if text.strip():
                    with lock:
                        counters["ocr"] += 1
                else:
                    text = await loop.run_in_executor(
                        self.executor,
                        lambda: extract_text_with_gemini(page_image, bbox),
                    )
                    with lock:
                        counters["gemini"] += 1

            region["text"] = clean_extracted_text(text)

        except Exception as exc:
            region["text"] = ""
            region["error"] = str(exc)

    async def process_page(
        self,
        page_data: dict,
        file_id: str,
        file_path: str,
        counters: dict[str, int],
        lock: threading.Lock,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Process all regions on a page concurrently.

        Leverages: asyncio.gather.
        """
        page_num: int = page_data.get("page", 0)
        regions: list[dict] = page_data.get("regions", [])

        page_image: np.ndarray = await loop.run_in_executor(
            self.executor,
            lambda: self.load_page_image(file_path, page_num - 1),
        )

        tasks: list[asyncio.coroutine] = []
        for region in regions:
            bbox = region.get("bbox", [])
            if not bbox or len(bbox) != 4:
                continue

            label = region.get("label", "").lower().replace(" ", "_")

            if label in self.GEMINI_LABELS:
                tasks.append(
                    self.process_gemini_region(
                        region, file_id, page_num, file_path,
                        page_image, lock, counters, loop,
                    )
                )
            else:
                tasks.append(
                    self.process_ocr_region(
                        region, file_id, page_num, file_path,
                        page_image, lock, counters, loop,
                    )
                )

        await asyncio.gather(*tasks, return_exceptions=True)

        page_has_text = any(r.get("text", "").strip() for r in regions)
        if not page_has_text and file_path.lower().endswith(".pdf"):
            full_text: str = await loop.run_in_executor(
                self.executor,
                lambda: extract_full_page_text(file_path, page_num),
            )
            if full_text.strip():
                text_region = next(
                    (
                        r for r in regions
                        if r.get("label", "").lower().replace(" ", "_") not in self.GEMINI_LABELS
                    ),
                    None,
                )
                if text_region:
                    text_region["text"] = clean_extracted_text(full_text)
                else:
                    regions.append({
                        "label": "text",
                        "bbox": [0, 0, 100, 100],
                        "text": clean_extracted_text(full_text),
                        "confidence": 1.0,
                        "id": "r_fullpage",
                        "reading_order": 0,
                    })
                with lock:
                    counters["pymupdf"] += 1

        set_page(file_id, page_num, page_data)

    async def run_pipeline_background(
        self,
        file_id: str,
        original_filename: str,
        file_path: str,
        category: str,
        total_pages: int,
        images: list,
        queue: asyncio.Queue,
        page_queue: list[int],
    ) -> None:
        """
        Background task orchestrating layout, sorting, and region processing.

        Leverages: process_page, analyze_layout, process_spatial_sort.
        """
        pipeline_start = time.time()
        loop = asyncio.get_event_loop()
        counters: dict[str, int] = {"ocr": 0, "gemini": 0, "pymupdf": 0}
        lock = threading.Lock()
        pages_processed = 0

        try:
            for page_num in page_queue:
                page_idx = page_num - 1
                page_t0 = time.time()

                img = images[page_idx]
                regions = analyze_layout(img)

                output_filename = f"{file_id}_page_{page_num}.jpg"
                output_path = os.path.join(self.DEBUG_DIR, output_filename)
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
                set_page(file_id, page_num, page_result)

                sort_payload = {"layout_data": [page_result]}
                ordered = process_spatial_sort(sort_payload)
                sorted_page = ordered["layout_data"][0]
                set_page(file_id, page_num, sorted_page)

                await self.process_page(
                    page_data=sorted_page,
                    file_id=file_id,
                    file_path=file_path,
                    counters=counters,
                    lock=lock,
                    loop=loop,
                )

                pages_processed += 1
                page_elapsed = (time.time() - page_t0) * 1000

                from services.redis_client import get_page
                final_page_data = get_page(file_id, page_num) or sorted_page

                await queue.put({
                    "type": "page_ready",
                    "data": {
                        "page": page_num,
                        "total_pages": total_pages,
                        "pages_processed": pages_processed,
                        "page_data": final_page_data,
                        "page_time_ms": round(page_elapsed, 2),
                    },
                })

            set_total_pages(file_id, total_pages)
            final_pages = get_all_pages(file_id)
            if not final_pages:
                raise RuntimeError("No pages found in Redis for finalization.")

            final_document: dict[str, Any] = {
                "file_id": file_id,
                "original_filename": original_filename,
                "category": category,
                "total_pages": len(final_pages),
                "pages": final_pages,
            }

            output_path = os.path.join(self.ICEBERG_DIR, f"{file_id}_final.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_document, f, indent=2, ensure_ascii=False)

            delete_file_keys(file_id)
            total_time = (time.time() - pipeline_start) * 1000

            await queue.put({
                "type": "complete",
                "data": {
                    "file_id": file_id,
                    "total_pages": total_pages,
                    "pages_processed": pages_processed,
                    "output_path": output_path,
                    "total_time_ms": round(total_time, 2),
                    "counters": counters,
                },
            })

        except Exception as exc:
            self.cleanup(file_id, file_path)
            await queue.put({
                "type": "error",
                "data": {
                    "file_id": file_id,
                    "error": str(exc),
                    "pages_processed": pages_processed,
                },
            })

        finally:
            await queue.put(None)
            await asyncio.sleep(2)
            self.active_pipelines.pop(file_id, None)

    async def start_pipeline(
        self,
        file: UploadFile = File(...),
        start_page: int = Query(1, ge=1),
        end_page: Optional[int] = Query(None, ge=1),
        segments: Optional[str] = Query(None),
    ) -> dict:
        """
        Upload a file and launch the background processing task.

        Leverages: filetype.guess, pdf_to_images, asyncio.create_task.
        """
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(self.STORAGE_DIR, f"{file_id}{file_extension}")

        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        kind = filetype.guess(file_path)
        category = "UNKNOWN"
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

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        images = pdf_to_images(file_bytes)
        total_pages = len(images)

        if segments:
            page_queue = self.parse_segments_str(segments, total_pages)
            if not page_queue:
                self.cleanup(file_id, file_path)
                raise HTTPException(status_code=400, detail="Invalid segments")
        else:
            if start_page > total_pages:
                self.cleanup(file_id, file_path)
                raise HTTPException(status_code=400, detail="start_page exceeds total pages")
            effective_end = min(end_page, total_pages) if end_page else total_pages
            page_queue = list(range(start_page, effective_end + 1))

        queue: asyncio.Queue = asyncio.Queue()
        self.active_pipelines[file_id] = queue

        asyncio.create_task(
            self.run_pipeline_background(
                file_id=file_id,
                original_filename=file.filename,
                file_path=file_path,
                category=category,
                total_pages=total_pages,
                images=images,
                queue=queue,
                page_queue=page_queue,
            )
        )

        return {
            "status": "started",
            "file_id": file_id,
            "filename": file.filename,
            "category": category,
            "total_pages": total_pages,
            "page_queue": page_queue,
            "start_page": page_queue[0] if page_queue else start_page,
        }

    async def _event_generator(self, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """
        Generate Server-Sent Events from the pipeline queue.

        Leverages: asyncio.Queue.get, sse_event.
        """
        while True:
            event = await queue.get()
            if event is None:
                break
            yield self.sse_event(event["type"], event["data"])

    async def stream_pipeline(self, file_id: str):
        """
        Stream processing events to the client.

        Leverages: StreamingResponse, _event_generator.
        """
        queue = self.active_pipelines.get(file_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        return StreamingResponse(
            self._event_generator(queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def get_history(self):
        """
        List previously processed documents.

        Leverages: os.scandir, json.load.
        """
        history = []
        if os.path.exists(self.ICEBERG_DIR):
            for entry in os.scandir(self.ICEBERG_DIR):
                if entry.is_file() and entry.name.endswith("_final.json"):
                    try:
                        with open(entry.path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            history.append({
                                "file_id": data.get("file_id"),
                                "filename": data.get("original_filename", entry.name.replace("_final.json", "")),
                                "category": data.get("category", "UNKNOWN"),
                                "total_pages": data.get("total_pages", 0),
                                "timestamp": os.path.getmtime(entry.path)
                            })
                    except Exception:
                        pass
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)

    async def get_history_file(self, file_id: str):
        """
        Retrieve a specific document's processed JSON result.

        Leverages: fastapi.responses.FileResponse.
        """
        hist_path = os.path.join(self.ICEBERG_DIR, f"{file_id}_final.json")
        if os.path.exists(hist_path):
            from fastapi.responses import FileResponse
            return FileResponse(hist_path, media_type="application/json")
        raise HTTPException(status_code=404, detail="History not found")


_instance = PipelineRouter()
router = _instance.router

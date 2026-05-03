"""
VEDA Ingest Router.

Handles document upload, file-type detection via magic numbers, and
serving of uploaded documents. Classifies PDFs as digital or scanned
by sampling text content from the first three pages.

This module exists as the entry point for all documents entering the
VEDA pipeline — every file must pass through upload before layout
analysis, OCR, or Gemini processing can begin.

Leverages: FastAPI, filetype (magic number detection), PyMuPDF (fitz),
           shutil, uuid.
"""

import os
import shutil
import time
import uuid

import filetype
import fitz
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from utils.logger import get_logger

logger = get_logger(__name__)


class IngestRouter:
    """
    Router for document upload, classification, and serving.

    Provides endpoints to upload files (with automatic type detection),
    serve previously uploaded documents, and classify PDFs as digital
    or scanned. The classify_pdf method is also used by the pipeline
    router for inline classification.

    Leverages: FastAPI APIRouter, filetype, fitz.
    """

    UPLOAD_DIR = "storage"

    def __init__(self):
        """
        Initialize the router and ensure the upload directory exists.

        Leverages: os.makedirs, FastAPI APIRouter.
        """
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Bind endpoint methods to their HTTP routes.

        Separated from __init__ to keep route registration explicit
        and to allow subclasses to override individual routes.

        Leverages: FastAPI APIRouter.add_api_route.
        """
        self.router.add_api_route(
            "/document/{file_id}",
            self.get_document,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/upload",
            self.upload_and_identify,
            methods=["POST"],
        )

    async def get_document(self, file_id: str):
        """
        Return the original uploaded file for a given file_id.

        Searches the storage directory for any file whose stem matches
        the file_id and serves it with the correct MIME type. Used by
        the frontend to embed PDFs and images directly.

        Leverages: os.scandir, FastAPI FileResponse.
        """
        for entry in os.scandir(self.UPLOAD_DIR):
            if entry.is_file() and os.path.splitext(entry.name)[0] == file_id:
                mime = "application/octet-stream"
                ext = os.path.splitext(entry.name)[1].lower()
                if ext == ".pdf":
                    mime = "application/pdf"
                elif ext in (".png", ".jpg", ".jpeg"):
                    mime = f"image/{ext.lstrip('.')}"
                elif ext in (".tif", ".tiff"):
                    mime = "image/tiff"
                return FileResponse(
                    path=entry.path,
                    media_type=mime,
                    headers={"Accept-Ranges": "bytes"},
                )
        raise HTTPException(
            status_code=404,
            detail=f"No file found for file_id '{file_id}'.",
        )

    async def upload_and_identify(self, file: UploadFile = File(...)):
        """
        Upload a document, detect its type, and return classification metadata.

        Generates a UUID file_id, saves the file to disk, detects the MIME
        type via magic number inspection, and classifies the document into
        categories: IMAGE, PDF_DIGITAL, PDF_SCANNED, OFFICE_WORD, OFFICE_PPT,
        TEXT_FILE, or UNKNOWN.

        Leverages: uuid, shutil, filetype, classify_pdf.
        """
        logger.info(f"Received upload request for file: {file.filename}")
        start_time = time.time()

        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = f"{self.UPLOAD_DIR}/{file_id}{file_extension}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        kind = filetype.guess(file_path)

        category = "UNKNOWN"
        mime_type = "unknown/unknown"

        if kind:
            mime_type = kind.mime

            if mime_type.startswith("image/"):
                category = "IMAGE"
            elif mime_type == "application/pdf":
                category = self.classify_pdf(file_path)
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

        process_time = (time.time() - start_time) * 1000

        logger.info(
            f"File upload complete: {file.filename} -> {file_id}, category: {category}"
        )

        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename,
            "detected_mime": mime_type,
            "category": category,
            "process_time_ms": round(process_time, 2),
        }

    def classify_pdf(self, path: str) -> str:
        """
        Determine whether a PDF is text-based (digital) or image-based (scanned).

        Samples the first three pages and checks cumulative text length.
        More than 50 characters indicates a digital PDF; otherwise it is
        classified as scanned.

        This method is also called by the pipeline router for inline
        PDF classification during streaming uploads.

        Leverages: fitz.open, page.get_text.
        """
        try:
            doc = fitz.open(path)
            text_length = 0

            for i in range(min(3, len(doc))):
                text_length += len(doc[i].get_text())

            if text_length > 50:
                return "PDF_DIGITAL"
            else:
                return "PDF_SCANNED"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return "PDF_SCANNED"


_instance = IngestRouter()
router = _instance.router
classify_pdf = _instance.classify_pdf
_classify_pdf = _instance.classify_pdf
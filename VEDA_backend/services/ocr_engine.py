"""
VEDA OCR Engine.

Provides text extraction from document regions using three strategies:
  1. PyMuPDF native extraction (fast, digital PDFs only)
  2. Tesseract OCR (scanned documents and images)
  3. Full-page PyMuPDF fallback (last resort for blank regions)

This module exists to centralize all OCR logic behind a single interface,
allowing the pipeline and routers to call extraction methods without
knowing which engine is available or appropriate.

Leverages: PyMuPDF (fitz), pytesseract, OpenCV (cv2), python-dotenv.
"""

import os

import cv2
import fitz
import numpy as np
import pytesseract
from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """
    Unified OCR engine supporting PyMuPDF, Tesseract, and full-page fallback.

    Discovers the Tesseract executable at initialization time using the
    TESSERACT_PATH environment variable or the standard Windows install
    location. If Tesseract is unavailable, region-level OCR silently
    returns empty strings and callers fall through to other strategies.

    Leverages: fitz (PyMuPDF), pytesseract, cv2, dotenv.
    """

    def __init__(self):
        """
        Configure the Tesseract executable path.

        Checks TESSERACT_PATH from the .env file, then the standard
        Windows install location. Sets an internal availability flag
        that guards all Tesseract calls.

        Leverages: python-dotenv, os.path, pytesseract.
        """
        _env_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", ".env"
        )
        load_dotenv(dotenv_path=os.path.abspath(_env_path))

        tesseract_path = os.getenv("TESSERACT_PATH")
        self._tesseract_available = False

        if not tesseract_path:
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_path):
                tesseract_path = default_path
                logger.info(
                    f"TESSERACT_PATH not set. Found Tesseract at default location: {default_path}"
                )
            else:
                logger.warning(
                    "TESSERACT_PATH not set and Tesseract not found at default location. "
                    "Tesseract OCR disabled — using PyMuPDF text extraction for digital PDFs."
                )

        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self._tesseract_available = True
            logger.info(f"Tesseract OCR available at: {tesseract_path}")
        elif tesseract_path:
            logger.warning(
                f"TESSERACT_PATH set to '{tesseract_path}' but file does not exist."
            )

    def extract_text_from_pdf_region(
        self, file_path: str, page_num: int, bbox: list, scale: float = 2.0
    ) -> str:
        """
        Extract text from a PDF region using PyMuPDF native extraction.

        Works for digital (text-based) PDFs without Tesseract. The bbox
        is in image coordinates (rendered at the given scale), so it is
        converted back to PDF coordinates by dividing by the scale factor.

        Returns an empty string if the file is not a PDF or no text is found.

        Leverages: fitz.open, page.get_text with clip rect.
        """
        try:
            if not file_path.lower().endswith(".pdf"):
                return ""

            doc = fitz.open(file_path)
            page_index = page_num - 1

            if page_index < 0 or page_index >= len(doc):
                doc.close()
                return ""

            page = doc[page_index]

            x1, y1, x2, y2 = [coord / scale for coord in bbox]

            pad = 2.0
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = x2 + pad
            y2 = y2 + pad

            clip_rect = fitz.Rect(x1, y1, x2, y2)

            text = page.get_text("text", clip=clip_rect)

            if not text.strip():
                blocks = page.get_text("blocks", clip=clip_rect)
                text = "\n".join(
                    block[4] for block in blocks if block[6] == 0
                )

            doc.close()
            return text.strip()

        except Exception as e:
            logger.debug(f"PyMuPDF text extraction failed: {e}")
            return ""

    def extract_full_page_text(self, file_path: str, page_num: int) -> str:
        """
        Extract ALL text from a full PDF page using PyMuPDF.

        This is a last-resort fallback when both region-level PyMuPDF
        extraction and Tesseract fail.

        Returns the full page text, or an empty string on failure.

        Leverages: fitz.open, page.get_text.
        """
        try:
            if not file_path.lower().endswith(".pdf"):
                return ""

            doc = fitz.open(file_path)
            page_index = page_num - 1

            if page_index < 0 or page_index >= len(doc):
                doc.close()
                return ""

            page = doc[page_index]
            text = page.get_text("text")
            doc.close()

            return text.strip()

        except Exception as e:
            logger.debug(f"PyMuPDF full-page text extraction failed: {e}")
            return ""

    def extract_text_from_region(self, image: np.ndarray, bbox: list) -> str:
        """
        Crop a bounding box from an image and extract text using Tesseract.

        Returns an empty string if Tesseract is not available or the
        bounding box is invalid.

        Leverages: cv2 for cropping and grayscale conversion, pytesseract.
        """
        if not self._tesseract_available:
            logger.debug("Tesseract not available, returning empty string.")
            return ""

        try:
            x1, y1, x2, y2 = map(int, bbox)

            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box: {bbox}")
                return ""

            cropped_region = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

            config = "--psm 6"
            text = pytesseract.image_to_string(gray, config=config)

            return text.strip()

        except Exception as e:
            logger.error(f"Error during OCR extraction: {str(e)}")
            return ""


_instance = OCREngine()

extract_text_from_pdf_region = _instance.extract_text_from_pdf_region
extract_full_page_text = _instance.extract_full_page_text
extract_text_from_region = _instance.extract_text_from_region

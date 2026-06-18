import cv2
import numpy as np
import os
import fitz  # PyMuPDF
import pytesseract
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)

# Load .env so TESSERACT_PATH is available
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path))

# Set the Tesseract executable path for Windows
tesseract_path = os.getenv("TESSERACT_PATH")
_tesseract_available = False

if not tesseract_path:
    # Check standard Windows install location as fallback
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        tesseract_path = default_path
        logger.info(f"TESSERACT_PATH not set. Found Tesseract at default location: {default_path}")
    else:
        logger.warning(
            "TESSERACT_PATH not set and Tesseract not found at default location. "
            "Tesseract OCR disabled — using PyMuPDF text extraction for digital PDFs."
        )

if tesseract_path and os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    _tesseract_available = True
    logger.info(f"Tesseract OCR available at: {tesseract_path}")
elif tesseract_path:
    logger.warning(f"TESSERACT_PATH set to '{tesseract_path}' but file does not exist.")


# ---------- PyMuPDF Text Extraction (for digital PDFs) ----------

def extract_text_from_pdf_region(
    file_path: str, page_num: int, bbox: list, scale: float = 2.0
) -> str:
    """
    Extract text from a PDF region using PyMuPDF's native text extraction.

    This works for digital (text-based) PDFs without needing Tesseract.
    The bbox is in image coordinates (after rendering at the given scale),
    so we convert back to PDF coordinates by dividing by the scale factor.

    Args:
        file_path: Path to the PDF file on disk.
        page_num:  1-indexed page number.
        bbox:      [x1, y1, x2, y2] in image coordinates (2x scale).
        scale:     The zoom factor used when rendering pages to images.

    Returns:
        Extracted text, or empty string if not a PDF or no text found.
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

        # Convert image-space bbox back to PDF-space
        x1, y1, x2, y2 = [coord / scale for coord in bbox]

        # Add a small padding to catch text at region edges
        pad = 2.0
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = x2 + pad
        y2 = y2 + pad

        clip_rect = fitz.Rect(x1, y1, x2, y2)

        # Extract text within the clipped rectangle
        text = page.get_text("text", clip=clip_rect)

        if not text.strip():
            # Try with "blocks" method which can catch more text
            blocks = page.get_text("blocks", clip=clip_rect)
            text = "\n".join(
                block[4] for block in blocks if block[6] == 0  # type 0 = text
            )

        doc.close()

        return text.strip()

    except Exception as e:
        logger.debug(f"PyMuPDF text extraction failed: {e}")
        return ""


def extract_full_page_text(file_path: str, page_num: int) -> str:
    """
    Extract ALL text from a full PDF page using PyMuPDF.

    This is a last-resort fallback when both region-level PyMuPDF
    extraction and Tesseract fail. Returns the complete page text.

    Args:
        file_path: Path to the PDF file on disk.
        page_num:  1-indexed page number.

    Returns:
        The full page text, or empty string on failure.
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


# ---------- Tesseract OCR (for scanned PDFs / images) ----------

def extract_text_from_region(image: np.ndarray, bbox: list) -> str:
    """
    Crops the specified bounding box from the image and extracts text using Tesseract OCR.

    Args:
        image: The full OpenCV image array (BGR format).
        bbox: The bounding box [x1, y1, x2, y2].

    Returns:
        The extracted text as a string.
    """
    if not _tesseract_available:
        logger.debug("Tesseract not available, returning empty string.")
        return ""

    try:
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Check if the cropped region is valid
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bounding box: {bbox}")
            return ""

        # Crop the region
        cropped_region = image[y1:y2, x1:x2]

        # Optional: Pre-processing for better OCR accuracy
        # Convert to grayscale
        gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Run OCR
        # --psm 6 assumes a single uniform block of text
        config = "--psm 6"
        text = pytesseract.image_to_string(gray, config=config)

        return text.strip()

    except Exception as e:
        logger.error(f"Error during OCR extraction: {str(e)}")
        return ""

import cv2
import numpy as np
import os
import pytesseract
from utils.logger import get_logger

logger = get_logger(__name__)

# Set the Tesseract executable path for Windows
tesseract_path = os.getenv("TESSERACT_PATH")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    logger.warning("TESSERACT_PATH not set. Using system default.")


def extract_text_from_region(image: np.ndarray, bbox: list) -> str:
    """
    Crops the specified bounding box from the image and extracts text using Tesseract OCR.

    Args:
        image: The full OpenCV image array (BGR format).
        bbox: The bounding box [x1, y1, x2, y2].

    Returns:
        The extracted text as a string.
    """
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

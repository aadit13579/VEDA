import time
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any
from services.ocr_engine import extract_text_from_region
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/layout/ocr")
async def perform_ocr(payload: Dict[str, Any], image: UploadFile = File(...)):
    """
    Performs OCR on sorted layout regions using bounding boxes.
    """
    start_time = time.time()
    logger.info("Received OCR request.")

    # Read image
    image_bytes = await image.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    for page in payload.get("layout_data", []):
        for region in page.get("regions", []):
            bbox = region.get("bbox")

            if bbox:
                text = extract_text_from_region(img, bbox)
                region["text"] = text

    process_time = (time.time() - start_time) * 1000
    payload["process_time_ms"] = round(process_time, 2)

    logger.info("OCR extraction completed.")

    return payload

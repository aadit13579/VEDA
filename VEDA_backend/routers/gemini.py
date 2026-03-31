"""
Gemini Image Description Router.

Endpoint:
  POST /describe-image  — Crop an image region, gather context, send to Gemini
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from services.gemini_engine import describe_image
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class DescribeImageRequest(BaseModel):
    """Payload for the Gemini image description endpoint."""
    file_id: str
    page: int = Field(..., ge=1, description="1-indexed page number")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    top_k: Optional[int] = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of spatially-scored regions to include as context",
    )


@router.post("/describe-image")
async def describe_image_endpoint(request: DescribeImageRequest):
    """
    Describe an image region using Gemini.

    Pipeline:
      1. Loads the page from disk (PDF or image)
      2. Crops the bounding box and preprocesses to minimal size
      3. Gathers caption + surrounding text via spatial scoring (top-k)
      4. Sends image + context + hardcoded prompt to Gemini 2.0 Flash
      5. Returns the AI-generated description
    """
    start_time = time.time()
    logger.info(
        f"Describe-image request: file_id={request.file_id}, "
        f"page={request.page}, bbox={request.bbox}"
    )

    try:
        result = describe_image(
            file_id=request.file_id,
            page=request.page,
            bbox=request.bbox,
            top_k=request.top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Gemini describe-image failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {e}")

    process_time = (time.time() - start_time) * 1000
    logger.info(f"Describe-image completed in {process_time:.2f}ms")

    return {
        "status": "success",
        "file_id": request.file_id,
        "page": request.page,
        "bbox": request.bbox,
        "caption": result["caption"],
        "context_text": result["context_text"],
        "gemini_response": result["gemini_response"],
        "process_time_ms": round(process_time, 2),
    }

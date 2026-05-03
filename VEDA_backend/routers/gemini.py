"""
VEDA Gemini Image Description Router.

Exposes an HTTP endpoint to describe an image region using the Gemini
vision model. Crops the region, gathers spatial context, sends it to
Gemini, and returns the AI-generated description.

This module exists to provide a standalone API for image description
that the frontend or other tools can call independently of the full
processing pipeline.

Leverages: FastAPI, Pydantic, gemini_engine.
"""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from services.gemini_engine import describe_image
from utils.logger import get_logger

logger = get_logger(__name__)


class DescribeImageRequest(BaseModel):
    """
    Payload for the Gemini image description endpoint.

    Specifies the document, page, bounding box, and the number of
    spatially-scored context regions to include.

    Leverages: Pydantic BaseModel with Field validation.
    """

    file_id: str
    page: int = Field(..., ge=1, description="1-indexed page number")
    bbox: List[int] = Field(
        ..., min_length=4, max_length=4, description="[x1, y1, x2, y2]"
    )
    top_k: Optional[int] = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of spatially-scored regions to include as context",
    )


class GeminiRouter:
    """
    Router for Gemini-powered image description.

    Validates the request, delegates to the gemini_engine, handles
    errors, and formats the response.

    Leverages: FastAPI APIRouter, gemini_engine.describe_image.
    """

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
            "/describe-image",
            self.describe_image_endpoint,
            methods=["POST"],
        )

    async def describe_image_endpoint(self, request: DescribeImageRequest):
        """
        Describe an image region using Gemini.

        Pipeline: load page → crop bbox → gather spatial context →
        send to Gemini 2.5 Flash → return AI description.

        Leverages: gemini_engine.describe_image.
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
            raise HTTPException(
                status_code=500, detail=f"Gemini call failed: {e}"
            )

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


_instance = GeminiRouter()
router = _instance.router

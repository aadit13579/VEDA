"""
Spatial Sort Router.

Accepts EITHER a direct JSON payload OR a file_id (to fetch from Redis).
After sorting, writes each page back to Redis.
"""

import time
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from services.spatial_sort_engine import process_spatial_sort
from services.redis_client import set_page, get_all_pages
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SpatialSortRequest(BaseModel):
    """
    Either provide layout_data directly, or provide file_id to fetch from Redis.
    If both are provided, layout_data takes priority (avoids unnecessary Redis call).
    """
    file_id: Optional[str] = None
    layout_data: Optional[List[Dict[str, Any]]] = None


@router.post("/layout/sort")
async def sort_layout(request: SpatialSortRequest):
    """
    Endpoint to receive unordered DocLayout-YOLO region payload,
    filter 'abandon' elements, and output JSON ordered with 'reading_order'.

    Input priority:
      1. Direct layout_data payload (if provided)
      2. Fallback: fetch all pages from Redis using file_id
    """
    start_time = time.time()
    logger.info("Received layout sort request.")

    # --- Resolve input data ---
    if request.layout_data is not None:
        # Direct payload provided — use it as-is
        payload = {"layout_data": request.layout_data}
        file_id = request.file_id  # may be None
        logger.info("Using direct layout_data payload.")

    elif request.file_id is not None:
        # Fallback: fetch from Redis
        file_id = request.file_id
        pages = get_all_pages(file_id)

        if not pages:
            raise HTTPException(
                status_code=404,
                detail=f"No pages found in Redis for file_id '{file_id}'. "
                       f"Run layout analysis first.",
            )

        payload = {"layout_data": pages}
        logger.info(f"Fetched {len(pages)} pages from Redis for file {file_id}.")

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'layout_data' or 'file_id'.",
        )

    # --- Sort ---
    ordered_payload = process_spatial_sort(payload)

    # --- Write sorted pages back to Redis ---
    if file_id:
        for page_data in ordered_payload.get("layout_data", []):
            page_num = page_data.get("page")
            if page_num is not None:
                set_page(file_id, page_num, page_data)
        logger.info(f"Updated sorted pages in Redis for file {file_id}.")

    process_time = (time.time() - start_time) * 1000
    ordered_payload["process_time_ms"] = round(process_time, 2)

    logger.info("Layout sort completed successfully.")
    return ordered_payload

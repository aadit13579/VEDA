"""
VEDA Spatial Sort Router.

Accepts layout data (either as a direct JSON payload or by fetching
from Redis via file_id) and sorts all regions into human reading order
using the Recursive XY-Cut algorithm. Writes sorted results back to
Redis for downstream OCR and pipeline consumption.

This module exists to expose spatial sorting as a standalone API
endpoint, allowing the frontend or pipeline to trigger resorting
independently.

Leverages: FastAPI, spatial_sort_engine, Redis client.
"""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from services.spatial_sort_engine import process_spatial_sort
from services.redis_client import set_page, get_all_pages
from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialSortRequest(BaseModel):
    """
    Request payload for the spatial sort endpoint.

    Either provide layout_data directly, or provide file_id to fetch
    from Redis. If both are provided, layout_data takes priority.

    Leverages: Pydantic BaseModel.
    """

    file_id: Optional[str] = None
    layout_data: Optional[List[Dict[str, Any]]] = None


class SpatialSortRouter:
    """
    Router for spatial sorting of document layout regions.

    Resolves input from either a direct payload or Redis, delegates
    sorting to the spatial_sort_engine, and writes results back to Redis.

    Leverages: FastAPI APIRouter, process_spatial_sort, Redis client.
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
            "/layout/sort",
            self.sort_layout,
            methods=["POST"],
        )

    async def sort_layout(self, request: SpatialSortRequest):
        """
        Sort unordered layout regions into reading order.

        Resolves input data from the request payload (priority) or
        Redis fallback, runs the XY-Cut spatial sort, writes sorted
        pages back to Redis, and returns the ordered payload.

        Leverages: process_spatial_sort, get_all_pages, set_page.
        """
        start_time = time.time()
        logger.info("Received layout sort request.")

        if request.layout_data is not None:
            payload = {"layout_data": request.layout_data}
            file_id = request.file_id
            logger.info("Using direct layout_data payload.")

        elif request.file_id is not None:
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

        ordered_payload = process_spatial_sort(payload)

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


_instance = SpatialSortRouter()
router = _instance.router

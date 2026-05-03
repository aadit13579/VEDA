"""
VEDA Redis CRUD API Router.

Provides HTTP endpoints for direct Redis interaction: storing full page
JSON, updating nested fields, fetching pages or sub-fields, and
finalizing documents by writing combined JSON to disk and cleaning Redis.

This module exists so that the frontend and debugging tools can inspect
and manipulate cached pipeline state without going through the full
processing pipeline.

Leverages: FastAPI, Pydantic, Redis client, json.
"""

import os
import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Optional

from services.redis_client import (
    set_page,
    get_page,
    get_all_pages,
    resolve_field,
    update_field,
    delete_file_keys,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class StorePageRequest(BaseModel):
    """
    Payload for storing a full page JSON in Redis.

    Leverages: Pydantic BaseModel with Field validation.
    """

    file_id: str
    page: int
    data: dict
    ttl: int = Field(default=3600, ge=60, description="TTL in seconds (min 60)")


class UpdateFieldRequest(BaseModel):
    """
    Payload for updating a single nested field in a cached page.

    Leverages: Pydantic BaseModel with Field validation.
    """

    file_id: str
    page: int
    field: str = Field(
        ..., description="Dot/bracket path, e.g. 'regions[0].text'"
    )
    value: Any


class RedisAPIRouter:
    """
    Router for Redis page CRUD operations and document finalization.

    Provides store, update, fetch, and finalize endpoints that operate
    directly on the Redis cache. Document finalization assembles all
    cached pages into a single JSON file on disk.

    Leverages: FastAPI APIRouter, redis_client functions.
    """

    OUTPUT_DIR = "storage"

    def __init__(self):
        """
        Initialize the router, ensure the output directory exists, and
        register all routes.

        Leverages: os.makedirs, FastAPI APIRouter.
        """
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Bind endpoint methods to their HTTP routes.

        Leverages: FastAPI APIRouter.add_api_route.
        """
        self.router.add_api_route(
            "/redis/store-page",
            self.store_page,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/redis/update-field",
            self.update_field_endpoint,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/redis/get-field",
            self.get_field_endpoint,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/finalize/{file_id}",
            self.finalize_document,
            methods=["POST"],
        )

    async def store_page(self, request: StorePageRequest):
        """
        Store or overwrite a full page JSON in Redis.

        Uses key format file:{file_id}:page:{page} and applies the
        specified TTL.

        Leverages: redis_client.set_page.
        """
        logger.info(f"Storing page {request.page} for file {request.file_id}")

        set_page(request.file_id, request.page, request.data, ttl=request.ttl)

        return {
            "status": "success",
            "message": f"Page {request.page} stored for file {request.file_id}",
            "key": f"file:{request.file_id}:page:{request.page}",
        }

    async def update_field_endpoint(self, request: UpdateFieldRequest):
        """
        Update a nested field inside a cached page JSON.

        Supports dot notation and list indexing (e.g. 'regions[0].text').

        Leverages: redis_client.get_page, update_field, set_page.
        """
        logger.info(
            f"Updating field '{request.field}' on page {request.page} "
            f"for file {request.file_id}"
        )

        page_data = get_page(request.file_id, request.page)
        if page_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Page {request.page} not found in Redis for file {request.file_id}.",
            )

        try:
            updated = update_field(page_data, request.field, request.value)
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid field path '{request.field}': {exc}",
            )

        set_page(request.file_id, request.page, updated)

        return {
            "status": "success",
            "message": f"Field '{request.field}' updated on page {request.page}",
        }

    async def get_field_endpoint(
        self,
        file_id: str = Query(..., description="Document file ID"),
        page: int = Query(..., description="1-indexed page number"),
        field: Optional[str] = Query(
            None, description="Optional dot/bracket field path"
        ),
    ):
        """
        Fetch a full page JSON or a specific nested field from Redis.

        If field is omitted, the entire page dict is returned.

        Leverages: redis_client.get_page, resolve_field.
        """
        logger.info(
            f"Fetching {'field ' + repr(field) if field else 'full page'} "
            f"for page {page} of file {file_id}"
        )

        page_data = get_page(file_id, page)
        if page_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Page {page} not found in Redis for file {file_id}.",
            )

        if field:
            try:
                value = resolve_field(page_data, field)
            except (KeyError, IndexError, TypeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid field path '{field}': {exc}",
                )
            return {
                "status": "success",
                "file_id": file_id,
                "page": page,
                "field": field,
                "value": value,
            }

        return {
            "status": "success",
            "file_id": file_id,
            "page": page,
            "data": page_data,
        }

    async def finalize_document(self, file_id: str):
        """
        Finalize the document processing pipeline.

        Pulls all pages from Redis, builds a combined final JSON document,
        writes it to disk, and deletes all Redis keys for the file_id.

        Leverages: get_all_pages, json.dump, delete_file_keys.
        """
        logger.info(f"Finalize request for file {file_id}")

        pages = get_all_pages(file_id)
        if not pages:
            raise HTTPException(
                status_code=404,
                detail=f"No pages found in Redis for file_id '{file_id}'. "
                f"Nothing to finalize.",
            )

        final_document = {
            "file_id": file_id,
            "total_pages": len(pages),
            "pages": pages,
        }

        output_path = os.path.join(self.OUTPUT_DIR, f"{file_id}_final.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_document, f, indent=2, ensure_ascii=False)

        logger.info(f"Final JSON written to {os.path.abspath(output_path)}")

        deleted_count = delete_file_keys(file_id)

        return {
            "status": "success",
            "file_id": file_id,
            "total_pages": len(pages),
            "output_path": output_path,
            "redis_keys_deleted": deleted_count,
        }


_instance = RedisAPIRouter()
router = _instance.router

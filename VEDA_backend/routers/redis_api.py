"""
Redis CRUD API Router.

Endpoints:
  POST /redis/store-page    — Store a full page JSON in Redis
  POST /redis/update-field   — Update a nested field in a cached page
  GET  /redis/get-field      — Fetch a full page or a specific nested field
  POST /finalize/{file_id}   — Build final JSON, write to disk, clean Redis
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Optional
import json
import os
from services.redis_client import (
    set_page, get_page, get_all_pages,
    resolve_field, update_field, delete_file_keys,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------- Request Models ----------

class StorePageRequest(BaseModel):
    """Payload for storing a full page JSON in Redis."""
    file_id: str
    page: int
    data: dict
    ttl: int = Field(default=3600, ge=60, description="TTL in seconds (min 60)")


class UpdateFieldRequest(BaseModel):
    """Payload for updating a single nested field in a cached page."""
    file_id: str
    page: int
    field: str = Field(..., description="Dot/bracket path, e.g. 'regions[0].text'")
    value: Any


# ---------- Endpoints ----------

@router.post("/redis/store-page")
async def store_page(request: StorePageRequest):
    """
    Store (or overwrite) a full page JSON in Redis.

    Uses key format: file:{file_id}:page:{page}
    Applies the specified TTL.
    """
    logger.info(f"Storing page {request.page} for file {request.file_id}")

    set_page(request.file_id, request.page, request.data, ttl=request.ttl)

    return {
        "status": "success",
        "message": f"Page {request.page} stored for file {request.file_id}",
        "key": f"file:{request.file_id}:page:{request.page}",
    }


@router.post("/redis/update-field")
async def update_field_endpoint(request: UpdateFieldRequest):
    """
    Update a nested field inside a cached page JSON.

    Supports dot notation and list indexing:
      - "regions[0].text"
      - "meta.model"
    """
    logger.info(
        f"Updating field '{request.field}' on page {request.page} "
        f"for file {request.file_id}"
    )

    # 1. Fetch existing page
    page_data = get_page(request.file_id, request.page)
    if page_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Page {request.page} not found in Redis for file {request.file_id}.",
        )

    # 2. Update the field
    try:
        updated = update_field(page_data, request.field, request.value)
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid field path '{request.field}': {exc}",
        )

    # 3. Save back to Redis
    set_page(request.file_id, request.page, updated)

    return {
        "status": "success",
        "message": f"Field '{request.field}' updated on page {request.page}",
    }


@router.get("/redis/get-field")
async def get_field_endpoint(
    file_id: str = Query(..., description="Document file ID"),
    page: int = Query(..., description="1-indexed page number"),
    field: Optional[str] = Query(None, description="Optional dot/bracket field path"),
):
    """
    Fetch a full page JSON or a specific nested field from Redis.

    If *field* is omitted the entire page dict is returned.
    """
    logger.info(
        f"Fetching {'field ' + repr(field) if field else 'full page'} "
        f"for page {page} of file {file_id}"
    )

    # 1. Fetch page
    page_data = get_page(file_id, page)
    if page_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page} not found in Redis for file {file_id}.",
        )

    # 2. Resolve field (if provided)
    if field:
        try:
            value = resolve_field(page_data, field)
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid field path '{field}': {exc}",
            )
        return {"status": "success", "file_id": file_id, "page": page, "field": field, "value": value}

    # 3. Return full page
    return {"status": "success", "file_id": file_id, "page": page, "data": page_data}


# ---------- Finalize ----------

OUTPUT_DIR = "storage"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/finalize/{file_id}")
async def finalize_document(file_id: str):
    """
    Finalize the document processing pipeline.

    1. Pull all pages from Redis for the given file_id
    2. Build a combined final JSON document
    3. Write the JSON to disk at storage/{file_id}_final.json
    4. Delete all Redis keys for this file_id
    """
    logger.info(f"Finalize request for file {file_id}")

    # 1. Pull all pages
    pages = get_all_pages(file_id)
    if not pages:
        raise HTTPException(
            status_code=404,
            detail=f"No pages found in Redis for file_id '{file_id}'. "
                   f"Nothing to finalize.",
        )

    # 2. Build final JSON
    final_document = {
        "file_id": file_id,
        "total_pages": len(pages),
        "pages": pages,
    }

    # 3. Write to disk
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_final.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_document, f, indent=2, ensure_ascii=False)

    logger.info(f"Final JSON written to {os.path.abspath(output_path)}")

    # 4. Clean up Redis
    deleted_count = delete_file_keys(file_id)

    return {
        "status": "success",
        "file_id": file_id,
        "total_pages": len(pages),
        "output_path": output_path,
        "redis_keys_deleted": deleted_count,
    }

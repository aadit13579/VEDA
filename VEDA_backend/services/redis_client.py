"""
Redis Utility Layer for VEDA Backend.

Key format:
  - Page data:    file:{file_id}:page:{page}
  - Page index:   file:{file_id}:pages        (SET of page numbers)
  - Total pages:  file:{file_id}:total_pages

All values are JSON-serialized dicts. Binary data (images) must NOT be stored.
"""

import re
import json
import os
import redis
from typing import Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------- Connection ----------

_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def get_redis_client() -> redis.Redis:
    """Return the shared Redis client instance."""
    return _client


# ---------- Page-Level Helpers ----------

def _page_key(file_id: str, page: int) -> str:
    """Build the Redis key for a specific page."""
    return f"file:{file_id}:page:{page}"


def _pages_index_key(file_id: str) -> str:
    """Build the Redis key for the page-index set."""
    return f"file:{file_id}:pages"


def _total_pages_key(file_id: str) -> str:
    """Build the Redis key for total page count."""
    return f"file:{file_id}:total_pages"


def set_page(file_id: str, page: int, data: dict, ttl: int = 3600) -> None:
    """
    Store a page's JSON data in Redis.

    Also maintains:
      - A SET of known page numbers  (file:{file_id}:pages)
      - A total-pages counter        (file:{file_id}:total_pages)

    Args:
        file_id: Unique document identifier.
        page:    1-indexed page number.
        data:    Dict payload (must be JSON-serializable).
        ttl:     Time-to-live in seconds (default 1 hour).
    """
    key = _page_key(file_id, page)
    _client.setex(key, ttl, json.dumps(data))

    # Update page index set and refresh its TTL
    idx_key = _pages_index_key(file_id)
    _client.sadd(idx_key, str(page))
    _client.expire(idx_key, ttl)

    # Keep total_pages TTL in sync with page data
    tp_key = _total_pages_key(file_id)
    if _client.exists(tp_key):
        _client.expire(tp_key, ttl)

    logger.debug(f"Redis SET  {key}  (TTL={ttl}s)")


def set_total_pages(file_id: str, total: int, ttl: int = 3600) -> None:
    """Store the total page count for a file."""
    key = _total_pages_key(file_id)
    _client.setex(key, ttl, str(total))
    logger.debug(f"Redis SET  {key} = {total}")


def get_page(file_id: str, page: int) -> Optional[dict]:
    """
    Fetch a page's JSON data from Redis.

    Returns None if the key does not exist.
    """
    key = _page_key(file_id, page)
    raw = _client.get(key)
    if raw is None:
        logger.debug(f"Redis MISS {key}")
        return None
    logger.debug(f"Redis HIT  {key}")
    return json.loads(raw)


def get_all_pages(file_id: str) -> list[dict]:
    """
    Retrieve all cached pages for a file using the page-index set.

    Returns a list of page dicts sorted by page number.
    Missing pages (expired between index write and read) are skipped.
    """
    idx_key = _pages_index_key(file_id)
    page_numbers = _client.smembers(idx_key)

    if not page_numbers:
        logger.debug(f"Redis: no page index found for file {file_id}")
        return []

    pages = []
    for pn in sorted(page_numbers, key=lambda x: int(x)):
        data = get_page(file_id, int(pn))
        if data is not None:
            pages.append(data)

    # Final safety sort by the "page" field inside each dict
    pages = sorted(pages, key=lambda x: int(x.get("page", 0)))

    logger.debug(f"Redis: fetched {len(pages)} pages for file {file_id}")
    return pages


def get_total_pages(file_id: str) -> Optional[int]:
    """Return the cached total-pages count, or None."""
    raw = _client.get(_total_pages_key(file_id))
    return int(raw) if raw is not None else None


# ---------- Field Resolution Helpers ----------

# Matches tokens like  "regions"  or  "regions[0]"
_TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def _parse_field_path(field: str) -> list:
    """
    Parse a dot/bracket field path into a list of keys/indices.

    Examples:
        "regions[0].text"  -> ["regions", 0, "text"]
        "meta.model"       -> ["meta", "model"]
        "regions[2].bbox"  -> ["regions", 2, "bbox"]
    """
    tokens: list = []
    for match in _TOKEN_RE.finditer(field):
        name, index = match.groups()
        if name is not None:
            tokens.append(name)
        else:
            tokens.append(int(index))
    return tokens


def resolve_field(data: Any, field_path: str) -> Any:
    """
    Traverse *data* using a dot/bracket path and return the value.

    Raises KeyError or IndexError if the path is invalid.
    """
    current = data
    for token in _parse_field_path(field_path):
        if isinstance(token, int):
            current = current[token]
        else:
            current = current[token]
    return current


def update_field(data: Any, field_path: str, value: Any) -> Any:
    """
    Traverse *data* and set the leaf to *value* (in-place).

    Returns the mutated *data* for convenience.
    Raises KeyError or IndexError if the path is invalid.
    """
    tokens = _parse_field_path(field_path)
    current = data
    for token in tokens[:-1]:
        if isinstance(token, int):
            current = current[token]
        else:
            current = current[token]

    last = tokens[-1]
    if isinstance(last, int):
        current[last] = value
    else:
        current[last] = value

    return data


# ---------- Bbox Tolerance ----------

def bbox_matches(a: List[int], b: List[int], tolerance: int = 5) -> bool:
    """
    Check if two bounding boxes match within a pixel tolerance.

    Useful because coordinates may shift slightly between
    layout analysis, spatial sort, and OCR stages.
    """
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(ai - bi) <= tolerance for ai, bi in zip(a, b))


# ---------- Cleanup ----------

def delete_file_keys(file_id: str) -> int:
    """
    Delete ALL Redis keys associated with a file_id.

    Removes:
      - Every  file:{file_id}:page:{N}  key listed in the index set
      - The    file:{file_id}:pages      index set itself
      - The    file:{file_id}:total_pages counter

    Returns the total number of keys deleted.
    """
    deleted = 0

    # Delete individual page keys using the index
    idx_key = _pages_index_key(file_id)
    page_numbers = _client.smembers(idx_key)
    for pn in page_numbers:
        deleted += _client.delete(_page_key(file_id, int(pn)))

    # Delete the index set and total_pages counter
    deleted += _client.delete(idx_key)
    deleted += _client.delete(_total_pages_key(file_id))

    logger.info(f"Redis: deleted {deleted} keys for file {file_id}")
    return deleted


# ---------- Legacy Helpers (backward compat) ----------

def save_context(session_id: str, paragraph_text: str) -> None:
    """Store text with a 10-minute expiry (legacy)."""
    _client.setex(f"context:{session_id}", 600, paragraph_text)


def get_context(session_id: str) -> Optional[str]:
    """Retrieve stored context text (legacy)."""
    return _client.get(f"context:{session_id}")
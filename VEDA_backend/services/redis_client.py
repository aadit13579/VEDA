"""
VEDA Redis Utility Layer.

Provides all Redis interactions for the VEDA backend: page-level CRUD,
field path resolution and mutation, bounding-box tolerance matching,
key cleanup, and legacy context storage.

This module exists as the single source of truth for Redis key formats,
serialization conventions, and TTL policies — preventing every router
and service from reimplementing its own Redis logic.

Key format:
  - Page data:    file:{file_id}:page:{page}
  - Page index:   file:{file_id}:pages        (SET of page numbers)
  - Total pages:  file:{file_id}:total_pages

All values are JSON-serialized dicts. Binary data must NOT be stored.

Leverages: redis-py, json, re.
"""

import re
import json
import redis
from typing import Any, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class FieldPathParser:
    """
    Parses and traverses dot/bracket field paths into nested data structures.

    Supports paths like 'regions[0].text' or 'meta.model' for reading and
    writing values deep inside cached page dictionaries.

    This class exists to isolate the path-parsing logic from Redis I/O,
    making it testable and reusable independently.

    Leverages: re module for tokenization.
    """

    _TOKEN_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")

    @staticmethod
    def parse(field: str) -> list:
        """
        Parse a dot/bracket field path into a list of keys and indices.

        Examples:
            'regions[0].text'  -> ['regions', 0, 'text']
            'meta.model'       -> ['meta', 'model']

        Leverages: re.finditer for token extraction.
        """
        tokens: list = []
        for match in FieldPathParser._TOKEN_RE.finditer(field):
            name, index = match.groups()
            if name is not None:
                tokens.append(name)
            else:
                tokens.append(int(index))
        return tokens

    @staticmethod
    def resolve(data: Any, field_path: str) -> Any:
        """
        Traverse data using a dot/bracket path and return the leaf value.

        Raises KeyError or IndexError if the path is invalid.

        Leverages: FieldPathParser.parse for tokenization.
        """
        current = data
        for token in FieldPathParser.parse(field_path):
            current = current[token]
        return current

    @staticmethod
    def update(data: Any, field_path: str, value: Any) -> Any:
        """
        Traverse data and set the leaf node to the given value (in-place).

        Returns the mutated data for convenience.
        Raises KeyError or IndexError if the path is invalid.

        Leverages: FieldPathParser.parse for tokenization.
        """
        tokens = FieldPathParser.parse(field_path)
        current = data
        for token in tokens[:-1]:
            current = current[token]
        current[tokens[-1]] = value
        return data


class RedisClient:
    """
    Redis client wrapper for VEDA document page storage.

    Manages the lifecycle of cached page data: creation, retrieval,
    field updates, total-page tracking, and cleanup. Also provides
    bounding-box tolerance matching and legacy context storage.

    This class exists as the single point of contact with Redis,
    encapsulating key naming, serialization, and TTL management.

    Leverages: redis-py for all I/O, json for serialization.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize the Redis connection.

        Accepts host, port, and db parameters to allow future
        configurability and testing with alternate Redis instances.

        Leverages: redis.Redis.
        """
        self._client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )

    def get_redis_client(self) -> redis.Redis:
        """
        Return the underlying redis-py client instance.

        Useful for callers that need direct Redis access beyond the
        helper methods provided by this class.

        Leverages: redis.Redis.
        """
        return self._client

    def _page_key(self, file_id: str, page: int) -> str:
        """
        Build the Redis key for a specific page.

        Follows the convention file:{file_id}:page:{page}.

        Leverages: Python string formatting.
        """
        return f"file:{file_id}:page:{page}"

    def _pages_index_key(self, file_id: str) -> str:
        """
        Build the Redis key for the page-index set.

        Follows the convention file:{file_id}:pages.

        Leverages: Python string formatting.
        """
        return f"file:{file_id}:pages"

    def _total_pages_key(self, file_id: str) -> str:
        """
        Build the Redis key for the total page count.

        Follows the convention file:{file_id}:total_pages.

        Leverages: Python string formatting.
        """
        return f"file:{file_id}:total_pages"

    def set_page(self, file_id: str, page: int, data: dict, ttl: int = 3600) -> None:
        """
        Store a page's JSON data in Redis with a TTL.

        Also maintains a SET of known page numbers and keeps the
        total-pages counter TTL synchronized.

        Leverages: redis SETEX, SADD, EXPIRE.
        """
        key = self._page_key(file_id, page)
        self._client.setex(key, ttl, json.dumps(data))

        idx_key = self._pages_index_key(file_id)
        self._client.sadd(idx_key, str(page))
        self._client.expire(idx_key, ttl)

        tp_key = self._total_pages_key(file_id)
        if self._client.exists(tp_key):
            self._client.expire(tp_key, ttl)

        logger.debug(f"Redis SET  {key}  (TTL={ttl}s)")

    def set_total_pages(self, file_id: str, total: int, ttl: int = 3600) -> None:
        """
        Store the total page count for a file.

        Leverages: redis SETEX.
        """
        key = self._total_pages_key(file_id)
        self._client.setex(key, ttl, str(total))
        logger.debug(f"Redis SET  {key} = {total}")

    def get_page(self, file_id: str, page: int) -> Optional[dict]:
        """
        Fetch a page's JSON data from Redis.

        Returns None if the key does not exist.

        Leverages: redis GET, json.loads.
        """
        key = self._page_key(file_id, page)
        raw = self._client.get(key)
        if raw is None:
            logger.debug(f"Redis MISS {key}")
            return None
        logger.debug(f"Redis HIT  {key}")
        return json.loads(raw)

    def get_all_pages(self, file_id: str) -> list[dict]:
        """
        Retrieve all cached pages for a file using the page-index set.

        Returns a list of page dicts sorted by page number. Missing
        pages (expired between index write and read) are skipped.

        Leverages: redis SMEMBERS, json.loads.
        """
        idx_key = self._pages_index_key(file_id)
        page_numbers = self._client.smembers(idx_key)

        if not page_numbers:
            logger.debug(f"Redis: no page index found for file {file_id}")
            return []

        pages = []
        for pn in sorted(page_numbers, key=lambda x: int(x)):
            data = self.get_page(file_id, int(pn))
            if data is not None:
                pages.append(data)

        pages = sorted(pages, key=lambda x: int(x.get("page", 0)))

        logger.debug(f"Redis: fetched {len(pages)} pages for file {file_id}")
        return pages

    def get_total_pages(self, file_id: str) -> Optional[int]:
        """
        Return the cached total-pages count, or None.

        Leverages: redis GET.
        """
        raw = self._client.get(self._total_pages_key(file_id))
        return int(raw) if raw is not None else None

    def bbox_matches(self, a: List[int], b: List[int], tolerance: int = 5) -> bool:
        """
        Check if two bounding boxes match within a pixel tolerance.

        Useful because coordinates may shift slightly between layout
        analysis, spatial sort, and OCR stages.

        Leverages: Python built-in abs/zip.
        """
        if len(a) != 4 or len(b) != 4:
            return False
        return all(abs(ai - bi) <= tolerance for ai, bi in zip(a, b))

    def delete_file_keys(self, file_id: str) -> int:
        """
        Delete ALL Redis keys associated with a file_id.

        Removes every page key listed in the index set, the index set
        itself, and the total_pages counter.

        Returns the total number of keys deleted.

        Leverages: redis DELETE, SMEMBERS.
        """
        deleted = 0

        idx_key = self._pages_index_key(file_id)
        page_numbers = self._client.smembers(idx_key)
        for pn in page_numbers:
            deleted += self._client.delete(self._page_key(file_id, int(pn)))

        deleted += self._client.delete(idx_key)
        deleted += self._client.delete(self._total_pages_key(file_id))

        logger.info(f"Redis: deleted {deleted} keys for file {file_id}")
        return deleted

    def save_context(self, session_id: str, paragraph_text: str) -> None:
        """
        Store text with a 10-minute expiry (legacy helper).

        Retained for backward compatibility with older callers.

        Leverages: redis SETEX.
        """
        self._client.setex(f"context:{session_id}", 600, paragraph_text)

    def get_context(self, session_id: str) -> Optional[str]:
        """
        Retrieve stored context text (legacy helper).

        Retained for backward compatibility with older callers.

        Leverages: redis GET.
        """
        return self._client.get(f"context:{session_id}")


_instance = RedisClient()

get_redis_client = _instance.get_redis_client
set_page = _instance.set_page
set_total_pages = _instance.set_total_pages
get_page = _instance.get_page
get_all_pages = _instance.get_all_pages
get_total_pages = _instance.get_total_pages
bbox_matches = _instance.bbox_matches
delete_file_keys = _instance.delete_file_keys
save_context = _instance.save_context
get_context = _instance.get_context
resolve_field = FieldPathParser.resolve
update_field = FieldPathParser.update
"""
VEDA Voice Command Parser Router.

Validates and parses voice transcripts into pipeline-consumable page
segments with optional section filters. Enforces an English-only
whitelist, requires a 'page' keyword, parses number words, handles
range syntax, and produces the segments string used by the pipeline
start endpoint.

This module exists to bridge voice input (raw transcripts) and the
pipeline's structured page-range parameters, enabling hands-free
document navigation.

Leverages: FastAPI, Pydantic, re (regex).
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel


WORD_TO_NUM: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

_NAV_WORDS: set[str] = {
    "page", "pages", "section", "sections",
    "to", "from", "then", "and", "after", "that",
    "go", "read", "play", "start", "begin", "end",
    "also", "next", "followed", "by", "through",
}
ALLOWED_TOKENS: set[str] = _NAV_WORDS | set(WORD_TO_NUM.keys())


class VoiceParseRequest(BaseModel):
    """
    Request payload containing the raw voice transcript.

    Leverages: Pydantic BaseModel.
    """

    transcript: str


class VoiceSegment(BaseModel):
    """
    A parsed page/section range from a voice command.

    from_page and to_page are 1-indexed; to_page of -1 means 'end'.
    Section fields are optional and only populated when the user
    specifies section constraints.

    Leverages: Pydantic BaseModel.
    """

    from_page: int
    to_page: int
    from_section: Optional[int] = None
    to_section: Optional[int] = None


class VoiceParseResponse(BaseModel):
    """
    Response from the voice command parser.

    Contains validation status, parsed segments, the pipeline_segments
    string for POST /pipeline/start, and an optional section filter map.

    Leverages: Pydantic BaseModel.
    """

    valid: bool
    reject_reason: Optional[str] = None
    segments: Optional[list[VoiceSegment]] = None
    pipeline_segments: Optional[str] = None
    section_filter: Optional[dict] = None


class VoiceCommandParser:
    """
    Parses voice transcripts into structured pipeline segments.

    Normalizes input, validates against an English-only whitelist,
    splits on conjunctions, parses page/section ranges with compound
    number-word support, and builds the pipeline_segments and
    section_filter outputs.

    This class exists to encapsulate all voice command parsing logic
    in one testable unit, separate from HTTP concerns.

    Leverages: re module, WORD_TO_NUM mapping.
    """

    _FILLER = {"go", "read", "play", "start", "begin", "from"}

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
            "/voice/parse",
            self.parse_voice_command,
            methods=["POST"],
            response_model=VoiceParseResponse,
        )

    def _normalize(self, text: str) -> str:
        """
        Lowercase, strip punctuation except hyphens, and collapse whitespace.

        Prepares raw transcript text for tokenization and parsing.

        Leverages: re.sub.
        """
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_english_only(self, tokens: list[str]) -> tuple[bool, str]:
        """
        Validate that all tokens are in the allowed whitelist.

        Pure digit strings are always allowed. Returns (ok, reject_reason).

        Leverages: ALLOWED_TOKENS set.
        """
        for tok in tokens:
            if tok.isdigit():
                continue
            if tok in ALLOWED_TOKENS:
                continue
            return False, f"Non-English or unrecognised token: '{tok}'"
        return True, ""

    def _next_num(self, tokens: list[str], i: int) -> tuple[int | None, int]:
        """
        Parse one integer from tokens starting at index i.

        Handles digit strings, the 'end' sentinel (-1), and compound
        number words like 'twenty one'. Returns (value, next_index).

        Leverages: WORD_TO_NUM mapping.
        """
        if i >= len(tokens):
            return None, i

        tok = tokens[i]

        if tok.isdigit():
            return int(tok), i + 1

        if tok == "end":
            return -1, i + 1

        if tok in WORD_TO_NUM:
            val = WORD_TO_NUM[tok]
            if val >= 20 and (i + 1) < len(tokens):
                nxt = tokens[i + 1]
                if nxt in WORD_TO_NUM and 0 < WORD_TO_NUM[nxt] < 10:
                    return val + WORD_TO_NUM[nxt], i + 2
            return val, i + 1

        return None, i

    def _split_into_raw_segments(self, text: str) -> list[str]:
        """
        Split a normalized utterance on conjunctions into per-range pieces.

        Recognizes 'then', 'and then', 'after that', 'followed by', 'also',
        and comma-separated 'page' keywords as segment boundaries.

        Leverages: str.replace, re.sub.
        """
        for sep in [" then ", " and then ", " after that ", " followed by ", " also "]:
            text = text.replace(sep, " | ")
        text = re.sub(r",\s*(?=page\b)", " | ", text)
        return [s.strip() for s in text.split("|") if s.strip()]

    def _parse_one_segment(self, seg_text: str) -> VoiceSegment | str:
        """
        Parse a single segment string into a VoiceSegment.

        Handles 'page N [to M]' and optional 'section S [to T]' syntax.
        Returns an error string on failure.

        Leverages: _next_num, VoiceSegment.
        """
        tokens = seg_text.split()
        i = 0

        while i < len(tokens) and tokens[i] in self._FILLER:
            i += 1

        from_page: int | None = None
        to_page: int | None = None
        from_section: int | None = None
        to_section: int | None = None

        if i < len(tokens) and tokens[i] in {"page", "pages"}:
            i += 1
            from_page, i = self._next_num(tokens, i)
            if from_page is None:
                return f"Expected a page number after 'page' in: '{seg_text}'"
            to_page = from_page

            if i < len(tokens) and tokens[i] in {"to", "through"}:
                i += 1
                to_page, i = self._next_num(tokens, i)
                if to_page is None:
                    return f"Expected a number after 'to' in: '{seg_text}'"

        else:
            from_page, i = self._next_num(tokens, i)
            if from_page is None:
                return f"Expected 'page' keyword or a number in segment: '{seg_text}'"
            to_page = from_page

        if i < len(tokens) and tokens[i] in {"section", "sections"}:
            i += 1
            from_section, i = self._next_num(tokens, i)
            if from_section is None:
                return f"Expected a section number in: '{seg_text}'"
            to_section = from_section

            if i < len(tokens) and tokens[i] in {"to", "through"}:
                i += 1
                to_section, i = self._next_num(tokens, i)
                if to_section is None:
                    return f"Expected a section end number in: '{seg_text}'"

        return VoiceSegment(
            from_page=from_page,
            to_page=to_page,
            from_section=from_section,
            to_section=to_section,
        )

    def _build_pipeline_segments(self, segments: list[VoiceSegment]) -> str:
        """
        Build the pipeline_segments string for POST /pipeline/start.

        Converts parsed VoiceSegments into comma-separated range notation
        (e.g. '5-7,1-3' or '5-end' or '3').

        Leverages: str.join.
        """
        parts: list[str] = []
        for s in segments:
            if s.from_page == s.to_page and s.to_page != -1:
                parts.append(str(s.from_page))
            elif s.to_page == -1:
                parts.append(f"{s.from_page}-end")
            else:
                parts.append(f"{s.from_page}-{s.to_page}")
        return ",".join(parts)

    def _build_section_filter(self, segments: list[VoiceSegment]) -> dict:
        """
        Build a section filter map: page_number -> [from_sec, to_sec].

        Only includes pages that have explicit section constraints.
        Pages with to_page == -1 ('end') only store the from_page entry;
        the backend resolves the range at runtime.

        Leverages: Python dict comprehension.
        """
        sf: dict[str, list[int]] = {}
        for s in segments:
            if s.from_section is None:
                continue
            if s.to_page != -1:
                for p in range(s.from_page, s.to_page + 1):
                    sf[str(p)] = [s.from_section, s.to_section]
            else:
                sf[str(s.from_page)] = [s.from_section, s.to_section]
        return sf

    def parse_voice_command(self, req: VoiceParseRequest) -> VoiceParseResponse:
        """
        Validate and parse a voice transcript into pipeline segments.

        Returns valid=false with a human-readable reject_reason if the
        transcript is empty, contains non-English tokens, lacks a 'page'
        keyword, or has unparseable grammar.

        Leverages: _normalize, _is_english_only, _split_into_raw_segments,
                   _parse_one_segment, _build_pipeline_segments, _build_section_filter.
        """
        text = self._normalize(req.transcript)

        if not text:
            return VoiceParseResponse(valid=False, reject_reason="Empty transcript.")

        tokens = text.split()

        ok, reason = self._is_english_only(tokens)
        if not ok:
            return VoiceParseResponse(valid=False, reject_reason=reason)

        if "page" not in tokens and "pages" not in tokens:
            return VoiceParseResponse(
                valid=False,
                reject_reason=(
                    "No 'page' keyword found. "
                    "Please say something like 'page 5' or 'page 3 to 7'."
                ),
            )

        raw_segments = self._split_into_raw_segments(text)
        segments: list[VoiceSegment] = []

        for raw in raw_segments:
            result = self._parse_one_segment(raw)
            if isinstance(result, str):
                return VoiceParseResponse(valid=False, reject_reason=result)
            segments.append(result)

        pipeline_segments = self._build_pipeline_segments(segments)
        section_filter = self._build_section_filter(segments) or None

        return VoiceParseResponse(
            valid=True,
            segments=segments,
            pipeline_segments=pipeline_segments,
            section_filter=section_filter,
        )


_instance = VoiceCommandParser()
router = _instance.router

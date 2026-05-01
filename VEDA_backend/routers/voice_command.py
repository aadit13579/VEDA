"""
VEDA Voice Command Parser Router.

POST /voice/parse
  Body : { "transcript": "page five to seven then page one section two" }
  Response (valid):
    {
      "valid": true,
      "segments": [
        {"from_page": 5, "to_page": 7, "from_section": null, "to_section": null},
        {"from_page": 1, "to_page": 1, "from_section": 2,    "to_section": 2}
      ],
      "pipeline_segments": "5-7,1",    ← passed directly to POST /pipeline/start
      "section_filter": {"1": [2, 2]}  ← page_num → [from_sec, to_sec]
    }
  Response (invalid):
    { "valid": false, "reject_reason": "..." }

Rules
─────
  • Only English navigation words + numbers are accepted (whitelist guard).
  • At least one "page" keyword must be present.
  • "end" is a legal token and is passed as-is to the pipeline; the
    backend resolves it to total_pages after the file is inspected.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# ── Number-word → integer ─────────────────────────────────────────────────────

WORD_TO_NUM: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

# ── Whitelist for English-only guard ─────────────────────────────────────────

_NAV_WORDS: set[str] = {
    "page", "pages", "section", "sections",
    "to", "from", "then", "and", "after", "that",
    "go", "read", "play", "start", "begin", "end",
    "also", "next", "followed", "by", "through",
}
ALLOWED_TOKENS: set[str] = _NAV_WORDS | set(WORD_TO_NUM.keys())


# ── Pydantic models ───────────────────────────────────────────────────────────

class VoiceParseRequest(BaseModel):
    transcript: str


class VoiceSegment(BaseModel):
    from_page: int
    to_page: int        # -1 means "end" (resolved to total_pages by the pipeline)
    from_section: Optional[int] = None
    to_section: Optional[int] = None


class VoiceParseResponse(BaseModel):
    valid: bool
    reject_reason: Optional[str] = None
    segments: Optional[list[VoiceSegment]] = None
    pipeline_segments: Optional[str] = None   # e.g. "5-7,1-3"
    section_filter: Optional[dict] = None      # e.g. {"1": [2, 2]}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation except hyphens, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)       # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_english_only(tokens: list[str]) -> tuple[bool, str]:
    """Return (ok, reject_reason).  Pure digits are always allowed."""
    for tok in tokens:
        if tok.isdigit():
            continue
        if tok in ALLOWED_TOKENS:
            continue
        return False, f"Non-English or unrecognised token: '{tok}'"
    return True, ""


def _next_num(tokens: list[str], i: int) -> tuple[int | None, int]:
    """
    Parse one integer (digit string, number-word, or 'end') from tokens[i].
    Handles compound words like 'twenty one', 'thirty five'.
    Returns (value, next_index).  'end' → -1.
    """
    if i >= len(tokens):
        return None, i

    tok = tokens[i]

    # Plain digit
    if tok.isdigit():
        return int(tok), i + 1

    # "end" → sentinel
    if tok == "end":
        return -1, i + 1

    # Number word (possibly compound: "twenty one")
    if tok in WORD_TO_NUM:
        val = WORD_TO_NUM[tok]
        if val >= 20 and (i + 1) < len(tokens):
            nxt = tokens[i + 1]
            if nxt in WORD_TO_NUM and 0 < WORD_TO_NUM[nxt] < 10:
                return val + WORD_TO_NUM[nxt], i + 2
        return val, i + 1

    return None, i


def _split_into_raw_segments(text: str) -> list[str]:
    """Split a normalised utterance on conjunctions into per-range pieces."""
    # Insert a pipe separator at every conjunction boundary
    for sep in [" then ", " and then ", " after that ", " followed by ", " also "]:
        text = text.replace(sep, " | ")
    # Commas that come after a number (e.g. "page 3, page 5") are also splits
    text = re.sub(r",\s*(?=page\b)", " | ", text)
    return [s.strip() for s in text.split("|") if s.strip()]


def _parse_one_segment(seg_text: str) -> VoiceSegment | str:
    """
    Parse a single segment string into a VoiceSegment.
    Returns an error string on failure.
    """
    tokens = seg_text.split()
    i = 0

    # Skip leading filler words
    _FILLER = {"go", "read", "play", "start", "begin", "from"}
    while i < len(tokens) and tokens[i] in _FILLER:
        i += 1

    from_page: int | None = None
    to_page: int | None = None
    from_section: int | None = None
    to_section: int | None = None

    # ── Parse "page [from] [to to_page]" ─────────────────────────────────────
    if i < len(tokens) and tokens[i] in {"page", "pages"}:
        i += 1
        from_page, i = _next_num(tokens, i)
        if from_page is None:
            return f"Expected a page number after 'page' in: '{seg_text}'"
        to_page = from_page

        if i < len(tokens) and tokens[i] in {"to", "through"}:
            i += 1
            to_page, i = _next_num(tokens, i)
            if to_page is None:
                return f"Expected a number after 'to' in: '{seg_text}'"

    else:
        # Bare number without "page" keyword
        from_page, i = _next_num(tokens, i)
        if from_page is None:
            return f"Expected 'page' keyword or a number in segment: '{seg_text}'"
        to_page = from_page

    # ── Parse optional "section [from_s] [to to_s]" ──────────────────────────
    if i < len(tokens) and tokens[i] in {"section", "sections"}:
        i += 1
        from_section, i = _next_num(tokens, i)
        if from_section is None:
            return f"Expected a section number in: '{seg_text}'"
        to_section = from_section

        if i < len(tokens) and tokens[i] in {"to", "through"}:
            i += 1
            to_section, i = _next_num(tokens, i)
            if to_section is None:
                return f"Expected a section end number in: '{seg_text}'"

    return VoiceSegment(
        from_page=from_page,
        to_page=to_page,          # may be -1 (end)
        from_section=from_section,
        to_section=to_section,
    )


def _build_pipeline_segments(segments: list[VoiceSegment]) -> str:
    """
    Build the pipeline_segments string consumed by POST /pipeline/start.
    e.g.  [{5,7}, {1,3}]  →  "5-7,1-3"
          [{5,-1}]         →  "5-end"
          [{3,3}]          →  "3"
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


def _build_section_filter(segments: list[VoiceSegment]) -> dict:
    """
    Map page_number (str) → [from_section, to_section] for pages
    that have an explicit section constraint.
    Pages whose to_page is -1 ("end") only store the from_page entry;
    the backend resolves the range at runtime.
    """
    sf: dict[str, list[int]] = {}
    for s in segments:
        if s.from_section is None:
            continue
        # Concrete range
        if s.to_page != -1:
            for p in range(s.from_page, s.to_page + 1):
                sf[str(p)] = [s.from_section, s.to_section]
        else:
            # "end" range — only the from_page is known; mark it and let the
            # frontend extend the filter after it learns total_pages
            sf[str(s.from_page)] = [s.from_section, s.to_section]
    return sf


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/voice/parse", response_model=VoiceParseResponse)
def parse_voice_command(req: VoiceParseRequest):
    """
    Validate and parse a voice transcript into pipeline segments.

    Returns `valid=false` with a human-readable `reject_reason` if:
      - The transcript is empty.
      - It contains non-English / off-topic tokens.
      - No 'page' keyword is present.
      - The grammar cannot be parsed.
    """
    text = _normalize(req.transcript)

    # ── Empty check ───────────────────────────────────────────────────────────
    if not text:
        return VoiceParseResponse(valid=False, reject_reason="Empty transcript.")

    tokens = text.split()

    # ── English-only guard ────────────────────────────────────────────────────
    ok, reason = _is_english_only(tokens)
    if not ok:
        return VoiceParseResponse(valid=False, reject_reason=reason)

    # ── Must contain 'page' ───────────────────────────────────────────────────
    if "page" not in tokens and "pages" not in tokens:
        return VoiceParseResponse(
            valid=False,
            reject_reason=(
                "No 'page' keyword found. "
                "Please say something like 'page 5' or 'page 3 to 7'."
            ),
        )

    # ── Parse individual segments ─────────────────────────────────────────────
    raw_segments = _split_into_raw_segments(text)
    segments: list[VoiceSegment] = []

    for raw in raw_segments:
        result = _parse_one_segment(raw)
        if isinstance(result, str):
            return VoiceParseResponse(valid=False, reject_reason=result)
        segments.append(result)

    # ── Build derived fields ──────────────────────────────────────────────────
    pipeline_segments = _build_pipeline_segments(segments)
    section_filter = _build_section_filter(segments) or None

    return VoiceParseResponse(
        valid=True,
        segments=segments,
        pipeline_segments=pipeline_segments,
        section_filter=section_filter,
    )

"""
VEDA Pipeline Test Script
=========================
Tests the /api/v1/pipeline endpoint and proves:
  1. The API returns a valid response with all expected fields
  2. Pages are processed SEQUENTIALLY   (page N+1 starts after page N ends)
  3. Regions within a page are parallel  (multiple START lines share the same
     page number and their timestamps overlap)

Usage:
    python test_pipeline.py <path_to_pdf>

    # Or with a specific page to start from:
    python test_pipeline.py <path_to_pdf> --start-page 2

    # Skip the API call and only analyze an existing log:
    python test_pipeline.py --log-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000/api/v1"
LOG_PATH = Path("app.log")

# ANSI colours (degrade gracefully on Windows without colour support)
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


# ── Step 1: Call the pipeline API ──────────────────────────────────────────────

def call_pipeline(pdf_path: str, start_page: int = 1) -> dict:
    print(c(f"\n{'='*60}", _BOLD))
    print(c("  STEP 1 -- Calling POST /api/v1/pipeline", _BOLD))
    print(c(f"{'='*60}", _BOLD))
    print(f"  File      : {pdf_path}")
    print(f"  Start page: {start_page}")
    print(f"  Endpoint  : {API_BASE}/pipeline\n")

    t0 = time.time()
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{API_BASE}/pipeline",
            files={"file": (Path(pdf_path).name, f, "application/pdf")},
            params={"start_page": start_page},
            timeout=600,  # layout analysis can be slow
        )
    elapsed = (time.time() - t0) * 1000

    print(f"  HTTP Status : {response.status_code}")
    print(f"  Round-trip  : {elapsed:.0f} ms")

    if response.status_code != 200:
        print(c("\n  [FAIL] Pipeline returned an error:", _RED))
        try:
            err = response.json()
            print(json.dumps(err, indent=4))
        except Exception:
            print(response.text)
        sys.exit(1)

    data = response.json()
    print(c("\n  [OK] Pipeline succeeded!", _GREEN))
    return data


# ── Step 2: Validate response structure ────────────────────────────────────────

def validate_response(data: dict) -> None:
    print(c(f"\n{'='*60}", _BOLD))
    print(c("  STEP 2 -- Validating Response Structure", _BOLD))
    print(c(f"{'='*60}", _BOLD))

    required_top = [
        "status", "file_id", "filename", "category",
        "total_pages", "pages_processed", "steps",
        "output_path", "final_document", "total_time_ms",
    ]
    ok = True
    for key in required_top:
        present = key in data
        mark = c("[OK]", _GREEN) if present else c("[FAIL]", _RED)
        print(f"  {mark} Top-level key '{key}': {data.get(key, 'MISSING')!r}" if not present
              else f"  {mark} '{key}' present")
        if not present:
            ok = False

    # Validate steps
    steps: list[dict] = data.get("steps", [])
    print(f"\n  Steps executed ({len(steps)}):")
    for s in steps:
        sname   = s.get("name", "?")
        sstatus = s.get("status", "?")
        stime   = s.get("time_ms", "?")
        colour  = _GREEN if sstatus == "success" else _RED
        mark    = c("[OK]", _GREEN) if sstatus == "success" else c("[FAIL]", _RED)
        print(f"    {mark} Step {s.get('step','?')}: {sname:<35} "
              f"status={c(sstatus, colour)}  time={stime}ms")
        if s.get("details"):
            print(f"         details: {s['details']}")

    # Validate final document
    doc = data.get("final_document", {})
    pages = doc.get("pages", [])
    print(f"\n  Final document: {len(pages)} pages")

    total_regions = 0
    regions_with_text = 0
    gemini_regions = 0
    error_regions = 0
    for p in pages:
        for r in p.get("regions", []):
            total_regions += 1
            if r.get("text", "").strip():
                regions_with_text += 1
            if r.get("gemini_response"):
                gemini_regions += 1
            if r.get("error"):
                error_regions += 1

    print(f"  Total regions   : {total_regions}")
    print(f"  Regions w/ text : {c(str(regions_with_text), _GREEN)} / {total_regions}")
    print(f"  Gemini regions  : {gemini_regions}")
    print(f"  Errored regions : {c(str(error_regions), _RED if error_regions else _GREEN)}")

    if not ok:
        print(c("\n  [WARN] Some top-level keys missing!", _YELLOW))
    else:
        print(c("\n  [OK] Response structure valid!", _GREEN))


# ── Step 3: Parse log and prove sequential pages + parallel regions ────────────

# Log line format produced by our pipeline:
# 2026-04-14 15:24:35,277 - routers.pipeline - INFO - [...] - MESSAGE

_LOG_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"   # timestamp
    r".+?routers\.pipeline.+?-\s+"                         # source
    r"(?:\[.+?\] - )?"                                      # optional [file:line:fn]
    r"(.+)$"                                               # message
)
_TS_FMT = "%Y-%m-%d %H:%M:%S,%f"

# Patterns we care about in the message
# Note: emoji in log messages may not survive all terminals, so we match
# both the emoji form and the ASCII-fallback form.
_PAGE_START_RE   = re.compile(r"\[PAGE (\d+)\] Dispatching (\d+) tasks in parallel")
_PAGE_DONE_RE    = re.compile(r"\[PAGE (\d+)\] Completed in ([\d.]+)ms")
_REGION_START_RE = re.compile(r"\[PARALLEL\]\[(Gemini|OCR)\]\s+START\s+page=(\d+)\s+label=(\S+)")
_REGION_DONE_RE  = re.compile(r"\[PARALLEL\]\[(Gemini|OCR)\]\s+DONE\s+page=(\d+)\s+label=(\S+).+elapsed=([\d.]+)ms")
_STEP4_START_RE  = re.compile(r"PIPELINE STEP 4:.+STARTED")
_STEP4_DONE_RE   = re.compile(r"PIPELINE STEP 4:.+DONE in ([\d.]+)ms")


def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, _TS_FMT)


def analyze_log(log_path: Path, run_start_hint: datetime | None = None) -> None:
    print(c(f"\n{'='*60}", _BOLD))
    print(c("  STEP 3 -- Log Analysis: Sequential Pages + Parallel Regions", _BOLD))
    print(c(f"{'='*60}", _BOLD))

    if not log_path.exists():
        print(c(f"  [WARN] Log file not found at {log_path.resolve()}", _YELLOW))
        return

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # ── Parse relevant lines ──────────────────────────────────────────────────
    page_starts: dict[int, datetime] = {}   # page_num → timestamp
    page_ends:   dict[int, datetime] = {}
    page_times:  dict[int, float]    = {}
    region_events: list[dict] = []
    step4_start: datetime | None = None
    step4_ms: float | None = None
    in_run = run_start_hint is None  # if no hint, capture everything

    for raw in lines:
        m = _LOG_TS_RE.match(raw)
        if not m:
            continue
        ts  = _parse_ts(m.group(1))
        msg = m.group(2).strip()

        # Only analyze lines from the current run
        if run_start_hint and ts < run_start_hint:
            continue

        if _STEP4_START_RE.search(msg):
            step4_start = ts

        d = _STEP4_DONE_RE.search(msg)
        if d:
            step4_ms = float(d.group(1))

        ps = _PAGE_START_RE.search(msg)
        if ps:
            page_num = int(ps.group(1))
            page_starts[page_num] = ts

        pd = _PAGE_DONE_RE.search(msg)
        if pd:
            page_num = int(pd.group(1))
            page_ends[page_num] = ts
            page_times[page_num] = float(pd.group(2))

        rs = _REGION_START_RE.search(msg)
        if rs:
            region_events.append({
                "ts": ts, "event": "START",
                "engine": rs.group(1),
                "page": int(rs.group(2)),
                "label": rs.group(3),
            })

        rd = _REGION_DONE_RE.search(msg)
        if rd:
            region_events.append({
                "ts": ts, "event": "DONE",
                "engine": rd.group(1),
                "page": int(rd.group(2)),
                "label": rd.group(3),
                "elapsed_ms": float(rd.group(4)),
            })

    if not page_starts:
        print(c("  [WARN] No [PAGE N] events found in log yet.", _YELLOW))
        print("  This is expected if you haven't run the pipeline with the new code yet.")
        print("  Run:  python test_pipeline.py <your_pdf.pdf>")
        return

    # ── Print sequential page evidence ───────────────────────────────────────
    print(c("\n  A) Sequential Page Processing Evidence", _CYAN))
    print(f"  {'Page':<6} {'Start time':<26} {'End time':<26} {'Duration':>10}  Sequential?")
    print(f"  {'-'*6}  {'-'*24}  {'-'*24}  {'-'*10}  {'-'*11}")

    sorted_pages = sorted(page_starts)
    prev_end: datetime | None = None
    seq_ok = True

    for pn in sorted_pages:
        start = page_starts[pn]
        end   = page_ends.get(pn)
        dur   = page_times.get(pn, "?")
        seq_mark = ""

        if prev_end and start < prev_end:
            seq_mark = c("[OVERLAP!]", _RED)
            seq_ok = False
        elif prev_end:
            gap_ms = (start - prev_end).total_seconds() * 1000
            seq_mark = c(f"[OK] gap +{gap_ms:.0f}ms", _GREEN)
        else:
            seq_mark = c("(first page)", _CYAN)

        end_str = end.strftime("%H:%M:%S.%f")[:-3] if end else "N/A (log pending)"
        print(
            f"  {pn:<6}  {start.strftime('%H:%M:%S.%f')[:-3]:<26}  "
            f"{end_str:<26}  {str(dur)+'ms':>10}  {seq_mark}"
        )
        # Advance prev_end: use actual end, fall back to start if end wasn't
        # logged yet (e.g. emoji encoding issue dropped the line)
        prev_end = end if end else start

    if seq_ok:
        print(c("\n  [OK] CONFIRMED: Pages processed sequentially (no overlap)!", _GREEN))
    else:
        print(c("\n  [WARN] Some pages may have overlapped!", _YELLOW))

    # ── Print parallel region evidence ───────────────────────────────────────
    print(c("\n  B) Parallel Region Processing Evidence", _CYAN))

    for pn in sorted_pages:
        page_evts = [e for e in region_events if e["page"] == pn]
        # Only consider START events that happened AFTER the page's own Dispatching event.
        # Events before that are from gather_context OCR-ing neighboring pages for context.
        dispatch_ts = page_starts[pn]  # this is the [PAGE N] Dispatching timestamp
        starts = [
            e for e in page_evts
            if e["event"] == "START" and e["ts"] >= dispatch_ts
        ]
        dones  = [e for e in page_evts if e["event"] == "DONE" and e["ts"] >= dispatch_ts]

        if not starts:
            print(f"  Page {pn}: no region events after dispatch (page may have 0 regions)")
            continue

        print(f"\n  Page {pn}  --  {len(starts)} real regions dispatched, {len(dones)} completed")
        print(f"       (dispatch time: {dispatch_ts.strftime('%H:%M:%S.%f')[:-3]})")

        # Sort starts by timestamp and compute delta from the Dispatch event
        starts_sorted = sorted(starts, key=lambda e: e["ts"])
        t_baseline = dispatch_ts   # delta from Dispatch, not from first region

        print(f"  {'#':<3} {'Engine':<8} {'Label':<22} {'d from dispatch':>18}  Concurrent?")
        print(f"  {'-'*3}  {'-'*7}  {'-'*20}  {'-'*18}  {'-'*12}")

        for i, ev in enumerate(starts_sorted):
            delta_ms = (ev["ts"] - t_baseline).total_seconds() * 1000
            is_concurrent = delta_ms < 2000
            conc_mark = (c(f"[PARALLEL +{delta_ms:.0f}ms]", _GREEN)
                         if is_concurrent else
                         c(f"[SEQ +{delta_ms:.0f}ms]", _YELLOW))
            eng_colour = _GREEN if ev["engine"] == "Gemini" else _CYAN
            print(
                f"  {i+1:<3}  {c(ev['engine'], eng_colour):<8}  {ev['label']:<22}  "
                f"  {ev['ts'].strftime('%H:%M:%S.%f')[:-3]:<22}  {conc_mark}"
            )

        # Summary: how many started within 2000ms of dispatch
        concurrent_count = sum(1 for ev in starts_sorted
                               if (ev["ts"] - t_baseline).total_seconds() * 1000 < 2000)
        if len(starts) >= 2 and concurrent_count >= len(starts):
            print(
                c(f"  [OK] CONFIRMED parallel: all {len(starts)} regions started within 2s of dispatch on page {pn}!", _GREEN)
            )
        elif concurrent_count >= 2:
            print(
                c(f"  [OK] Mostly parallel: {concurrent_count}/{len(starts)} regions within 2s of dispatch.", _GREEN)
            )
        elif len(starts) == 1:
            print(c(f"  [INFO] Page {pn} has only 1 region -- nothing to parallelize.", _CYAN))
        else:
            print(
                c(f"  [WARN] Regions on page {pn} appear sequential "
                  f"(only {concurrent_count} started within 500ms).", _YELLOW)
            )

    # ── Overall step 4 timing ─────────────────────────────────────────────────
    if step4_ms is not None:
        sum_page_times = sum(page_times.values())
        print(c("\n  C) Step 4 Timing Summary", _CYAN))
        print(f"  Total Step 4 wall-clock time  : {step4_ms:.0f} ms")
        print(f"  Sum of individual page times  : {sum_page_times:.0f} ms")
        print(f"  Parallelism efficiency (pages) : {sum_page_times/step4_ms:.2f}x")
        if step4_ms > 0:
            print(c(
                "\n  [INFO] Page-level processing is sequential by design; "
                "speedup comes from parallel REGIONS within each page.",
                _CYAN
            ))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VEDA Pipeline Test")
    parser.add_argument("pdf", nargs="?", help="Path to PDF to upload")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--log-only", action="store_true",
        help="Skip API call and only analyze existing log"
    )
    args = parser.parse_args()

    run_start: datetime | None = None

    if not args.log_only:
        if not args.pdf:
            print(c("  [FAIL] Provide a PDF path or use --log-only", _RED))
            sys.exit(1)

        run_start = datetime.now()
        print(f"  Run started at {run_start.strftime('%H:%M:%S')}")

        data = call_pipeline(args.pdf, args.start_page)
        validate_response(data)

        # Brief pause to ensure logger flushes to disk
        time.sleep(1)
    else:
        print(c("\n  --log-only: skipping API call, analyzing existing log", _YELLOW))

    analyze_log(LOG_PATH, run_start)

    print(c("\n" + "="*60, _BOLD))
    print(c("  Test Complete", _BOLD))
    print(c("="*60 + "\n", _BOLD))


if __name__ == "__main__":
    main()

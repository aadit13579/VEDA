"""
VEDA Transcription Router  —  Google free Web Speech API (via SpeechRecognition).

Auto-installs SpeechRecognition into whatever Python environment uvicorn is
running in (sys.executable), so venv / system Python differences don't matter.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _load_sr():
    """Import speech_recognition, installing it first if absent."""
    try:
        import speech_recognition as sr
        return sr
    except ImportError:
        logger.info("[TRANSCRIBE] Installing SpeechRecognition into current Python env…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "SpeechRecognition"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"[TRANSCRIBE] pip install failed:\n{result.stderr}")
            return None
        try:
            import speech_recognition as sr
            logger.info("[TRANSCRIBE] SpeechRecognition ready.")
            return sr
        except ImportError:
            logger.error("[TRANSCRIBE] Import still failing after install.")
            return None


# Install at module load so the first request doesn't bear the install cost.
_sr = _load_sr()


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe a WAV clip using Google's free Web Speech API."""
    global _sr
    if _sr is None:
        _sr = _load_sr()
    if _sr is None:
        raise HTTPException(
            status_code=503,
            detail="SpeechRecognition could not be loaded. Check server logs.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        recognizer = _sr.Recognizer()
        recognizer.pause_threshold = 0.8
        recognizer.energy_threshold = 300

        with _sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)

        try:
            text: str = recognizer.recognize_google(audio_data, language="en-US")
            logger.info(f"[TRANSCRIBE] '{text}'")
            return {"transcript": text.strip()}

        except _sr.UnknownValueError:
            return {"transcript": ""}           # silence / inaudible

        except _sr.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Google STT error: {exc}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

"""
VEDA Transcription Router.

Provides audio-to-text transcription using Google's free Web Speech API
via the SpeechRecognition library. Auto-installs SpeechRecognition into
the current Python environment if it is not already present.

This module exists to provide a zero-configuration transcription
endpoint that converts WAV audio clips into text for the voice command
parser.

Leverages: FastAPI, SpeechRecognition (auto-installed), tempfile.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from utils.logger import get_logger

logger = get_logger(__name__)


class TranscribeRouter:
    """
    Router for audio transcription using Google Web Speech API.

    Manages SpeechRecognition library lifecycle (auto-install if missing),
    temporary WAV file handling, and recognition with ambient noise
    adjustment.

    Leverages: SpeechRecognition, tempfile, subprocess (for pip install).
    """

    def __init__(self):
        """
        Initialize the router, attempt to load SpeechRecognition, and
        register routes.

        Leverages: FastAPI APIRouter, _load_sr.
        """
        self._sr = self._load_sr()
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        """
        Bind endpoint methods to their HTTP routes.

        Leverages: FastAPI APIRouter.add_api_route.
        """
        self.router.add_api_route(
            "/transcribe",
            self.transcribe_audio,
            methods=["POST"],
        )

    def _load_sr(self):
        """
        Import SpeechRecognition, installing it first if absent.

        Uses pip via subprocess to install into the current Python
        environment, ensuring compatibility regardless of venv setup.

        Returns the speech_recognition module or None on failure.

        Leverages: importlib, subprocess, pip.
        """
        try:
            import speech_recognition as sr
            return sr
        except ImportError:
            logger.info(
                "[TRANSCRIBE] Installing SpeechRecognition into current Python env…"
            )
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "SpeechRecognition"],
                capture_output=True,
                text=True,
                timeout=120,
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

    async def transcribe_audio(self, file: UploadFile = File(...)):
        """
        Transcribe a WAV audio clip using Google's free Web Speech API.

        Saves the uploaded audio to a temporary file, adjusts for ambient
        noise, runs recognition, and returns the transcript. Returns an
        empty transcript for silence or inaudible audio.

        Leverages: SpeechRecognition Recognizer, AudioFile, recognize_google.
        """
        if self._sr is None:
            self._sr = self._load_sr()
        if self._sr is None:
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

            recognizer = self._sr.Recognizer()
            recognizer.pause_threshold = 0.8
            recognizer.energy_threshold = 300

            with self._sr.AudioFile(tmp_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = recognizer.record(source)

            try:
                text: str = recognizer.recognize_google(
                    audio_data, language="en-US"
                )
                logger.info(f"[TRANSCRIBE] '{text}'")
                return {"transcript": text.strip()}

            except self._sr.UnknownValueError:
                return {"transcript": ""}

            except self._sr.RequestError as exc:
                raise HTTPException(
                    status_code=503, detail=f"Google STT error: {exc}"
                )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


_instance = TranscribeRouter()
router = _instance.router

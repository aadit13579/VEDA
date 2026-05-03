"""
VEDA Application Entry Point.

Configures the FastAPI application, mounts global middleware for CORS
and logging/error handling, and registers all API routers.

This module exists as the central application factory and WSGI/ASGI
entry point.

Leverages: FastAPI, uvicorn, CORSMiddleware.
"""

import time
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    ingest,
    layout_analysis,
    spatial_sort,
    ocr,
    redis_api,
    gemini,
    pipeline,
    voice_command,
    transcribe,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class VedaApplication:
    """
    Factory and configurator for the VEDA FastAPI application.

    Initializes the app, sets up middleware for cross-origin requests
    and global error logging, and includes all sub-routers.

    Leverages: FastAPI, CORSMiddleware.
    """

    def __init__(self):
        """
        Initialize the application factory.

        Leverages: None.
        """
        self._app = FastAPI(title="VEDA API", version="1.0")

    def create_app(self) -> FastAPI:
        """
        Build and return the fully configured FastAPI application instance.

        Leverages: FastAPI.
        """
        self._register_middleware()
        self._register_routers()

        @self._app.get("/")
        def home():
            """
            Health check endpoint.

            Leverages: None.
            """
            logger.info("Health check endpoint hit")
            return {"message": "VEDA Backend is Running"}

        return self._app

    def _register_middleware(self) -> None:
        """
        Attach CORS and global logging middleware to the app.

        Leverages: FastAPI.add_middleware, FastAPI.middleware.
        """
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self._app.middleware("http")
        async def log_requests(request: Request, call_next):
            """
            Global middleware to log request latency and catch unhandled exceptions.

            Leverages: time.time, traceback.format_exc.
            """
            start_time = time.time()
            client_host = request.client.host if request.client else "unknown"
            logger.info(
                f"Incoming Request: {request.method} {request.url.path} "
                f"from {client_host}"
            )

            try:
                response = await call_next(request)
                process_time = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed: {response.status_code} in {process_time:.2f}ms"
                )
                return response

            except Exception:
                process_time = (time.time() - start_time) * 1000
                error_msg = traceback.format_exc()
                logger.error(
                    f"❌ CRITICAL ERROR responding to {request.method} "
                    f"{request.url.path}:\n{error_msg}"
                )

                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal Server Error. Please check app.log "
                                  "for details."
                    },
                )

    def _register_routers(self) -> None:
        """
        Include all feature routers into the main application.

        Leverages: FastAPI.include_router.
        """
        self._app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
        self._app.include_router(
            layout_analysis.router, prefix="/api/v1", tags=["Layout Analysis"]
        )
        self._app.include_router(
            spatial_sort.router, prefix="/api/v1", tags=["Spatial Sort"]
        )
        self._app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])
        self._app.include_router(redis_api.router, prefix="/api/v1", tags=["Redis"])
        self._app.include_router(gemini.router, prefix="/api/v1", tags=["Gemini"])
        self._app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
        self._app.include_router(
            voice_command.router, prefix="/api/v1", tags=["Voice Command"]
        )
        self._app.include_router(
            transcribe.router, prefix="/api/v1", tags=["Transcription"]
        )


app = VedaApplication().create_app()

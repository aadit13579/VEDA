from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest, layout_analysis, spatial_sort, ocr, redis_api, gemini
from utils.logger import get_logger
import time
import traceback

logger = get_logger(__name__)

app = FastAPI(title="VEDA API", version="1.0")

# Enable CORS so your Electron/React App can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL LOGGING MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the incoming request
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Incoming Request: {request.method} {request.url.path} from {client_host}")
    
    try:
        response = await call_next(request)
        
        # Log the response status code
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Completed: {response.status_code} in {process_time:.2f}ms")
        
        return response
        
    except Exception as e:
        # Catch ALL unhandled exceptions (500 Internal Server Errors)
        process_time = (time.time() - start_time) * 1000
        
        # Log the full traceback to app.log
        error_msg = traceback.format_exc()
        logger.error(f"❌ CRITICAL ERROR responding to {request.method} {request.url.path}:\n{error_msg}")
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error. Please check app.log for details."}
        )

# Register the Routers
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(layout_analysis.router, prefix="/api/v1", tags=["Layout Analysis"])
app.include_router(spatial_sort.router, prefix="/api/v1", tags=["Spatial Sort"])
app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])
app.include_router(redis_api.router, prefix="/api/v1", tags=["Redis"])
app.include_router(gemini.router, prefix="/api/v1", tags=["Gemini"])


@app.get("/")
def home():
    logger.info("Health check endpoint hit")
    return {"message": "VEDA Backend is Running"}


# Run with: uvicorn main:app --reload

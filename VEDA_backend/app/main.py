from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest  # We only need the ingest router now
from app.utils.logger import get_logger
import time
import traceback

logger = get_logger(__name__)

app = FastAPI(title="VEDA API", version="1.0")

# --- CORS SETTINGS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL LOGGING MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"🚀 {request.method} {request.url.path} | Host: {client_host}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"✅ {response.status_code} | Time: {process_time:.2f}ms")
        return response
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"❌ CRITICAL ERROR:\n{error_msg}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error. Check logs."}
        )

# --- ROUTER REGISTRATION ---
# We point everything to ingest.py which will handle the task orchestration
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "VEDA Backend",
        "pipeline": "Classifier -> Layout -> Spatial Sort"
    }

# Run with: uvicorn app.main:app --reload
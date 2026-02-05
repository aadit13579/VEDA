from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest

app = FastAPI(title="VEDA API", version="1.0")

# Enable CORS so your Electron/React App can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change to ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the Routers
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])

@app.get("/")
def home():
    return {"message": "VEDA Backend is Running"}

# Run with: uvicorn main:app --reload
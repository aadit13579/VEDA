from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
# Ensure the prefix 'app.' is used for all internal imports
from app.tasks import process_veda_document
from app.services.redis_client import redis_instance
import uuid
import json

router = APIRouter()

@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_bytes = await file.read()
    
    # Trigger the unified pipeline in the background
    background_tasks.add_task(process_veda_document, job_id, file_bytes)
    
    return {"job_id": job_id, "message": "Processing started"}

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    status = redis_instance.get(f"status:{job_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    # Redis returns bytes, so we decode to string
    return {
        "job_id": job_id, 
        "status": status.decode("utf-8") if isinstance(status, bytes) else status
    }

@router.get("/result/{job_id}")
async def get_result(job_id: str):
    result = redis_instance.get(f"result:{job_id}")
    if not result:
        raise HTTPException(status_code=404, detail="Result not ready or job failed")
    # Parse the JSON string stored in Redis back into a dictionary
    return json.loads(result)
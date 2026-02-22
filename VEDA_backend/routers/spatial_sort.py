import time
from fastapi import APIRouter
from typing import Dict, Any
from services.spatial_sort_engine import process_spatial_sort
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("/layout/sort")
async def sort_layout(payload: Dict[str, Any]):
    """
    Endpoint to receive unordered DocLayout-YOLO region payload, 
    filter 'abandon' elements, and output JSON ordered with 'reading_order'.
    """
    start_time = time.time()
    logger.info("Received layout sort request.")
    
    ordered_payload = process_spatial_sort(payload)
    
    process_time = (time.time() - start_time) * 1000
    ordered_payload["process_time_ms"] = round(process_time, 2)
    
    logger.info("Layout sort completed successfully.")
    return ordered_payload

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import glob
from services.layout_engine import pdf_to_images, analyze_layout, draw_layout_on_image
from utils.logger import get_logger
import traceback
import time

logger = get_logger(__name__)

router = APIRouter()

STORAGE_DIR = "storage"
DEBUG_DIR = "storage/debug"  # Where we save images with boxes
os.makedirs(DEBUG_DIR, exist_ok=True)


@router.post("/analyze_layout/{file_id}")
async def generate_bounding_boxes(file_id: str):
    logger.info(f"Received layout analysis request for file_id: {file_id}")
    start_time = time.time()
    """
    1. Finds the file using file_id.
    2. Runs YOLO layout analysis.
    3. Draws boxes on the images.
    4. Returns JSON data + URLs to the debug images.
    """

    # 1. Find file (we don't know extension, so we glob)
    search_pattern = os.path.join(STORAGE_DIR, f"{file_id}.*")
    files = glob.glob(search_pattern)

    if not files:
        raise HTTPException(
            status_code=404, detail="File not found. Please upload via /ingest first."
        )

    file_path = files[0]

    try:
        # 2. Convert to Images
        # Note: If it's already an image, we handle that; if PDF, we convert.
        # For this snippet, assuming PDF as per your ingest logic, or we can add image logic later.
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        images = pdf_to_images(file_bytes)

        results = []

        # 3. Process each page
        for page_num, img in enumerate(images):
            # Run AI
            regions = analyze_layout(img)

            # Draw Boxes
            output_filename = f"{file_id}_page_{page_num+1}.jpg"
            output_path = os.path.join(DEBUG_DIR, output_filename)
            draw_layout_on_image(img, regions, output_path)

            results.append(
                {
                    "page": page_num + 1,
                    "regions": regions,
                    "debug_image_url": f"/api/v1/layout/debug_image/{output_filename}",
                }
            )

        process_time = (time.time() - start_time) * 1000
        
        logger.info(f"Layout analysis completed for {file_id} in {process_time:.2f}ms")

        return {
            "status": "success",
            "file_id": file_id,
            "pages_processed": len(images),
            "layout_data": results,
            "process_time_ms": round(process_time, 2)
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"❌ Error during layout analysis for {file_id}:\n{error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/layout/debug_image/{filename}")
async def get_debug_image(filename: str):
    """Serve the generated images so Frontend can display them"""
    file_path = os.path.join(DEBUG_DIR, filename)
    logger.info(f"Requested debug image: {filename}")
    logger.info(f"Looking for file at: {os.path.abspath(file_path)}")
    
    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    logger.error(f"❌ Image not found at: {os.path.abspath(file_path)}")
    raise HTTPException(status_code=404, detail="Image not found")

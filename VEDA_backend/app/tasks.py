import json
import logging
from app.services.layout_engine import pdf_to_images, analyze_layout
from app.services.spatial_sort_engine import process_spatial_sort
from app.services.redis_client import redis_instance
from app.services.ocr_engine import extract_text_from_region
import cv2
import numpy as np
import filetype

logger = logging.getLogger("veda")

async def process_veda_document(file_id: str, file_bytes: bytes):
    """
    VEDA Backend Pipeline: PDF/Image -> Layout Engine -> Spatial Sorter -> OCR
    """
    try:
        # --- STEP 0: DETERMINE FILE TYPE ---
        redis_instance.set(f"status:{file_id}", "Detecting file type...")
        kind = filetype.guess(file_bytes)
        
        images = []
        if kind is None:
            raise ValueError("Unknown file type")
        elif kind.mime == "application/pdf":
            # Convert PDF to images
            images = pdf_to_images(file_bytes)
        elif kind.mime.startswith("image/"):
            # Decode single image
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
            else:
                raise ValueError("Could not decode image")
        else:
            raise ValueError(f"Unsupported file type: {kind.mime}")
            
        doc_type = "general_document" 
        
        layout_payload = {
            "layout_data": []
        }
        
        # Process each page
        for page_num, image in enumerate(images):
            # --- STEP 1: LAYOUT ANALYSIS ---
            redis_instance.set(f"status:{file_id}", f"Extracting layout regions (Page {page_num + 1}/{len(images)})...")
            
            raw_regions = analyze_layout(image)
            
            layout_payload["layout_data"].append({
                "page": page_num + 1,
                "regions": raw_regions
            })

        # --- STEP 2: SPATIAL SORTING ---
        redis_instance.set(f"status:{file_id}", "Applying XY-Cut sorting...")
        
        # This calls spatial sort on the entire document payload
        final_sorted_payload = process_spatial_sort(layout_payload)

        # --- STEP 3: TEXT EXTRACTION (OCR) ---
        redis_instance.set(f"status:{file_id}", "Extracting text using OCR...")
        
        final_text = ""
        
        # Iterate through the sorted payload to extract text in reading order
        for page_data in final_sorted_payload.get("layout_data", []):
            page_num = page_data.get("page", 1) - 1
            if page_num >= len(images):
                continue
                
            image = images[page_num]
            regions = page_data.get("regions", [])
            
            for region in regions:
                r_type = region.get("type", "").lower()
                
                # We mainly care about text, titles, subtitles, etc. Add table handling later if needed.
                if r_type in ["text", "title", "text_block", "list", "paragraph"]:
                    bbox = region.get("bbox", [])
                    if len(bbox) == 4:
                        text = extract_text_from_region(image, bbox)
                        if text:
                            final_text += text + "\n\n"
                elif r_type in ["table", "figure", "picture"]:
                    # Placeholder for tables/figures
                    final_text += f"\n[{r_type.upper()} PLACEHOLDER]\n\n"

        # --- STEP 4: STORAGE ---
        result = {
            "file_id": file_id,
            "text": final_text.strip(),
            "processed_data": final_sorted_payload
        }

        redis_instance.set(f"result:{file_id}", json.dumps(result))
        redis_instance.set(f"status:{file_id}", "Completed")
        
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        redis_instance.set(f"status:{file_id}", f"Error: {str(e)}")
        raise e
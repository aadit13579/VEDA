from fastapi import APIRouter, UploadFile, File
import shutil
import os
import uuid
import filetype  # The library that reads magic numbers
import fitz      # PyMuPDF (for deep PDF checking)

router = APIRouter()

UPLOAD_DIR = "storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

import time

@router.post("/upload")
async def upload_and_identify(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Generate a Unique ID and Save Path
    file_id = str(uuid.uuid4())
    # We keep the original extension to help with fallback detection
    file_extension = os.path.splitext(file.filename)[1].lower()
    file_path = f"{UPLOAD_DIR}/{file_id}{file_extension}"
    
    # 2. Save the File Locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 3. Detect File Type (Magic Numbers)
    # We guess the type based on the file header, not the user's filename
    kind = filetype.guess(file_path)
    
    # Default values
    category = "UNKNOWN"
    mime_type = "unknown/unknown"

    if kind:
        mime_type = kind.mime
        
        # --- CASE A: IMAGES (PNG, JPEG, JPG) ---
        if mime_type.startswith('image/'):
            category = "IMAGE"

        # --- CASE B: PDF (Deep Scan for Digital vs Scanned) ---
        elif mime_type == 'application/pdf':
            category = _classify_pdf(file_path)

        # --- CASE C: OFFICE DOCUMENTS (Word/PPT) ---
        # Modern Office files (docx, pptx) are actually ZIPs, so we check specific mimes
        elif 'word' in mime_type or 'officedocument.wordprocessingml' in mime_type:
            category = "OFFICE_WORD"
        elif 'presentation' in mime_type or 'powerpoint' in mime_type:
            category = "OFFICE_PPT"
            
    # --- FALLBACK: If Magic Numbers fail (common for older .doc/.ppt) ---
    if category == "UNKNOWN":
        if file_extension in ['.doc', '.docx']:
            category = "OFFICE_WORD"
        elif file_extension in ['.ppt', '.pptx']:
            category = "OFFICE_PPT"
        elif file_extension in ['.txt']:
            category = "TEXT_FILE"
            
    process_time = (time.time() - start_time) * 1000
    
    return {
        "status": "success",
        "file_id": file_id,
        "filename": file.filename,
        "detected_mime": mime_type,
        "category": category,
        "process_time_ms": round(process_time, 2)
    }

def _classify_pdf(path):
    """
    Helper function to check if a PDF is Text-Based (Digital) or Image-Based (Scanned).
    """
    try:
        doc = fitz.open(path)
        text_length = 0
        
        # Check the first 3 pages (checking all pages is too slow)
        for i in range(min(3, len(doc))):
            text_length += len(doc[i].get_text())
            
        # If we found meaningful text (more than 50 chars), it's Digital
        if text_length > 50:
            return "PDF_DIGITAL"
        else:
            return "PDF_SCANNED"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "PDF_SCANNED" # Default to scanned if we can't read it
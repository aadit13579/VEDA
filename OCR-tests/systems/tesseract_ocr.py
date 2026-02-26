import pytesseract
import cv2
import time

# Point to tesseract executable if on Windows (uncomment below)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_tesseract(image_path):
    """
    Returns: (recognized_text, time_taken)
    """
    img = cv2.imread(image_path)
    
    # Preprocessing (crucial for Tesseract)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary thresholding often helps Tesseract
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    start_time = time.time()
    
    # --psm 6 assumes a single uniform block of text
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    end_time = time.time()
    
    return text.strip(), (end_time - start_time)
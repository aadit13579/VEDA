from paddleocr import PaddleOCR
import time
import logging

# Initialize global instance to avoid reloading model per image
# use_angle_cls=True helps if text is rotated

logging.getLogger("ppocr").setLevel(logging.ERROR) # Suppress verbose logs

ocr_engine = PaddleOCR(use_angle_cls=False, lang='en')

def run_paddle(image_path):
    start_time = time.time()
    
    result = ocr_engine.ocr(image_path, cls=True)
    
    end_time = time.time()
    
    # Paddle returns a list of [box, (text, score)]
    # We join all detected text blocks
    full_text = ""
    if result and result[0]:
        full_text = " ".join([line[1][0] for line in result[0]])
        
    return full_text.strip(), (end_time - start_time)
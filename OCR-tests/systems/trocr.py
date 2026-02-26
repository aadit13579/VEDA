from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time
import torch
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize model and processor 
# 'microsoft/trocr-base-printed' is good for standard text. 
# Swap to 'microsoft/trocr-base-handwritten' for cursive/handwriting.
print("Loading TrOCR Model (this may take a minute)...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Move to GPU if available for faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_trocr(image_path):
    # TrOCR expects a PIL Image
    image = Image.open(image_path).convert("RGB")
    
    start_time = time.time()
    
    # Process image and generate text
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    end_time = time.time()
    
    return generated_text.strip(), (end_time - start_time)
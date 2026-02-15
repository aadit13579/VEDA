from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import requests
import time
import io

# 1. Load the Model
print("⬇️ Loading TrOCR Model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")


def run_inference(image, label="Test"):
    print(f"\n--- Running Inference: {label} ---")
    start = time.time()

    # Preprocess
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end = time.time()
    print(f"✅ Result: '{generated_text}'")
    print(f"⏱️ Time  : {end - start:.4f}s")
    return generated_text


# --- TEST 1: Synthetic Image (The "Sanity Check") ---
# We create a perfect image locally to prove the model works
print("\n[Test 1] Generating synthetic text image...")
img_white = Image.new("RGB", (400, 100), color=(255, 255, 255))
d = ImageDraw.Draw(img_white)
# Use default font or a simple one if available
try:
    # Attempt to use a larger font if available on system, else default (tiny)
    font = ImageFont.truetype("arial.ttf", 30)
except IOError:
    font = ImageFont.load_default()

d.text((50, 30), "Project VEDA", fill=(0, 0, 0), font=font)
img_white.save("debug_synthetic.jpg")  # Save to check
run_inference(img_white, "Synthetic 'Project VEDA'")


# --- TEST 2: Real Invoice Crop ---
url = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
print(f"\n[Test 2] Downloading real invoice from {url}...")
try:
    response = requests.get(url, stream=True)
    real_image = Image.open(response.raw).convert("RGB")

    # Crop the top-right header
    # NOTE: If this reads "LOCO" or "LOGO", it's because the template
    # has a "LOGO" placeholder in that spot!
    cropped_image = real_image.crop((450, 50, 750, 110))
    cropped_image.save("debug_real_crop.jpg")

    run_inference(cropped_image, "Real Invoice Header")
    print(
        "\nℹ️ Note: If result is 'LOCO', the template likely says 'LOGO' in that spot."
    )

except Exception as e:
    print(f"❌ Network test failed: {e}")

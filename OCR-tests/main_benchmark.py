import os
import pandas as pd

from systems.trocr import run_trocr       # <-- MOVED TO THE TOP
from systems.tesseract_ocr import run_tesseract
from systems.paddle_ocr import run_paddle

from metrics import calculate_metrics

IMAGE_DIR = 'data/images'
GT_DIR = 'data/ground_truth'
results = []

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Starting Benchmark on {len(image_files)} images...")

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    txt_file = os.path.splitext(img_file)[0] + ".txt"
    gt_path = os.path.join(GT_DIR, txt_file)
    
    ground_truth = ""
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()

    # Run Tesseract
    pred_tess, time_tess = run_tesseract(img_path)
    cer_tess, _ = calculate_metrics(ground_truth, pred_tess)

    # Run PaddleOCR
    pred_paddle, time_paddle = run_paddle(img_path)
    cer_paddle, _ = calculate_metrics(ground_truth, pred_paddle)

    # Run TrOCR
    try:
        pred_trocr, time_trocr = run_trocr(img_path)
        cer_trocr, _ = calculate_metrics(ground_truth, pred_trocr)
    except Exception as e:
        print(f"TrOCR Failed on {img_file}: {e}")
        pred_trocr, time_trocr, cer_trocr = ("Error", 0, 1)

    results.append({
        "Image": img_file,
        "Tess_CER": round(cer_tess, 4), "Tess_Time": round(time_tess, 4),
        "Paddle_CER": round(cer_paddle, 4), "Paddle_Time": round(time_paddle, 4),
        "TrOCR_CER": round(cer_trocr, 4), "TrOCR_Time": round(time_trocr, 4)
    })

# Save and print
df = pd.DataFrame(results)
df.to_csv("ocr_benchmark_results.csv", index=False)
print("\n--- Final Summary (Average CER | Lower is Better) ---")
print(f"Tesseract: {df['Tess_CER'].mean():.4f}")
print(f"PaddleOCR: {df['Paddle_CER'].mean():.4f}")
print(f"TrOCR:     {df['TrOCR_CER'].mean():.4f}")
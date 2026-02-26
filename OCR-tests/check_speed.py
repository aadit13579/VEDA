import pandas as pd

# Load the results you already generated
df = pd.read_csv("ocr_benchmark_results.csv")

print("\n--- Average Inference Time (Seconds per Image) ---")
print(f"Tesseract: {df['Tess_Time'].mean():.4f}s")
print(f"PaddleOCR: {df['Paddle_Time'].mean():.4f}s")
print(f"TrOCR:     {df['TrOCR_Time'].mean():.4f}s")
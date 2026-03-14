"""Test with user's test.pdf - longer timeout for multi-page docs."""
import requests, time, json

BASE = "http://127.0.0.1:8000/api/v1"
FILE_PATH = "c:/Users/dines/OneDrive/Desktop/AM/VEDA/OCR-tests/test.pdf"

# 1. Upload
r = requests.post(f"{BASE}/upload", files={"file": open(FILE_PATH, "rb")})
job = r.json()
job_id = job["job_id"]
print(f"Uploaded -> job_id: {job_id}")

# 2. Poll status (max 120s for larger docs)
for i in range(24):
    time.sleep(5)
    s = requests.get(f"{BASE}/status/{job_id}").json()
    status = s.get("status", s.get("detail", "unknown"))
    print(f"  [{(i+1)*5}s] Status: {status}")
    if status == "Completed" or "Error" in str(status):
        break

# 3. Fetch result
res = requests.get(f"{BASE}/result/{job_id}")
if res.status_code == 200:
    data = res.json()
    print("\n=== EXTRACTED TEXT ===")
    print(data.get("text", "(no text field)"))
    print("\n=== LAYOUT INFO ===")
    for page in data.get("processed_data", {}).get("layout_data", []):
        layout = page.get('page_layout', '?')
        n_regions = len(page.get("regions", []))
        print(f"Page {page['page']} | Layout: {layout} | Regions: {n_regions}")
else:
    print(f"Result fetch failed: {res.json()}")

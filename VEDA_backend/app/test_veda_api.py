import requests
import sys
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_V1 = f"{BASE_URL}/api/v1"

def test_flow(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"🚀 Starting test flow for: {file_path}")

    # 1. Upload File (Ingest)
    print("\n1️⃣  Uploading file...")
    try:
        url = f"{API_V1}/upload"
        print(f"   POST {url}")
        
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            return
            
        data = response.json()
        print(f"✅ Upload successful!")
        file_id = data.get("file_id")
        category = data.get("category")
        print(f"   File ID: {file_id}")
        print(f"   Detected Category: {category}")

        if not file_id:
            print("❌ No file_id returned!")
            return

        # 2. Analyze Layout
        print(f"\n2️⃣  Requesting Layout Analysis for ID: {file_id}...")
        analyze_url = f"{API_V1}/analyze_layout/{file_id}"
        print(f"   POST {analyze_url}")
        
        response = requests.post(analyze_url)

        if response.status_code == 200:
            layout_data = response.json()
            print("✅ Layout Analysis Complete!")
            print(f"   Pages Processed: {layout_data.get('pages_processed')}")
            
            # Check for debug images
            if layout_data.get("layout_data"):
                first_page = layout_data["layout_data"][0]
                debug_url = first_page.get("debug_image_url")
                if debug_url:
                    full_debug_url = f"{BASE_URL}{debug_url}"
                    print(f"   Debug Image URL: {full_debug_url}")
                    print(f"   (Open this URL in your browser to see the bounding boxes)")
                # Print sample regions found
                regions = first_page.get("regions", [])
                if regions:
                    print(f"   Found {len(regions)} regions on page 1 (Example: {regions[0]['type']})")
        else:
            print(f"❌ Analysis failed: {response.text}")

    except Exception as e:
        print(f"❌ Error during request: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_veda_api.py <path_to_pdf_or_image>")
        print("Example: python test_veda_api.py sample_invoice.pdf")
    else:
        test_flow(sys.argv[1])

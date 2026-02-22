import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download
from utils.logger import get_logger
from doclayout_yolo import YOLOv10

# Get logger
logger = get_logger(__name__)

# --- CONFIGURATION ---
REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

# Global state to track loading status
MODEL_LOAD_ERROR = None

def load_layout_model():
    """
    Downloads key model weights if needed and loads the DocLayout-YOLO model.
    """
    global MODEL_LOAD_ERROR
    logger.info(f"🔄 Loading Layout Model from {REPO_ID}...")
    try:
        # Download the specific weights
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        logger.info(f"✅ Model downloaded to: {model_path}")
        
        # Load Model
        model = YOLOv10(model_path)
        logger.info("✅ DocLayout-YOLO Model Loaded Successfully!")
        MODEL_LOAD_ERROR = None
        return model
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(f"❌ {error_msg}")
        MODEL_LOAD_ERROR = error_msg
        
        # dummy fallback to prevent import crash, but analyze_layout will fail
        class Dummy:
            def predict(self, x, **k): return []
            names = {0: "dummy"}
        return Dummy()

# Initialize model once (Global singleton)
model = load_layout_model()


def pdf_to_images(pdf_bytes):
    """Converts PDF bytes to a list of OpenCV images (numpy arrays)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        # Render page to image (Zoom=2 for better OCR quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        # Convert PyMuPDF Pixmap -> Numpy Array (OpenCV format)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # Convert RGB to BGR (OpenCV expects BGR)
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)

    return images


def sort_boxes(boxes, y_tolerance=20):
    """
    Sorts bounding boxes in reading order (Top-Left -> Bottom-Right).
    Handles slight misalignments using y_tolerance.
    """
    # Key: Round Y to nearest 20px, then sort by X
    return sorted(
        boxes, key=lambda b: ((b["bbox"][1] // y_tolerance) * y_tolerance, b["bbox"][0])
    )


def analyze_layout(image_array):
    """
    Input: Single Page Image (OpenCV Array)
    Output: Sorted List of regions [{'type': 'Text', 'bbox': [...], 'confidence': ...}]
    """
    logger.info(f"Starting analyze_layout for image of shape {image_array.shape}")
    # Check for load error
    if MODEL_LOAD_ERROR:
        logger.error(f"Analyze layout failed: Model not loaded. Error: {MODEL_LOAD_ERROR}")
        raise RuntimeError(f"Model failed to load at startup: {MODEL_LOAD_ERROR}")

    # DocLayout-YOLO Inference
    # imgsz=1024 is recommended for this model
    logger.debug(f"Running inference on image {image_array.shape} with DocLayout-YOLO")
    results = model.predict(image_array, imgsz=1024, verbose=False, conf=0.25)
    
    detected_regions = []

    # Process Results
    for result in results:
        logger.debug(f"Found {len(result.boxes)} raw boxes from inference.")
        for box in result.boxes:
            # 1. Get Coordinates [x1, y1, x2, y2]
            coords = [int(x) for x in box.xyxy[0].tolist()]

            # 2. Get Class Name (Text, Table, Picture, Header...)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            detected_regions.append(
                {
                    "type": class_name,
                    "bbox": coords,
                    "confidence": confidence,
                }
            )

    # Sort standard reading order
    sorted_regions = sort_boxes(detected_regions)
    logger.info(f"Finished analyze_layout. Processed {len(sorted_regions)} regions.")

    return sorted_regions


def draw_layout_on_image(image, regions, output_path):
    """
    Draws bounding boxes and labels on the image and saves it.
    """
    # specific colors for different classes could be added here
    color = (0, 0, 255)  # Red for all for now
    thickness = 2

    # Make a copy to draw on
    debug_image = image.copy()

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        label = f"{region['type']} {region['confidence']:.2f}"

        # Draw Rectangle
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness)

        # Draw Label
        cv2.putText(
            debug_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

    # Save to disk
    if cv2.imwrite(output_path, debug_image):
        print(f"✅ Saved debug image to: {os.path.abspath(output_path)}")
    else:
        print(f"❌ Failed to save debug image to: {output_path}")

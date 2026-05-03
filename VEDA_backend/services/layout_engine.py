"""
VEDA Layout Engine.

Provides document layout analysis using the DocLayout-YOLO model. Converts
PDF pages to images, runs YOLO inference to detect text, table, figure, and
other regions, sorts them into preliminary reading order, and draws debug
bounding-box overlays for visual inspection.

This module exists as the bridge between raw document pages and the
structured region data consumed by downstream OCR, spatial sort, and
Gemini description stages.

Leverages: DocLayout-YOLO (YOLOv10), PyMuPDF (fitz), OpenCV (cv2),
           huggingface_hub for model weight download.
"""

import os

import cv2
import fitz
import numpy as np
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

from utils.logger import get_logger

logger = get_logger(__name__)


class LayoutEngine:
    """
    Document layout analysis engine backed by DocLayout-YOLO.

    Downloads model weights from HuggingFace Hub on first load, runs
    YOLO inference at 1024px resolution, and returns sorted region
    dictionaries with labels, bounding boxes, and confidence scores.

    Leverages: YOLOv10, hf_hub_download, fitz, cv2.
    """

    REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
    FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

    class _DummyModel:
        """
        Fallback model used when the real YOLO weights fail to load.

        Prevents import-time crashes and lets the application start,
        but analyze_layout will raise a RuntimeError if called.

        Leverages: None (stub only).
        """

        def predict(self, x, **k):
            return []

        names = {0: "dummy"}

    def __init__(self):
        """
        Download model weights (if needed) and load the YOLO model.

        On failure, installs a dummy model so the module can be imported
        without crashing; analyze_layout will raise at call time instead.

        Leverages: hf_hub_download, YOLOv10.
        """
        self._model_load_error = None
        self._model = self._load_model()

    def _load_model(self):
        """
        Perform the actual model download and initialization.

        Returns the loaded YOLOv10 model on success or a _DummyModel
        on failure, recording the error message for later reporting.

        Leverages: hf_hub_download, YOLOv10.
        """
        logger.info(f"🔄 Loading Layout Model from {self.REPO_ID}...")
        try:
            model_path = hf_hub_download(
                repo_id=self.REPO_ID, filename=self.FILENAME
            )
            logger.info(f"✅ Model downloaded to: {model_path}")

            model = YOLOv10(model_path)
            logger.info("✅ DocLayout-YOLO Model Loaded Successfully!")
            self._model_load_error = None
            return model
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(f"❌ {error_msg}")
            self._model_load_error = error_msg
            return self._DummyModel()

    def pdf_to_images(self, pdf_bytes) -> list:
        """
        Convert PDF bytes to a list of OpenCV images (numpy arrays).

        Each page is rendered at 2x zoom for better downstream OCR quality.

        Leverages: fitz.open, page.get_pixmap, cv2.cvtColor.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []

        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)

        return images

    def sort_boxes(self, boxes: list, y_tolerance: int = 20) -> list:
        """
        Sort bounding boxes in reading order (top-left to bottom-right).

        Groups boxes by rounding their Y coordinate to the nearest
        y_tolerance pixels, then sorts within each row by X.

        Leverages: Python sorted with composite key.
        """
        return sorted(
            boxes,
            key=lambda b: (
                (b["bbox"][1] // y_tolerance) * y_tolerance,
                b["bbox"][0],
            ),
        )

    def analyze_layout(self, image_array: np.ndarray) -> list:
        """
        Run YOLO layout analysis on a single page image.

        Returns a sorted list of region dicts, each containing 'label',
        'bbox' [x1,y1,x2,y2], 'confidence', and 'id'. Regions labeled
        'abandon' are filtered out.

        Raises RuntimeError if the model failed to load at startup.

        Leverages: YOLOv10.predict at imgsz=1024.
        """
        logger.info(f"Starting analyze_layout for image of shape {image_array.shape}")

        if self._model_load_error:
            logger.error(
                f"Analyze layout failed: Model not loaded. Error: {self._model_load_error}"
            )
            raise RuntimeError(
                f"Model failed to load at startup: {self._model_load_error}"
            )

        logger.debug(
            f"Running inference on image {image_array.shape} with DocLayout-YOLO"
        )
        results = self._model.predict(
            image_array, imgsz=1024, verbose=False, conf=0.25
        )

        detected_regions = []

        for result in results:
            logger.debug(f"Found {len(result.boxes)} raw boxes from inference.")
            for box in result.boxes:
                coords = [int(x) for x in box.xyxy[0].tolist()]

                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]

                if class_name.lower() == "abandon":
                    continue

                confidence = float(box.conf[0])

                detected_regions.append(
                    {
                        "label": class_name,
                        "bbox": coords,
                        "confidence": round(confidence, 2),
                    }
                )

        sorted_regions = self.sort_boxes(detected_regions)

        for idx, region in enumerate(sorted_regions, start=1):
            region["id"] = f"r{idx}"

        logger.info(
            f"Finished analyze_layout. Processed {len(sorted_regions)} regions."
        )

        return sorted_regions

    def draw_layout_on_image(
        self, image: np.ndarray, regions: list, output_path: str
    ) -> None:
        """
        Draw bounding boxes and labels on an image and save to disk.

        Used to generate debug images that the frontend can display
        for visual verification of the layout analysis output.

        Leverages: cv2.rectangle, cv2.putText, cv2.imwrite.
        """
        color = (0, 0, 255)
        thickness = 2

        debug_image = image.copy()

        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            label = f"{region['label']} {region['confidence']:.2f}"

            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness)

            cv2.putText(
                debug_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        if cv2.imwrite(output_path, debug_image):
            logger.info(f"✅ Saved debug image to: {os.path.abspath(output_path)}")
        else:
            logger.error(f"❌ Failed to save debug image to: {output_path}")


_instance = LayoutEngine()

pdf_to_images = _instance.pdf_to_images
sort_boxes = _instance.sort_boxes
analyze_layout = _instance.analyze_layout
draw_layout_on_image = _instance.draw_layout_on_image

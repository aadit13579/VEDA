"""
VEDA Gemini Vision Engine.

Provides AI-powered image description and OCR fallback using Google's
Gemini generative model. Crops image regions from document pages,
gathers contextual text via spatial proximity scoring, sends the
combined image + context to Gemini, and caches responses in Redis.

This module exists to give visually complex regions (figures, tables,
formulas) rich textual descriptions that pure OCR cannot produce,
enabling meaningful text-to-speech output for visual content.

Leverages: google-generativeai (Gemini 2.5 Flash), PyMuPDF (fitz),
           OpenCV (cv2), Pillow (PIL), python-dotenv, Redis.
"""

import os
import io
import glob

import cv2
import fitz
import numpy as np
from PIL import Image
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

from services.redis_client import get_page, set_page, bbox_matches
from services.ocr_engine import extract_text_from_region
from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialScorer:
    """
    Scores document regions by spatial proximity to a target image region.

    Combines Manhattan distance, type-based weights, column alignment,
    and vertical direction bias to rank which text regions are most
    likely to be contextually relevant to a given image.

    This class exists to isolate the geometric scoring logic from the
    Gemini API interaction, making it independently testable.

    Leverages: Python math (abs, min, max).
    """

    TYPE_WEIGHTS: dict[str, float] = {
        "figure_caption": 300,
        "table_caption": 300,
        "caption": 300,
        "title": 150,
        "section_header": 120,
        "section-header": 120,
        "text": 50,
        "plain_text": 50,
        "paragraph": 50,
        "list": 40,
    }

    COLUMN_ALIGN_BONUS = 100
    VERTICAL_BELOW_BONUS = 40
    VERTICAL_ABOVE_BONUS = 20

    @staticmethod
    def box_center(bbox: List[int]) -> tuple[float, float]:
        """
        Compute the center (cx, cy) of a bounding box [x1, y1, x2, y2].

        Leverages: Python arithmetic.
        """
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def manhattan_distance(a: List[int], b: List[int]) -> float:
        """
        Compute Manhattan distance between the centers of two bounding boxes.

        Leverages: SpatialScorer.box_center.
        """
        cx_a, cy_a = SpatialScorer.box_center(a)
        cx_b, cy_b = SpatialScorer.box_center(b)
        return abs(cx_a - cx_b) + abs(cy_a - cy_b)

    @staticmethod
    def horizontal_overlap(a: List[int], b: List[int]) -> float:
        """
        Compute horizontal pixel overlap between two bounding boxes.

        Returns zero if the boxes do not overlap horizontally.

        Leverages: Python min/max.
        """
        return max(0, min(a[2], b[2]) - max(a[0], b[0]))

    @staticmethod
    def score_region(
        region: dict,
        image_bbox: List[int],
        image_width: float,
        page_diag: float,
    ) -> float:
        """
        Compute a relevance score for a region relative to the image.

        Combines normalized proximity, type weight, column alignment
        bonus, and vertical direction bias into a single score.

        Leverages: manhattan_distance, horizontal_overlap, TYPE_WEIGHTS.
        """
        r_bbox = region.get("bbox", [0, 0, 0, 0])

        dist = SpatialScorer.manhattan_distance(r_bbox, image_bbox)
        dist_norm = dist / page_diag if page_diag > 0 else dist
        score = -dist_norm * 1000

        label = region.get("label", "").lower().replace(" ", "_")
        score += SpatialScorer.TYPE_WEIGHTS.get(label, 0)

        overlap = SpatialScorer.horizontal_overlap(r_bbox, image_bbox)
        if image_width > 0 and (overlap / image_width) > 0.3:
            score += SpatialScorer.COLUMN_ALIGN_BONUS

        img_y1, img_y2 = image_bbox[1], image_bbox[3]
        r_y1, r_y2 = r_bbox[1], r_bbox[3]

        if r_y1 >= img_y2:
            score += SpatialScorer.VERTICAL_BELOW_BONUS
        elif r_y2 <= img_y1:
            score += SpatialScorer.VERTICAL_ABOVE_BONUS

        return score


class GeminiEngine:
    """
    Gemini-powered image description and OCR fallback engine.

    Manages the Gemini API connection, image preprocessing, spatial
    context gathering, prompt construction, API calls, and Redis
    caching of responses.

    This class exists to encapsulate all Gemini interactions behind a
    clean interface that the pipeline and router layers can call
    without managing API keys, image formats, or prompt engineering.

    Leverages: google-generativeai, PIL, cv2, fitz, Redis client.
    """

    STORAGE_DIR = "storage"
    MAX_IMAGE_DIM = 512
    JPEG_QUALITY = 50

    GEMINI_PROMPT = (
        "You are an expert in analyzing scientific and technical documents.\n"
        "Describe the image clearly using the caption and nearby text.\n"
        "Focus on what the image shows and its role in the document.\n"
        "Use bullet points if helpful."
    )

    def __init__(self):
        """
        Load the Gemini API key from .env and configure the model.

        Logs a warning if the key is missing — Gemini calls will fail
        at runtime rather than at import time.

        Leverages: python-dotenv, google.generativeai.configure.
        """
        _env_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", ".env"
        )
        load_dotenv(dotenv_path=os.path.abspath(_env_path))

        _api_key = os.getenv("GENERATIVE_LANGUAGE_KEY")
        if not _api_key:
            logger.warning(
                "GENERATIVE_LANGUAGE_KEY not found in .env — Gemini calls will fail."
            )
        else:
            genai.configure(api_key=_api_key)
            logger.info("Gemini API configured successfully.")

        self._model = genai.GenerativeModel("gemini-2.5-flash")

    def _load_page_image(self, file_id: str, page: int) -> np.ndarray:
        """
        Load the original page as an OpenCV image from disk.

        Handles both raw image files and multi-page PDFs. Raises
        FileNotFoundError if no file matches the file_id, or
        ValueError if the page index is out of range.

        Leverages: cv2.imread, fitz.open, glob.
        """
        search_pattern = os.path.join(self.STORAGE_DIR, f"{file_id}.*")
        files = [f for f in glob.glob(search_pattern) if not f.endswith("_final.json")]

        if not files:
            raise FileNotFoundError(
                f"No file found on disk for file_id '{file_id}'"
            )

        file_path = files[0]

        img = cv2.imread(file_path)
        if img is not None:
            return img

        doc = fitz.open(file_path)
        page_index = page - 1
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(
                f"Page {page} does not exist. Document has {len(doc)} page(s)."
            )

        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        return cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    def crop_and_preprocess(self, image: np.ndarray, bbox: List[int]) -> Image.Image:
        """
        Crop a bounding box from an OpenCV image and prepare it for Gemini.

        Clamps coordinates to image bounds, converts BGR to RGB, resizes
        to MAX_IMAGE_DIM, and re-encodes as low-quality JPEG to minimize
        token usage.

        Leverages: cv2.cvtColor, PIL Image.thumbnail, JPEG encoding.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = image[y1:y2, x1:x2]

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pil_img.thumbnail((self.MAX_IMAGE_DIM, self.MAX_IMAGE_DIM), Image.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.JPEG_QUALITY, optimize=True)
        buf.seek(0)
        return Image.open(buf)

    def gather_context(
        self,
        file_id: str,
        page: int,
        image_bbox: List[int],
        full_image: np.ndarray,
        top_k: int = 3,
    ) -> dict:
        """
        Gather caption and surrounding text context for an image region.

        Uses spatial proximity scoring to find the most relevant caption
        and top-k text regions. Performs OCR on any regions missing text
        and updates Redis with the results.

        Returns a dict with 'caption' (str or None) and 'context_text'.

        Leverages: SpatialScorer, Redis client, OCR engine.
        """
        page_data = get_page(file_id, page)
        if page_data is None:
            return {"caption": None, "context_text": ""}

        regions = page_data.get("regions", [])
        image_width = float(image_bbox[2] - image_bbox[0])

        if regions:
            all_bboxes = [
                r.get("bbox", [0, 0, 0, 0])
                for r in regions
                if len(r.get("bbox", [])) == 4
            ]
            if all_bboxes:
                page_width = max(b[2] for b in all_bboxes)
                page_height = max(b[3] for b in all_bboxes)
            else:
                page_width, page_height = 1, 1
        else:
            page_width, page_height = 1, 1

        page_diag = (page_width**2 + page_height**2) ** 0.5

        caption = None
        best_caption_region = None
        caption_labels = {"figure_caption", "table_caption", "caption"}
        best_caption_score = float("-inf")

        for r in regions:
            label = r.get("label", "").lower().replace(" ", "_")
            if label not in caption_labels:
                continue

            r_bbox = r.get("bbox", [])
            if len(r_bbox) != 4:
                continue

            score = SpatialScorer.score_region(r, image_bbox, image_width, page_diag)
            if score > best_caption_score:
                best_caption_score = score
                best_caption_region = r

        scored_regions: list[tuple[float, dict]] = []
        visual_labels = {"figure", "image", "picture"}

        for r in regions:
            r_bbox = r.get("bbox", [])
            if len(r_bbox) != 4:
                continue

            label = r.get("label", "").lower().replace(" ", "_")

            if bbox_matches(r_bbox, image_bbox):
                continue

            if label in caption_labels:
                continue

            if label in visual_labels:
                continue

            score = SpatialScorer.score_region(r, image_bbox, image_width, page_diag)
            scored_regions.append((score, r))

        scored_regions.sort(key=lambda pair: pair[0], reverse=True)
        top_k_regions = [r for _, r in scored_regions[:top_k]]

        top_k_regions.sort(
            key=lambda r: r.get("reading_order", r.get("bbox", [0, 0])[1])
        )

        redis_needs_update = False

        if best_caption_region:
            if not best_caption_region.get("text"):
                ocr_text = extract_text_from_region(
                    full_image, best_caption_region["bbox"]
                )
                if ocr_text:
                    best_caption_region["text"] = ocr_text
                    redis_needs_update = True
            caption = best_caption_region.get("text", "").strip() or None

        for r in top_k_regions:
            if not r.get("text"):
                ocr_text = extract_text_from_region(full_image, r["bbox"])
                if ocr_text:
                    r["text"] = ocr_text
                    redis_needs_update = True

        if redis_needs_update:
            set_page(file_id, page, page_data)
            logger.info(
                f"Gather context performed OCR on missing regions. "
                f"Updated Redis for page {page}."
            )

        context_text = "\n".join(
            r.get("text", "").strip() for r in top_k_regions if r.get("text")
        )

        return {
            "caption": caption,
            "context_text": context_text,
        }

    def describe_image(
        self,
        file_id: str,
        page: int,
        bbox: List[int],
        top_k: int = 8,
    ) -> dict:
        """
        Full image description pipeline: cache check, crop, context, Gemini call.

        Steps:
          1. Check Redis for a cached Gemini response
          2. Load the page image from disk
          3. Crop and preprocess the image region
          4. Gather caption + context via spatial scoring
          5. Send image + context + prompt to Gemini
          6. Cache the response in Redis and return

        Returns a dict with caption, context_text, and gemini_response.

        Leverages: Redis cache, crop_and_preprocess, gather_context, Gemini API.
        """
        page_data = get_page(file_id, page)
        target_region = None
        if page_data:
            for r in page_data.get("regions", []):
                if bbox_matches(r.get("bbox", []), bbox):
                    target_region = r
                    break

            if target_region and target_region.get("gemini_response"):
                logger.info(f"Redis Cache HIT for Gemini explanation of bbox {bbox}")
                return {
                    "caption": target_region.get("gemini_caption"),
                    "context_text": target_region.get("gemini_context_text"),
                    "gemini_response": target_region.get("gemini_response"),
                }

        full_image = self._load_page_image(file_id, page)

        pil_image = self.crop_and_preprocess(full_image, bbox)
        logger.info(
            f"Image cropped & preprocessed: {pil_image.size}, mode={pil_image.mode}"
        )

        ctx = self.gather_context(file_id, page, bbox, full_image, top_k)
        caption = ctx["caption"]
        context_text = ctx["context_text"]

        text_parts = []
        if context_text:
            text_parts.append(f"### Surrounding Text Context:\n{context_text}")
        if caption:
            text_parts.append(f"### Image Caption:\n{caption}")
        text_parts.append(f"### Instruction:\n{self.GEMINI_PROMPT}")

        full_prompt = "\n\n".join(text_parts)

        logger.info("--- GEMINI FULL PROMPT START ---")
        logger.info(full_prompt)
        logger.info(
            f"Image attached: size={pil_image.size}, format={pil_image.format}"
        )
        logger.info("--- GEMINI FULL PROMPT END ---")

        response = self._model.generate_content([full_prompt, pil_image])
        gemini_text = response.text

        logger.info("--- GEMINI RAW RESPONSE START ---")
        logger.info(gemini_text)
        logger.info("--- GEMINI RAW RESPONSE END ---")

        if target_region and page_data:
            target_region["gemini_response"] = gemini_text
            target_region["gemini_caption"] = caption
            target_region["gemini_context_text"] = context_text
            set_page(file_id, page, page_data)
            logger.info(f"Saved Gemini response to Redis for bbox {bbox}.")

        return {
            "caption": caption,
            "context_text": context_text,
            "gemini_response": gemini_text,
        }

    def extract_text_with_gemini(self, full_image: np.ndarray, bbox: List[int]) -> str:
        """
        Fallback OCR using Gemini when Tesseract is missing or fails.

        Crops and preprocesses the image region, sends it to Gemini with
        a text-extraction-only prompt, and returns the raw text.

        Leverages: crop_and_preprocess, Gemini API.
        """
        try:
            pil_image = self.crop_and_preprocess(full_image, bbox)
            prompt = (
                "Extract all readable text from this image exactly as written. "
                "Only return the extracted text, no commentary. "
                "If there's no text, return nothing."
            )
            response = self._model.generate_content([prompt, pil_image])
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini OCR fallback failed: {e}")
            return ""


_instance = GeminiEngine()

crop_and_preprocess = _instance.crop_and_preprocess
gather_context = _instance.gather_context
describe_image = _instance.describe_image
extract_text_with_gemini = _instance.extract_text_with_gemini

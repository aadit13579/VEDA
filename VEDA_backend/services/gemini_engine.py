"""
Gemini Vision Engine for VEDA Backend.

Responsible for:
  - Cropping an image region from a document page
  - Gathering context (caption + surrounding text) from Redis
  - Preprocessing the image for minimal token usage
  - Sending the image + context to Gemini and returning the response
"""

import os
import io
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

from services.redis_client import get_page, set_page, bbox_matches
from services.ocr_engine import extract_text_from_region
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------- Gemini Configuration ----------

# Load .env from the parent VEDA directory
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path))

_api_key = os.getenv("GENERATIVE_LANGUAGE_KEY")
if not _api_key:
    logger.warning(
        "GENERATIVE_LANGUAGE_KEY not found in .env — Gemini calls will fail."
    )
else:
    genai.configure(api_key=_api_key)
    logger.info("Gemini API configured successfully.")

# Model singleton — gemini-2.0-flash for speed + vision
_model = genai.GenerativeModel("gemini-2.5-flash")


# ---------- Image Helpers ----------

STORAGE_DIR = "storage"
# Maximum dimension (px) for the image sent to Gemini — keeps token count low
MAX_IMAGE_DIM = 512
JPEG_QUALITY = 50  # Lower quality = fewer tokens


def _load_page_image(file_id: str, page: int) -> np.ndarray:
    """
    Load the original page as an OpenCV image.

    Handles both raw images and multi-page PDFs.
    Raises FileNotFoundError / ValueError on failure.
    """
    import glob

    search_pattern = os.path.join(STORAGE_DIR, f"{file_id}.*")
    files = [f for f in glob.glob(search_pattern) if not f.endswith("_final.json")]

    if not files:
        raise FileNotFoundError(f"No file found on disk for file_id '{file_id}'")

    file_path = files[0]

    # Try reading as a plain image first
    img = cv2.imread(file_path)
    if img is not None:
        return img

    # Fallback: read as PDF
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


def crop_and_preprocess(image: np.ndarray, bbox: List[int]) -> Image.Image:
    """
    Crop bbox from the OpenCV image, resize, and convert to a
    low-quality PIL Image suitable for Gemini (minimal tokens).
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clamp all coordinates to image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))  # Ensure x2 > x1
    y2 = max(y1 + 1, min(y2, h))  # Ensure y2 > y1

    cropped = image[y1:y2, x1:x2]

    # Convert BGR → RGB → PIL
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Resize keeping aspect ratio — cap longest side at MAX_IMAGE_DIM
    pil_img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)

    # Re-encode as JPEG at low quality to reduce token usage
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    return Image.open(buf)


# ---------- Spatial Scoring for Context Gathering ----------

# Type-based weight bonuses   (higher = more relevant)
_TYPE_WEIGHTS: dict[str, float] = {
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

# Bonus when a region is in the same column as the image
_COLUMN_ALIGN_BONUS = 100


def _box_center(bbox: List[int]) -> tuple[float, float]:
    """Compute the center (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _manhattan_distance(a: List[int], b: List[int]) -> float:
    """Manhattan distance between the centers of two bounding boxes."""
    cx_a, cy_a = _box_center(a)
    cx_b, cy_b = _box_center(b)
    return abs(cx_a - cx_b) + abs(cy_a - cy_b)


def _horizontal_overlap(a: List[int], b: List[int]) -> float:
    """
    Horizontal overlap between two bounding boxes (in pixels).

    overlap_x = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
    """
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


# Vertical direction bias (below > above > side)
_VERTICAL_BELOW_BONUS = 40
_VERTICAL_ABOVE_BONUS = 20


def _score_region(
    region: dict,
    image_bbox: List[int],
    image_width: float,
    page_diag: float,
) -> float:
    """
    Compute a relevance score for a region relative to the image.

    score = -dist_norm * 1000 + w_type + w_column + w_vertical

    Distance is normalized by the page diagonal so that weights
    have a stable, meaningful scale across different page sizes.

    Args:
        region:      Region dict with 'bbox', 'label', etc.
        image_bbox:  The image's bounding box.
        image_width: Width of the image bbox (for normalizing overlap).
        page_diag:   Diagonal length of the page (for normalizing distance).
    """
    r_bbox = region.get("bbox", [0, 0, 0, 0])

    # 1. Proximity: normalized Manhattan distance (closer = higher score)
    dist = _manhattan_distance(r_bbox, image_bbox)
    dist_norm = dist / page_diag if page_diag > 0 else dist
    score = -dist_norm * 1000

    # 2. Type weight
    label = region.get("label", "").lower().replace(" ", "_")
    score += _TYPE_WEIGHTS.get(label, 0)

    # 3. Column alignment bonus
    overlap = _horizontal_overlap(r_bbox, image_bbox)
    if image_width > 0 and (overlap / image_width) > 0.3:
        # >30% horizontal overlap => same column
        score += _COLUMN_ALIGN_BONUS

    # 4. Vertical direction bias
    img_y1, img_y2 = image_bbox[1], image_bbox[3]
    r_y1, r_y2 = r_bbox[1], r_bbox[3]

    if r_y1 >= img_y2:  # region is BELOW the image
        score += _VERTICAL_BELOW_BONUS
    elif r_y2 <= img_y1:  # region is ABOVE the image
        score += _VERTICAL_ABOVE_BONUS
    # else: region is to the side — no bonus

    return score


def gather_context(
    file_id: str,
    page: int,
    image_bbox: List[int],
    full_image: np.ndarray,
    top_k: int = 3,
) -> dict:
    """
    Gather caption and surrounding text context for an image region
    using spatial proximity scoring.

    Algorithm:
      1. Represent each region as a geometric bounding box
      2. Compute Manhattan distance between region center and image center
      3. Check column alignment via horizontal overlap
      4. Score = -distance + w_type + w_column
      5. Select top-k scoring regions, order by reading_order

    Returns:
        {
          "caption": str | None,
          "context_text": str,      # concatenated text from top-k regions
        }
    """
    page_data = get_page(file_id, page)
    if page_data is None:
        return {"caption": None, "context_text": ""}

    regions = page_data.get("regions", [])
    image_width = float(image_bbox[2] - image_bbox[0])

    # Compute page diagonal from the bounding area of all regions
    if regions:
        all_bboxes = [
            r.get("bbox", [0, 0, 0, 0]) for r in regions if len(r.get("bbox", [])) == 4
        ]
        if all_bboxes:
            page_width = max(b[2] for b in all_bboxes)
            page_height = max(b[3] for b in all_bboxes)
        else:
            page_width, page_height = 1, 1
    else:
        page_width, page_height = 1, 1

    page_diag = (page_width**2 + page_height**2) ** 0.5

    # --- Find best caption (highest-scoring caption-type region) ---
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

        score = _score_region(r, image_bbox, image_width, page_diag)
        if score > best_caption_score:
            best_caption_score = score
            best_caption_region = r

    # --- Score all non-caption regions ---
    scored_regions: list[tuple[float, dict]] = []
    visual_labels = {"figure", "image", "picture"}

    for r in regions:
        r_bbox = r.get("bbox", [])
        if len(r_bbox) != 4:
            continue

        label = r.get("label", "").lower().replace(" ", "_")

        # Skip the image region itself
        if bbox_matches(r_bbox, image_bbox):
            continue

        # Skip caption regions (already captured above)
        if label in caption_labels:
            continue

        # Skip other visual elements (we want text context, not another image)
        if label in visual_labels:
            continue

        score = _score_region(r, image_bbox, image_width, page_diag)
        scored_regions.append((score, r))

    # --- Select top-k by score ---
    scored_regions.sort(key=lambda pair: pair[0], reverse=True)
    top_k_regions = [r for _, r in scored_regions[:top_k]]

    # --- Order selected regions by reading_order for coherent text ---
    top_k_regions.sort(key=lambda r: r.get("reading_order", r.get("bbox", [0, 0])[1]))

    # --- Fetch OCR text if missing, and save to Redis ---
    redis_needs_update = False

    # 1. Process caption
    if best_caption_region:
        if not best_caption_region.get("text"):
            ocr_text = extract_text_from_region(full_image, best_caption_region["bbox"])
            if ocr_text:
                best_caption_region["text"] = ocr_text
                redis_needs_update = True
        caption = best_caption_region.get("text", "").strip() or None

    # 2. Process context text
    for r in top_k_regions:
        if not r.get("text"):
            ocr_text = extract_text_from_region(full_image, r["bbox"])
            if ocr_text:
                r["text"] = ocr_text
                redis_needs_update = True

    if redis_needs_update:
        set_page(file_id, page, page_data)
        logger.info(
            f"Gather context performed OCR on missing regions. Updated Redis for page {page}."
        )

    context_text = "\n".join(
        r.get("text", "").strip() for r in top_k_regions if r.get("text")
    )

    return {
        "caption": caption,
        "context_text": context_text,
    }


GEMINI_PROMPT = """
You are an expert in analyzing scientific and technical documents.
Describe the image clearly using the caption and nearby text.
Focus on what the image shows and its role in the document.
Use bullet points if helpful.
""".strip()


def describe_image(
    file_id: str,
    page: int,
    bbox: List[int],
    top_k: int = 8,
) -> dict:
    """
    Full pipeline:
      1. Check Redis to see if we already analyzed this image
      2. Load page image from disk
      3. Crop + preprocess the image region
      4. Gather caption + context from Redis (spatial scoring, top-k)
      5. Send everything to Gemini
      6. Cache response in Redis and return

    Args:
        file_id:  Document identifier.
        page:     1-indexed page number.
        bbox:     [x1, y1, x2, y2] bounding box of the image region.
        top_k:    Number of spatially-scored regions to include as context.

    Returns:
        dict with caption, context_text, gemini_response.
    """
    # 1. Check Redis cache first
    page_data = get_page(file_id, page)
    target_region = None
    if page_data:
        for r in page_data.get("regions", []):
            if bbox_matches(r.get("bbox", []), bbox):
                target_region = r
                break

        # If cache hit, return immediately
        if target_region and target_region.get("gemini_response"):
            logger.info(f"Redis Cache HIT for Gemini explanation of bbox {bbox}")
            return {
                "caption": target_region.get("gemini_caption"),
                "context_text": target_region.get("gemini_context_text"),
                "gemini_response": target_region.get("gemini_response"),
            }

    # 2. Load full page image
    full_image = _load_page_image(file_id, page)

    # 2. Crop and preprocess
    pil_image = crop_and_preprocess(full_image, bbox)
    logger.info(
        f"Image cropped & preprocessed: {pil_image.size}, mode={pil_image.mode}"
    )

    # 3. Gather context via spatial scoring
    ctx = gather_context(file_id, page, bbox, full_image, top_k)
    caption = ctx["caption"]
    context_text = ctx["context_text"]

    # 4. Build the Gemini prompt parts
    text_parts = []
    if context_text:
        text_parts.append(f"### Surrounding Text Context:\n{context_text}")
    if caption:
        text_parts.append(f"### Image Caption:\n{caption}")
    text_parts.append(f"### Instruction:\n{GEMINI_PROMPT}")

    full_prompt = "\n\n".join(text_parts)

    logger.info("--- GEMINI FULL PROMPT START ---")
    logger.info(full_prompt)
    logger.info(f"Image attached: size={pil_image.size}, format={pil_image.format}")
    logger.info("--- GEMINI FULL PROMPT END ---")

    # 5. Call Gemini
    response = _model.generate_content([full_prompt, pil_image])
    gemini_text = response.text

    logger.info("--- GEMINI RAW RESPONSE START ---")
    logger.info(gemini_text)
    logger.info("--- GEMINI RAW RESPONSE END ---")

    # 6. Save response back to the image region in Redis
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


def extract_text_with_gemini(full_image: np.ndarray, bbox: List[int]) -> str:
    """
    Fallback OCR using Gemini 1.5 Flash when Tesseract is missing.
    """
    try:
        pil_image = crop_and_preprocess(full_image, bbox)
        prompt = "Extract all readable text from this image exactly as written. Only return the extracted text, no commentary. If there's no text, return nothing."
        response = _model.generate_content([prompt, pil_image])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR fallback failed: {e}")
        return ""


# VEDA: Visual Engine for Document Accessibility

VEDA is a robust, modular, and state-driven pipeline designed for complex document processing and semantic analysis. It breaks down scanned digital documents and multi-page PDFs, analyzes their structural layout, accurately determines reading order through geometry, dynamically performs Optical Character Recognition (OCR), and enriches visual elements (like charts or figures) using Large Vision-Language Models (LVLMs) like Gemini.

Current State: **Backend Core Complete**

## Architecture Pipeline

The VEDA Backend operates as a series of disconnected, independent FastAPI micro-routers. Rather than forcing a file through a rigid in-memory sequence of functions, each stage runs independently and interacts with a centralized **Redis Session State**.

1. **Ingest (Upload & Identify):**
   - Receives raw PDFs or images (`POST /upload`).
   - Determines if the file is a digital PDF or a scanned PDF.
   - Saves the file to disk and generates a unique `file_id`.

2. **Layout Analysis:**
   - Powered by **DocLayout-YOLO**.
   - Takes an image/page and generates bounding boxes for regions like `title`, `text`, `figure`, `table`, `caption`, etc.
   - Outputs the raw geometrical structure to Redis.

3. **Spatial Sorting (Reading Order):**
   - Implements a dynamic **Recursive X-Y Cut** algorithm.
   - Calculates the structural gaps between regions to dynamically detect if a page is a single column or multi-column layout.
   - Reorders the bounding boxes into a logical human `reading_order` and updates the Redis state.

4. **Dynamic Context Gathering & OCR:**
   - Powered by **Tesseract OCR**.
   - Traditional pipelines OCR entire pages upfront. VEDA targets specific bounding boxes. 

5. **LVLM Visual Explanation (Gemini Engine):**
   - Powered by **Gemini 2.5 Flash**.
   - Accepts a single image bounding box, gathers the surrounding contextual text natively, and asks Gemini to provide a comprehensive explanation of the visual material, factoring in the document's surrounding text.

6. **Finalization:**
   - Aggregates the Redis cache for all pages.
   - Compiles a final JSON structural map of the document.
   - Cleans up temporary Redis keys.

---

## What Makes VEDA Unique?

VEDA departs from traditional linear document parsers in three distinct ways:

### 1. Redis-Backed Distributed State & BBox Tolerance
Intermediate steps don't pass massive JSON blobs to one another. Each module reads from and writes to **Redis**. 
Because ML models are imperfect, VEDA implements a `bbox_matches(a, b, tolerance=5)` algorithm. This allows independent pipeline stages (like Spatial Sort or OCR) to fetch a region, update its content (e.g., adding `reading_order` or `text`), and merge it back into the cached Redis layout even if the bounding box coordinates drift slightly between transformations.

### 2. Mathematics-Driven Image Contextualization (Spatial Scoring)
When sending an extracted diagram to an AI like Gemini for explanation, passing the *entire* text of a page breaks token limits and causes hallucination. VEDA uses a rigorous **Spatial Proximity Scoring** algorithm to select only the most relevant text context for the image:
*   **Normalized Distance**: It calculates the Manhattan distance between the center of the image and the center of every text block, normalized by the page's diagonal. *This ensures distance-weighting scales flawlessly whether it's a massive A3 scan or a mobile screenshot.*
*   **Column Alignment Check**: It identifies horizontal overlap. If a text block shares >30% horizontal space with the image, it's flagged as being in the "same column" and receives a massive relevance boost (+100).
*   **Directional Vertical Bias**: Humans put captions *below* images and explanations *above* them. VEDA adds a +40 bonus to text located below the image, and a +20 bonus to text above, completely ignoring text floating off to the side in adjacent columns.

### 3. "Just-In-Time" (Lazy) OCR Evaluation
Traditional pipelines OCR an entire page, which is tremendously slow and computationally expensive. VEDA does not. 
When the Spatial Scoring engine identifies the top-K text boxes that correspond to an image, it checks if those boxes have been OCR'd yet.
*   If they haven't (e.g. the user just uploaded the file and immediately clicked the image), VEDA transparently intercepts the request.
*   It fires up the local Tesseract engine, crops *only* those specific geometric bounding boxes, OCRs only the exact context lines needed, and saves it permanently to Redis.
*   It then forwards the context to Gemini. 

This **Lazy Evaluation** ensures that VEDA does zero wasted mathematical operations and enables instant, real-time interactions with documents without waiting ages for full-page processing. Furthermore, because it immediately caches both the OCR result and the final Gemini response in Redis, asking for the same explanation twice returns instantly (0ms logic overhead).

---

## Tech Stack
*   **Backend Framework**: FastAPI (Python)
*   **Machine Learning**: Ultralytics (DocLayout-YOLO), OpenCV, NumPy
*   **OCR Engine**: Tesseract OCR
*   **LLM Engine**: Google GenAI SDK (Gemini 2.0 Flash)
*   **Data Persistence**: Redis (Dockerized), JSON Storage
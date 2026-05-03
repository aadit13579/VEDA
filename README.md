# VEDA: Visual Engine for Document Accessibility

VEDA is a robust, modular, and state-driven pipeline designed for complex document processing and semantic analysis. It breaks down scanned digital documents and multi-page PDFs, analyzes their structural layout, accurately determines reading order through geometry, dynamically performs Optical Character Recognition (OCR), and enriches visual elements (like charts or figures) using Large Vision-Language Models (LVLMs) like Gemini.

Current State: **Class-Based Architecture & Streaming Pipeline Complete**

## Architecture Pipeline

The VEDA Backend operates as a series of independent FastAPI micro-routers built upon a **strict class-based, SOLID architecture**. Rather than forcing a file through a rigid in-memory sequence of functions, it utilizes a highly-concurrent, page-by-page **Server-Sent Events (SSE) Streaming Pipeline** backed by a centralized **Redis Session State**.

1. **Ingest (Upload & Identify):**
   - Receives raw PDFs or images (`POST /pipeline/start`).
   - Determines if the file is a digital PDF or a scanned PDF.
   - Generates a unique `file_id` and immediately returns to the client to establish a streaming connection.

2. **Layout Analysis:**
   - Powered by **DocLayout-YOLO**.
   - Takes a page and generates bounding boxes for regions like `title`, `text`, `figure`, `table`, `caption`, etc.
   - Outputs the raw geometrical structure to Redis.

3. **Spatial Sorting (Reading Order):**
   - Implements a dynamic **Recursive X-Y Cut** algorithm.
   - Calculates the structural gaps between regions to dynamically detect if a page is a single column or multi-column layout.
   - Reorders the bounding boxes into a logical human `reading_order`.

4. **Highly Concurrent Region Processing (OCR & Gemini):**
   - Within each page, *all* regions are processed concurrently using asynchronous thread pools.
   - **Text Regions:** Uses a fallback chain: PyMuPDF (native PDF text) → Tesseract OCR → Gemini OCR fallback.
   - **Visual Regions:** Handled by Gemini 2.5 Flash. It accepts an image bounding box, gathers surrounding contextual text natively, and asks Gemini to provide a comprehensive explanation of the visual material.

5. **SSE Streaming & Finalization:**
   - As each page completes processing, a `page_ready` event is pushed to the frontend via Server-Sent Events, enabling instant Text-To-Speech playback without waiting for the whole document.
   - Once all pages are done, a final JSON structural map of the document is compiled and saved to disk (Iceberg Storage).
   - Temporary Redis keys are cleaned up.

---

## What Makes VEDA Unique?

VEDA departs from traditional linear document parsers in three distinct ways:

### 1. Redis-Backed Distributed State & BBox Tolerance
Intermediate steps don't pass massive JSON blobs to one another. Each module reads from and writes to **Redis**. 
Because ML models are imperfect, VEDA implements a `bbox_matches(a, b, tolerance=5)` algorithm. This allows independent pipeline stages to fetch a region, update its content, and merge it back into the cached Redis layout even if the bounding box coordinates drift slightly between transformations.

### 2. Mathematics-Driven Image Contextualization (Spatial Scoring)
When sending an extracted diagram to an AI like Gemini for explanation, passing the *entire* text of a page breaks token limits and causes hallucination. VEDA uses a rigorous **Spatial Proximity Scoring** algorithm to select only the most relevant text context for the image:
*   **Normalized Distance**: Calculates the Manhattan distance between the center of the image and the center of every text block, normalized by the page's diagonal.
*   **Column Alignment Check**: Identifies horizontal overlap. If a text block shares >30% horizontal space with the image, it's flagged as being in the "same column" and receives a relevance boost (+100).
*   **Directional Vertical Bias**: Humans put captions *below* images and explanations *above* them. VEDA adds a +40 bonus to text located below the image, and a +20 bonus to text above.

### 3. "Just-In-Time" (Lazy) OCR Evaluation
Traditional pipelines OCR an entire page, which is tremendously slow and computationally expensive. VEDA does not. 
When the Spatial Scoring engine identifies the top-K text boxes that correspond to an image, it checks if those boxes have been OCR'd yet.
*   If they haven't (e.g. the user just uploaded the file and immediately clicked the image), VEDA transparently intercepts the request.
*   It fires up the local Tesseract engine, crops *only* those specific geometric bounding boxes, OCRs only the exact context lines needed, and saves it permanently to Redis.
*   It then forwards the context to Gemini. 

This **Lazy Evaluation** ensures that VEDA does zero wasted mathematical operations and enables instant, real-time interactions with documents without waiting ages for full-page processing. Furthermore, because it immediately caches both the OCR result and the final Gemini response in Redis, asking for the same explanation twice returns instantly (0ms logic overhead).

---

## Tech Stack
*   **Backend Framework**: FastAPI (Python), asyncio, Server-Sent Events (SSE)
*   **Machine Learning**: Ultralytics (DocLayout-YOLO), OpenCV, NumPy
*   **OCR Engines**: PyMuPDF (fitz), Tesseract OCR
*   **LLM Engine**: Google GenAI SDK (Gemini 2.5 Flash)
*   **Data Persistence**: Redis (Working Memory), JSON Storage (Iceberg)
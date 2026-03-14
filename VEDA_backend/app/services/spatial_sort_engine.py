from typing import List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

def recursive_xy_cut(boxes: List[Dict[str, Any]], vertical_gap_threshold: float, horizontal_gap_threshold: float, depth: int = 0) -> List[Dict[str, Any]]:
    """
    Recursively sorts bounding boxes into human reading order using the XY-Cut algorithm.
    """
    if not boxes or len(boxes) <= 1:
        return boxes

    logger.debug(f"Entered recursive_xy_cut at depth {depth} with {len(boxes)} boxes")

    # Sort boxes by X1 for vertical cut scanning
    boxes_by_x = sorted(boxes, key=lambda b: b['bbox'][0])

    # 1. Try Vertical Cut
    # A gap is a space between the max X2 of everything to the left, and the min X1 of everything to the right
    max_x2_so_far = boxes_by_x[0]['bbox'][2]
    best_v_gap = 0
    best_v_index = -1

    for i in range(1, len(boxes_by_x)):
        current_x1 = boxes_by_x[i]['bbox'][0]
        gap = current_x1 - max_x2_so_far
        
        # Check if gap is valid and largest found
        if gap > vertical_gap_threshold and gap > best_v_gap:
            # Semantic Masking Check
            gap_midpoint = (max_x2_so_far + current_x1) / 2
            is_masked = False
            for box in boxes:
                if box.get('type') in ["table", "figure", "isolate_formula", "table_caption"]:
                    if box['bbox'][0] < gap_midpoint < box['bbox'][2]:
                        is_masked = True
                        break
            
            if not is_masked:
                best_v_gap = gap
                best_v_index = i

        max_x2_so_far = max(max_x2_so_far, boxes_by_x[i]['bbox'][2])

    if best_v_index != -1:
        logger.debug(f"[Depth {depth}] Chose Vertical Cut with gap: {best_v_gap:.2f} (Threshold: {vertical_gap_threshold:.2f})")
        left_list = boxes_by_x[:best_v_index]
        right_list = boxes_by_x[best_v_index:]
        return recursive_xy_cut(left_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1) + \
               recursive_xy_cut(right_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1)

    # 2. Try Horizontal Cut
    # Sort by Y1 for horizontal cut scanning
    boxes_by_y = sorted(boxes, key=lambda b: b['bbox'][1])
    max_y2_so_far = boxes_by_y[0]['bbox'][3]
    best_h_gap = 0
    best_h_index = -1

    for i in range(1, len(boxes_by_y)):
        current_y1 = boxes_by_y[i]['bbox'][1]
        gap = current_y1 - max_y2_so_far

        if gap > horizontal_gap_threshold and gap > best_h_gap:
            best_h_gap = gap
            best_h_index = i
            
        max_y2_so_far = max(max_y2_so_far, boxes_by_y[i]['bbox'][3])

    if best_h_index != -1:
        logger.debug(f"[Depth {depth}] Chose Horizontal Cut with gap: {best_h_gap:.2f} (Threshold: {horizontal_gap_threshold:.2f})")
        top_list = boxes_by_y[:best_h_index]
        bottom_list = boxes_by_y[best_h_index:]
        return recursive_xy_cut(top_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1) + \
               recursive_xy_cut(bottom_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1)

    # 3. Base Case: No cuts possible, sort by Y1
    return sorted(boxes, key=lambda b: b['bbox'][1])


def process_spatial_sort(layout_payload: dict) -> dict:
    """
    Takes layout_data JSON, cleans 'abandon' regions, calculates dynamic thresholds,
    and sorts the bounding boxes using Guided Recursive XY-Cut logic.
    """
    logger.info("Starting process_spatial_sort.")
    if "layout_data" not in layout_payload:
        logger.warning("No 'layout_data' found in payload.")
        return layout_payload

    # Process page by page
    for page_data in layout_payload.get("layout_data", []):
        regions = page_data.get("regions", [])
        
        # Step 1: Clean
        valid_regions = [r for r in regions if r.get("type", "").lower() != "abandon"]
        
        if not valid_regions:
            page_data["regions"] = []
            continue

        # Step 2: Dynamic Area Calculation
        # bbox format: [x1, y1, x2, y2]
        min_x1 = min(r['bbox'][0] for r in valid_regions)
        max_x2 = max(r['bbox'][2] for r in valid_regions)
        min_y1 = min(r['bbox'][1] for r in valid_regions)
        max_y2 = max(r['bbox'][3] for r in valid_regions)

        # Step 3: Dynamic Thresholds
        active_width = max_x2 - min_x1
        active_height = max_y2 - min_y1
        
        vertical_gap_threshold = active_width * 0.015
        horizontal_gap_threshold = active_height * 0.015
        
        logger.debug(f"Calculated thresholds - Vertical: {vertical_gap_threshold:.2f}, Horizontal: {horizontal_gap_threshold:.2f}")

        # Page Layout Detection (Count major vertical gaps)
        boxes_by_x = sorted(valid_regions, key=lambda b: b['bbox'][0])
        max_x2_so_far = boxes_by_x[0]['bbox'][2]
        valid_v_gaps = 0
        
        for i in range(1, len(boxes_by_x)):
            current_x1 = boxes_by_x[i]['bbox'][0]
            gap = current_x1 - max_x2_so_far
            
            if gap > vertical_gap_threshold:
                gap_midpoint = (max_x2_so_far + current_x1) / 2
                is_masked = False
                for box in valid_regions:
                    if box.get('type') in ["table", "figure", "isolate_formula", "table_caption"]:
                        if box['bbox'][0] < gap_midpoint < box['bbox'][2]:
                            is_masked = True
                            break
                if not is_masked:
                    valid_v_gaps += 1
                    
            max_x2_so_far = max(max_x2_so_far, boxes_by_x[i]['bbox'][2])
            
        if valid_v_gaps == 0:
            page_data["page_layout"] = "single column"
        elif valid_v_gaps == 1:
            page_data["page_layout"] = "2 col"
        else:
            page_data["page_layout"] = "3 col"
            
        logger.info(f"Detected page layout: {page_data['page_layout']}")

        # Step 4: Recursive Selection
        sorted_regions = recursive_xy_cut(
            valid_regions, 
            vertical_gap_threshold, 
            horizontal_gap_threshold
        )

        # Step 5: Format (Assign reading order)
        for idx, region in enumerate(sorted_regions):
            region["reading_order"] = idx + 1

        page_data["regions"] = sorted_regions
        logger.info(f"Finished spatial sorting for page. Processed {len(sorted_regions)} regions.")

    return layout_payload

"""
VEDA Spatial Sort Engine.

Implements the Guided Recursive XY-Cut algorithm to sort document regions
into human reading order. Processes unordered bounding boxes from YOLO
layout analysis, calculates dynamic gap thresholds based on page
dimensions, detects column layouts, and assigns reading_order indices.

This module exists because YOLO outputs regions in arbitrary detection
order, but downstream OCR concatenation and TTS require logical reading
sequence.

Leverages: Python standard library only (typing).
"""

from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class SpatialSortEngine:
    """
    Sorts document regions into reading order using Recursive XY-Cut.

    The algorithm alternates between vertical and horizontal cuts,
    splitting the page recursively at the largest gap that exceeds a
    dynamic threshold. Semantic masking prevents cuts through tables,
    figures, and formulas.

    Leverages: Python sorted, list comprehensions.
    """

    def recursive_xy_cut(
        self,
        boxes: List[Dict[str, Any]],
        vertical_gap_threshold: float,
        horizontal_gap_threshold: float,
        depth: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Recursively sort bounding boxes into reading order.

        Attempts a vertical cut first (left/right split), then a
        horizontal cut (top/bottom split). If neither cut exceeds the
        threshold, falls back to simple Y-coordinate sorting.

        Semantic masking skips vertical cuts that would bisect tables,
        figures, or formulas.

        Leverages: Python sorted with lambda keys.
        """
        if not boxes or len(boxes) <= 1:
            return boxes

        logger.debug(
            f"Entered recursive_xy_cut at depth {depth} with {len(boxes)} boxes"
        )

        boxes_by_x = sorted(boxes, key=lambda b: b["bbox"][0])

        max_x2_so_far = boxes_by_x[0]["bbox"][2]
        best_v_gap = 0
        best_v_index = -1

        for i in range(1, len(boxes_by_x)):
            current_x1 = boxes_by_x[i]["bbox"][0]
            gap = current_x1 - max_x2_so_far

            if gap > vertical_gap_threshold and gap > best_v_gap:
                gap_midpoint = (max_x2_so_far + current_x1) / 2
                is_masked = False
                for box in boxes:
                    if box.get("label", "").lower() in [
                        "table",
                        "figure",
                        "isolate_formula",
                        "table_caption",
                    ]:
                        if box["bbox"][0] < gap_midpoint < box["bbox"][2]:
                            is_masked = True
                            break

                if not is_masked:
                    best_v_gap = gap
                    best_v_index = i

            max_x2_so_far = max(max_x2_so_far, boxes_by_x[i]["bbox"][2])

        if best_v_index != -1:
            logger.debug(
                f"[Depth {depth}] Chose Vertical Cut with gap: {best_v_gap:.2f} "
                f"(Threshold: {vertical_gap_threshold:.2f})"
            )
            left_list = boxes_by_x[:best_v_index]
            right_list = boxes_by_x[best_v_index:]
            return self.recursive_xy_cut(
                left_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1
            ) + self.recursive_xy_cut(
                right_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1
            )

        boxes_by_y = sorted(boxes, key=lambda b: b["bbox"][1])
        max_y2_so_far = boxes_by_y[0]["bbox"][3]
        best_h_gap = 0
        best_h_index = -1

        for i in range(1, len(boxes_by_y)):
            current_y1 = boxes_by_y[i]["bbox"][1]
            gap = current_y1 - max_y2_so_far

            if gap > horizontal_gap_threshold and gap > best_h_gap:
                best_h_gap = gap
                best_h_index = i

            max_y2_so_far = max(max_y2_so_far, boxes_by_y[i]["bbox"][3])

        if best_h_index != -1:
            logger.debug(
                f"[Depth {depth}] Chose Horizontal Cut with gap: {best_h_gap:.2f} "
                f"(Threshold: {horizontal_gap_threshold:.2f})"
            )
            top_list = boxes_by_y[:best_h_index]
            bottom_list = boxes_by_y[best_h_index:]
            return self.recursive_xy_cut(
                top_list, vertical_gap_threshold, horizontal_gap_threshold, depth + 1
            ) + self.recursive_xy_cut(
                bottom_list,
                vertical_gap_threshold,
                horizontal_gap_threshold,
                depth + 1,
            )

        return sorted(boxes, key=lambda b: b["bbox"][1])

    def process_spatial_sort(self, layout_payload: dict) -> dict:
        """
        Clean abandon regions, compute dynamic thresholds, and sort
        all pages in the layout payload using Recursive XY-Cut.

        Detects single-column vs. two-column page layouts and assigns
        sequential reading_order indices to each region.

        Leverages: recursive_xy_cut for the core sorting algorithm.
        """
        logger.info("Starting process_spatial_sort.")
        if "layout_data" not in layout_payload:
            logger.warning("No 'layout_data' found in payload.")
            return layout_payload

        for page_data in layout_payload.get("layout_data", []):
            regions = page_data.get("regions", [])

            valid_regions = [
                r for r in regions if r.get("label", "").lower() != "abandon"
            ]

            if not valid_regions:
                page_data["regions"] = []
                continue

            min_x1 = min(r["bbox"][0] for r in valid_regions)
            max_x2 = max(r["bbox"][2] for r in valid_regions)
            min_y1 = min(r["bbox"][1] for r in valid_regions)
            max_y2 = max(r["bbox"][3] for r in valid_regions)

            active_width = max_x2 - min_x1
            active_height = max_y2 - min_y1

            vertical_gap_threshold = active_width * 0.015
            horizontal_gap_threshold = active_height * 0.015

            logger.debug(
                f"Calculated thresholds - Vertical: {vertical_gap_threshold:.2f}, "
                f"Horizontal: {horizontal_gap_threshold:.2f}"
            )

            page_mid = (min_x1 + max_x2) / 2
            x_centers = [
                (r["bbox"][0] + r["bbox"][2]) / 2 for r in valid_regions
            ]

            left = [x for x in x_centers if x < page_mid]
            right = [x for x in x_centers if x >= page_mid]
            if len(left) > 2 and len(right) > 2:
                page_data["page_layout"] = "2 col"
            else:
                page_data["page_layout"] = "single column"

            logger.info(f"Detected page layout: {page_data['page_layout']}")

            sorted_regions = self.recursive_xy_cut(
                valid_regions, vertical_gap_threshold, horizontal_gap_threshold
            )

            for idx, region in enumerate(sorted_regions):
                region["reading_order"] = idx + 1

            page_data["regions"] = sorted_regions
            logger.info(
                f"Finished spatial sorting for page. "
                f"Processed {len(sorted_regions)} regions."
            )

        return layout_payload


_instance = SpatialSortEngine()

recursive_xy_cut = _instance.recursive_xy_cut
process_spatial_sort = _instance.process_spatial_sort

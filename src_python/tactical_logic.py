"""
tactical_logic.py
=================
Decision logic, thin-structure-safe filtering, and spatial NMS for civil defect detection.

Features:
- Preserves 1-2 px hairline cracks (avoiding destructive 5x5 MORPH_OPEN).
- Aspect-padded square 64x64 ROI crop extraction for Stage 3 MobileNetV3.
- Advisory AASHTO (MBEI 1130) condition state severity evaluation.
- Pure NumPy/OpenCV reference anomaly calculus (serves as oracle & fallback).
"""

import cv2
import numpy as np


def compute_pixelwise_mse_numpy(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    gaussian_sigma: float = 0.8,
) -> tuple[float, np.ndarray]:
    """Pure NumPy/OpenCV reference implementation for pixel-wise MSE heatmap.

    Supports both 2D (H, W) and 3D (H, W, C) float32 / uint8 inputs.
    Returns:
        (mean_mse: float, heatmap_2d: np.ndarray float32)
    """
    if frame_a.size == 0 or frame_b.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.float32)

    # Normalize to float32 in [0, 1]
    a = frame_a.astype(np.float32)
    if a.max() > 1.0:
        a = a / 255.0

    b = frame_b.astype(np.float32)
    if b.max() > 1.0:
        b = b / 255.0

    # Ensure matching resolutions
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Squared error: (A - B)^2
    diff = a - b
    sq_diff = diff * diff

    # If 3-channel, average error across channels
    if sq_diff.ndim == 3 and sq_diff.shape[2] > 1:
        heatmap = np.mean(sq_diff, axis=2).astype(np.float32)
    else:
        heatmap = sq_diff.astype(np.float32)

    # Mild Gaussian blur to suppress single-pixel sensor noise without erasing cracks
    if gaussian_sigma > 0.0:
        ksize = 3
        heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), gaussian_sigma)

    mean_mse = float(np.mean(heatmap))
    return mean_mse, heatmap


class TacticalDecisionLogic:
    """Filters spatial MSE anomaly heatmaps, runs contour clustering, and applies NMS."""

    def __init__(
        self,
        mse_threshold: float = 0.06,
        min_area: int = 15,
        max_area: int = 40000,
        nms_iou_threshold: float = 0.3,
        min_crack_length: int = 10,
    ):
        self.mse_threshold = mse_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.nms_iou_threshold = nms_iou_threshold
        self.min_crack_length = min_crack_length

    def set_mse_threshold(self, th: float):
        self.mse_threshold = th

    def compute_iou(self, boxA: list[int], boxB: list[int]) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)

    def apply_nms(
        self, boxes: list[list[int]], scores: list[float]
    ) -> tuple[list[list[int]], list[float]]:
        """Performs Non-Maximum Suppression (NMS) over bounding boxes."""
        if len(boxes) == 0:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)

        idxs = np.argsort(scores_arr)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)

            if len(idxs) == 1:
                break

            rest = idxs[1:]
            filtered_idxs = []
            for i in rest:
                iou = self.compute_iou(boxes_arr[current], boxes_arr[i])
                if iou < self.nms_iou_threshold:
                    filtered_idxs.append(i)
            idxs = np.array(filtered_idxs)

        keep_boxes = [boxes[i] for i in keep]
        keep_scores = [scores[i] for i in keep]
        return keep_boxes, keep_scores

    def filter_thin_structures(self, binary_mask: np.ndarray) -> np.ndarray:
        """Cleans binary mask while strictly preserving 1-2 px hairline cracks.

        Uses small 3x3 morphological closing (to bridge micro-disconnects)
        and rejects small isolated speckle noise dots (< 8 px area with no elongation).
        """
        # Small 3x3 closing bridges disconnected segments along a continuous crack line
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Connected component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        cleaned_mask = np.zeros_like(closed_mask)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            diagonal = np.hypot(w, h)

            # Keep if area is sufficient OR if elongated (hairline crack signature)
            is_large = (area >= self.min_area) and (area <= self.max_area)
            is_thin_crack = (diagonal >= self.min_crack_length) and (area >= 6)

            if is_large or is_thin_crack:
                cleaned_mask[labels == label] = 255

        return cleaned_mask

    def detect_defects(
        self,
        mse_heatmap: np.ndarray,
        gsd_mm_per_px: float = None,
    ) -> tuple[list[dict], np.ndarray]:
        """Processes anomaly heatmap and returns civil defect alerts."""
        if mse_heatmap is None or mse_heatmap.size == 0:
            return [], None

        # 1. Absolute thresholding based on calibrated MSE
        binary_mask = (mse_heatmap > self.mse_threshold).astype(np.uint8) * 255

        # 2. Thin-structure-safe filtering
        cleaned_mask = self.filter_thin_structures(binary_mask)

        # 3. Contour Detection & Candidate Extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        candidate_scores = []
        candidate_contours = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            diag = np.hypot(w, h)

            if (self.min_area <= area <= self.max_area) or (diag >= self.min_crack_length and area >= 6):
                roi_mse = mse_heatmap[y : y + h, x : x + w]
                roi_mask = cleaned_mask[y : y + h, x : x + w]
                active_pts = roi_mse[roi_mask > 0]
                score = float(np.mean(active_pts)) if active_pts.size > 0 else float(self.mse_threshold)
                candidate_boxes.append([x, y, w, h])
                candidate_scores.append(score)
                candidate_contours.append(cnt)

        # 4. Spatial NMS
        nms_boxes, nms_scores = self.apply_nms(candidate_boxes, candidate_scores)

        # 5. Format Civil Defect Alerts with advisory severity
        defects = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x, y, w, h = box
            mask_roi = cleaned_mask[y : y + h, x : x + w]

            # Measure crack width in pixels via distance transform
            dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 3)
            max_dist = float(np.max(dist)) if dist.size > 0 and np.max(dist) > 0 else 0.5
            est_width_px = 2.0 * max_dist

            severity_info = self.evaluate_advisory_severity(
                bbox=box,
                score=score,
                width_px=est_width_px,
                gsd_mm_per_px=gsd_mm_per_px,
            )

            defects.append({
                "id": i + 1,
                "bbox": [x, y, w, h],
                "score": score,
                "confidence": min(1.0, score / (self.mse_threshold * 2.5 + 1e-6)),
                "type": severity_info["defect_type"],
                "condition_state": severity_info["condition_state"],
                "est_width_mm": severity_info["est_width_mm"],
                "est_width_px": est_width_px,
                "standard": severity_info["standard"],
            })

        return defects, cleaned_mask

    def detect_tactical_threats(self, mse_heatmap: np.ndarray) -> tuple[list[dict], np.ndarray]:
        """Backward-compatible alias for detect_defects."""
        return self.detect_defects(mse_heatmap)

    def evaluate_advisory_severity(
        self,
        bbox: list[int],
        score: float,
        width_px: float,
        gsd_mm_per_px: float = None,
    ) -> dict:
        """Maps detected geometry to advisory AASHTO MBEI 1130 Condition States."""
        x, y, w, h = bbox
        diag = np.hypot(w, h)
        area = w * h

        # Rough initial heuristic before Stage 3 classifier confirmation
        if area > 1200:
            defect_type = "spalling"
        elif diag >= self.min_crack_length * 1.5:
            defect_type = "crack"
        else:
            defect_type = "surface_anomaly"

        if gsd_mm_per_px is not None:
            width_mm = width_px * gsd_mm_per_px
            standard = "AASHTO_MBEI_1130"
            if width_mm < 0.3:
                cs = "CS1 (Good / Hairline <0.3mm)"
            elif width_mm <= 1.25:
                cs = "CS2 (Fair / 0.3-1.25mm)"
            elif width_mm <= 2.0:
                cs = "CS3 (Poor / 1.25-2.0mm)"
            else:
                cs = "CS4 (Severe / Structural Review >2.0mm)"
        else:
            width_mm = None
            standard = "ADVISORY_PIXEL_ESTIMATE"
            cs = "CS_PENDING_GSD"

        return {
            "defect_type": defect_type,
            "condition_state": cs,
            "est_width_mm": width_mm,
            "standard": standard,
        }

    @staticmethod
    def extract_roi_crops(
        image: np.ndarray,
        boxes: list[list[int]],
        crop_size: int = 64,
        pad_fraction: float = 0.15,
    ) -> list[np.ndarray]:
        """Extracts square, aspect-padded 64x64 ROI crops for Stage 3 Classifier.

        Avoids distorting diagonal cracks by padding to a 1:1 aspect ratio.
        """
        if len(boxes) == 0:
            return []

        h_img, w_img = image.shape[:2]
        crops = []

        for box in boxes:
            x, y, w, h = box
            # Calculate square side length with padding
            max_dim = max(w, h)
            pad = int(max_dim * pad_fraction)
            side = max(16, max_dim + 2 * pad)

            # Center coordinates
            cx = x + w // 2
            cy = y + h // 2

            # Square bounds clamped to image
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w_img, x1 + side)
            y2 = min(h_img, y1 + side)

            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Ensure square via border padding if near edge
            ph, pw = patch.shape[:2]
            if ph != pw:
                target_dim = max(ph, pw)
                pad_top = (target_dim - ph) // 2
                pad_bottom = target_dim - ph - pad_top
                pad_left = (target_dim - pw) // 2
                pad_right = target_dim - pw - pad_left
                patch = cv2.copyMakeBorder(
                    patch,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REFLECT_101,
                )

            # Resize to exactly (crop_size, crop_size)
            resized = cv2.resize(patch, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            crops.append(resized)

        return crops

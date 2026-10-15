"""
MicroCount - Watershed Segmentation Module
Separates touching/overlapping colonies using distance transform + watershed.
Applied when YOLO detections indicate merged/clustered colonies.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from loguru import logger

from app.services.detector import DetectionBox


@dataclass
class WatershedResult:
    separated_boxes: list[DetectionBox] = field(default_factory=list)
    original_boxes: list[DetectionBox] = field(default_factory=list)
    separation_count: int = 0  # How many colonies were separated
    total_final: int = 0


class WatershedSeparator:
    """
    Applies watershed segmentation to regions where colonies are detected
    as touching or merged blobs.
    
    Strategy:
    1. For each YOLO detection box, extract the region
    2. Compute distance transform on binary mask
    3. Find local maxima (colony centers)
    4. Apply watershed to segment individual colonies
    5. Return new bounding boxes for separated colonies
    """

    # Thresholds for deciding when to apply watershed
    ASPECT_RATIO_THRESHOLD = 1.8  # If bbox is too wide/tall, likely merged
    AREA_MULTIPLIER_THRESHOLD = 2.5  # If area >> median, likely merged
    MIN_SEPARATION_AREA_PX = 50  # Minimum area for a separated colony

    def separate(
        self,
        image: np.ndarray,
        detections: list[DetectionBox],
        petri_mask: Optional[np.ndarray] = None,
    ) -> WatershedResult:
        """
        Process all detections and apply watershed where needed.
        """
        if not detections:
            return WatershedResult(total_final=0)

        # Compute median box area to identify anomalously large boxes
        areas = [d.area for d in detections]
        median_area = float(np.median(areas)) if areas else 1.0

        final_boxes = []
        separation_count = 0

        for box in detections:
            if self._should_apply_watershed(box, median_area):
                separated = self._apply_watershed_to_box(image, box)
                if separated and len(separated) > 1:
                    final_boxes.extend(separated)
                    separation_count += 1
                    logger.debug(
                        f"Watershed: split box {box.x1:.0f},{box.y1:.0f} → "
                        f"{len(separated)} colonies"
                    )
                else:
                    # Watershed didn't split — keep original
                    final_boxes.append(box)
            else:
                final_boxes.append(box)

        return WatershedResult(
            separated_boxes=final_boxes,
            original_boxes=detections,
            separation_count=separation_count,
            total_final=len(final_boxes),
        )

    def _should_apply_watershed(self, box: DetectionBox, median_area: float) -> bool:
        """Decide whether this box likely contains merged colonies."""
        # Check aspect ratio
        aspect_ratio = max(box.width, box.height) / max(min(box.width, box.height), 1)
        if aspect_ratio > self.ASPECT_RATIO_THRESHOLD:
            return True

        # Check if area significantly exceeds median
        if median_area > 0 and box.area > median_area * self.AREA_MULTIPLIER_THRESHOLD:
            return True

        return False

    def _apply_watershed_to_box(
        self,
        image: np.ndarray,
        box: DetectionBox,
    ) -> list[DetectionBox]:
        """
        Apply watershed segmentation within a bounding box region.
        Returns list of new DetectionBoxes for each separated colony.
        """
        h, w = image.shape[:2]

        # Extract ROI with padding
        pad = 5
        x1 = max(0, int(box.x1) - pad)
        y1 = max(0, int(box.y1) - pad)
        x2 = min(w, int(box.x2) + pad)
        y2 = min(h, int(box.y2) + pad)

        if x2 <= x1 or y2 <= y1:
            return [box]

        roi = image[y1:y2, x1:x2].copy()

        if roi.size == 0:
            return [box]

        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            roi_gray = roi

        # Create binary mask using Otsu threshold
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If the region is mostly background, try inverse
        if np.sum(binary > 0) < np.sum(binary == 0):
            binary = cv2.bitwise_not(binary)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Distance transform
        binary_bool = (binary > 0).astype(np.uint8)
        dist_transform = ndi.distance_transform_edt(binary_bool)

        if dist_transform.max() == 0:
            return [box]

        # Normalize
        dist_norm = dist_transform / dist_transform.max()

        # Find local maxima (colony centers)
        min_distance = max(5, int(min(roi.shape[:2]) * 0.15))
        coordinates = peak_local_max(
            dist_norm,
            min_distance=min_distance,
            threshold_rel=0.3,
            exclude_border=False,
        )

        if len(coordinates) <= 1:
            # Only one or no peaks found - no separation needed
            return [box]

        # Create markers for watershed
        markers = np.zeros(dist_norm.shape, dtype=np.int32)
        for idx, (row, col) in enumerate(coordinates):
            markers[row, col] = idx + 1

        # Apply watershed
        labels = watershed(-dist_norm, markers, mask=binary_bool)

        # Extract bounding boxes for each label
        new_boxes = []
        for label_id in range(1, labels.max() + 1):
            region = (labels == label_id)
            region_area = np.sum(region)

            if region_area < self.MIN_SEPARATION_AREA_PX:
                continue

            # Find bounding box of this region
            rows = np.where(region.any(axis=1))[0]
            cols = np.where(region.any(axis=0))[0]

            if len(rows) == 0 or len(cols) == 0:
                continue

            # Convert back to full image coordinates
            rx1 = float(cols.min() + x1)
            ry1 = float(rows.min() + y1)
            rx2 = float(cols.max() + x1)
            ry2 = float(rows.max() + y1)

            new_boxes.append(
                DetectionBox(
                    x1=rx1,
                    y1=ry1,
                    x2=rx2,
                    y2=ry2,
                    confidence=box.confidence * 0.95,  # Slight confidence penalty
                    class_id=box.class_id,
                )
            )

        if not new_boxes:
            return [box]

        return new_boxes


# Singleton
_separator: Optional[WatershedSeparator] = None


def get_separator() -> WatershedSeparator:
    global _separator
    if _separator is None:
        _separator = WatershedSeparator()
    return _separator

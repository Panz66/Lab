"""
MicroCount - Colony Morphology Analysis Module
Analyzes morphological properties of each detected colony:
diameter, circularity, color, opacity, edge smoothness, size category.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from app.services.detector import DetectionBox


@dataclass
class ColonyMorphologyData:
    colony_index: int
    bbox: tuple[float, float, float, float]
    centroid_x: float
    centroid_y: float
    confidence: float

    # Morphology
    diameter_px: float = 0.0
    diameter_mm: Optional[float] = None
    area_px: float = 0.0
    circularity: float = 0.0
    mean_intensity: float = 0.0
    color_r: float = 0.0
    color_g: float = 0.0
    color_b: float = 0.0
    opacity: float = 0.0
    edge_smoothness: float = 0.0
    size_category: str = "medium"
    was_separated: bool = False

    # Derived
    quadrant: Optional[int] = None


# Colony size categories in pixels (diameter)
SIZE_CATEGORIES = {
    "pinpoint": (0, 8),      # < 8px diameter
    "small": (8, 20),         # 8–20px
    "medium": (20, 50),       # 20–50px
    "large": (50, 999),       # > 50px
}


class MorphologyAnalyzer:
    """
    Extracts morphological features for each colony bounding box.
    Optionally converts pixel measurements to mm using petri dish scale.
    """

    # Standard petri dish diameter is 90mm (most common lab size)
    STANDARD_PETRI_DIAMETER_MM = 90.0

    def analyze_all(
        self,
        image: np.ndarray,
        boxes: list[DetectionBox],
        petri_radius_px: Optional[float] = None,
        separated_indices: Optional[set[int]] = None,
    ) -> list[ColonyMorphologyData]:
        """
        Analyze morphology for all detected colonies.

        Args:
            image: RGB image
            boxes: List of detection boxes
            petri_radius_px: Pixel radius of petri dish (for mm conversion)
            separated_indices: Set of box indices that came from watershed separation
        """
        if not boxes:
            return []

        # Calculate px/mm scale factor
        px_per_mm = None
        if petri_radius_px and petri_radius_px > 0:
            px_per_mm = (2 * petri_radius_px) / self.STANDARD_PETRI_DIAMETER_MM

        results = []
        for idx, box in enumerate(boxes):
            was_sep = separated_indices is not None and idx in separated_indices
            morph = self._analyze_colony(image, box, idx, px_per_mm, was_sep)
            results.append(morph)

        return results

    def _analyze_colony(
        self,
        image: np.ndarray,
        box: DetectionBox,
        idx: int,
        px_per_mm: Optional[float],
        was_separated: bool,
    ) -> ColonyMorphologyData:
        """Analyze a single colony."""
        h, w = image.shape[:2]

        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(w, int(box.x2))
        y2 = min(h, int(box.y2))

        # Dimensions
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return self._default_morph(idx, box, was_separated)

        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return self._default_morph(idx, box, was_separated)

        # Diameter: average of width and height
        diameter_px = (bw + bh) / 2

        # Area from ellipse approximation
        area_px = np.pi * (bw / 2) * (bh / 2)

        # Diameter in mm
        diameter_mm = None
        if px_per_mm and px_per_mm > 0:
            diameter_mm = round(diameter_px / px_per_mm, 2)

        # Size category
        size_cat = self._categorize_size(diameter_px)

        # Mean color in ROI
        if len(roi.shape) == 3:
            mean_r = float(np.mean(roi[:, :, 0]))
            mean_g = float(np.mean(roi[:, :, 1]))
            mean_b = float(np.mean(roi[:, :, 2]))
            mean_intensity = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)))
        else:
            mean_r = mean_g = mean_b = float(np.mean(roi))
            mean_intensity = mean_r

        # Opacity: contrast between colony and local background
        opacity = self._compute_opacity(image, box, x1, y1, x2, y2)

        # Circularity: 4πA/P²
        circularity = self._compute_circularity(roi, bw, bh)

        # Edge smoothness (inverse of edge gradient variance)
        edge_smoothness = self._compute_edge_smoothness(roi)

        return ColonyMorphologyData(
            colony_index=idx,
            bbox=(box.x1, box.y1, box.x2, box.y2),
            centroid_x=box.centroid_x,
            centroid_y=box.centroid_y,
            confidence=box.confidence,
            diameter_px=round(diameter_px, 2),
            diameter_mm=diameter_mm,
            area_px=round(area_px, 2),
            circularity=round(circularity, 3),
            mean_intensity=round(mean_intensity, 2),
            color_r=round(mean_r, 2),
            color_g=round(mean_g, 2),
            color_b=round(mean_b, 2),
            opacity=round(opacity, 3),
            edge_smoothness=round(edge_smoothness, 3),
            size_category=size_cat,
            was_separated=was_separated,
        )

    def _categorize_size(self, diameter_px: float) -> str:
        """Categorize colony by diameter in pixels."""
        for cat, (lo, hi) in SIZE_CATEGORIES.items():
            if lo <= diameter_px < hi:
                return cat
        return "large"

    def _compute_circularity(self, roi: np.ndarray, bw: int, bh: int) -> float:
        """Compute circularity from ROI contours."""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                # Estimate from aspect ratio
                aspect = min(bw, bh) / max(bw, bh)
                return aspect

            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if perimeter == 0:
                return 0.0

            return min(1.0, float(4 * np.pi * area / (perimeter ** 2)))
        except Exception:
            return 0.5

    def _compute_opacity(
        self,
        image: np.ndarray,
        box: DetectionBox,
        x1: int, y1: int, x2: int, y2: int,
    ) -> float:
        """
        Estimate opacity as contrast ratio between colony region
        and a small border around it.
        """
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # Colony region
            colony_region = gray[y1:y2, x1:x2]
            colony_mean = float(np.mean(colony_region))

            # Border region (expand by 5px)
            bx1 = max(0, x1 - 5)
            by1 = max(0, y1 - 5)
            bx2 = min(w, x2 + 5)
            by2 = min(h, y2 + 5)

            # Full border frame
            border_region = gray[by1:by2, bx1:bx2]
            border_mean = float(np.mean(border_region))

            if border_mean == 0:
                return 0.5

            # Contrast ratio
            contrast = abs(colony_mean - border_mean) / (border_mean + 1e-6)
            return min(1.0, contrast)
        except Exception:
            return 0.5

    def _compute_edge_smoothness(self, roi: np.ndarray) -> float:
        """
        Compute edge smoothness as inverse of Sobel gradient std dev.
        High smoothness = low edge variance = smooth colony edge.
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            std = float(np.std(gradient_mag))
            # Normalize: lower std = smoother edge
            smoothness = 1.0 - min(1.0, std / 100.0)
            return smoothness
        except Exception:
            return 0.5

    def _default_morph(self, idx: int, box: DetectionBox, was_separated: bool) -> ColonyMorphologyData:
        """Return a minimal morphology object when analysis fails."""
        return ColonyMorphologyData(
            colony_index=idx,
            bbox=(box.x1, box.y1, box.x2, box.y2),
            centroid_x=box.centroid_x,
            centroid_y=box.centroid_y,
            confidence=box.confidence,
            was_separated=was_separated,
        )

    def compute_quadrants(
        self,
        colonies: list[ColonyMorphologyData],
        center_x: float,
        center_y: float,
    ) -> list[ColonyMorphologyData]:
        """
        Assign quadrant (1-4) to each colony based on petri dish center.
        Q1=top-right, Q2=top-left, Q3=bottom-left, Q4=bottom-right.
        """
        for c in colonies:
            dx = c.centroid_x - center_x
            dy = c.centroid_y - center_y

            if dx >= 0 and dy < 0:
                c.quadrant = 1  # Top-right
            elif dx < 0 and dy < 0:
                c.quadrant = 2  # Top-left
            elif dx < 0 and dy >= 0:
                c.quadrant = 3  # Bottom-left
            else:
                c.quadrant = 4  # Bottom-right

        return colonies

    def summarize(
        self, colonies: list[ColonyMorphologyData]
    ) -> dict:
        """Generate morphology summary statistics."""
        if not colonies:
            return {}

        counts = {"pinpoint": 0, "small": 0, "medium": 0, "large": 0}
        for c in colonies:
            counts[c.size_category] = counts.get(c.size_category, 0) + 1

        total = len(colonies)
        distribution = {k: round(v / total * 100, 1) for k, v in counts.items()}

        diameters = [c.diameter_mm for c in colonies if c.diameter_mm]
        mean_diameter = round(float(np.mean(diameters)), 2) if diameters else None

        circularities = [c.circularity for c in colonies if c.circularity > 0]
        mean_circularity = round(float(np.mean(circularities)), 3) if circularities else None

        # Dominant color (most common colony appearance)
        intensities = [c.mean_intensity for c in colonies]
        if intensities:
            mean_int = np.mean(intensities)
            if mean_int < 80:
                dominant_color = "dark"
            elif mean_int < 160:
                dominant_color = "cream/beige"
            else:
                dominant_color = "light/white"
        else:
            dominant_color = "unknown"

        return {
            "pinpoint_count": counts["pinpoint"],
            "small_count": counts["small"],
            "medium_count": counts["medium"],
            "large_count": counts["large"],
            "mean_diameter_mm": mean_diameter,
            "mean_circularity": mean_circularity,
            "dominant_color": dominant_color,
            "size_distribution": distribution,
        }

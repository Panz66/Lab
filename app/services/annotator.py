"""
MicroCount - Image Annotation Module
Draws detection results, quadrant lines, colony labels on images.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from app.services.morphology import ColonyMorphologyData


# Color palette per size category (BGR)
SIZE_COLORS = {
    "pinpoint": (255, 100, 100),   # Blue
    "small": (100, 255, 100),      # Green
    "medium": (100, 100, 255),     # Red
    "large": (255, 165, 0),        # Orange
}

QC_GRADE_COLORS = {
    "Excellent": (0, 200, 0),
    "Good": (100, 200, 100),
    "Fair": (200, 200, 0),
    "Poor": (0, 0, 200),
}


class ImageAnnotator:
    """Draws visual annotations on petri dish images."""

    def annotate(
        self,
        image: np.ndarray,
        colonies: list[ColonyMorphologyData],
        petri_center: Optional[tuple[float, float]] = None,
        petri_radius: Optional[float] = None,
        qc_grade: str = "Good",
        total_count: int = 0,
        cfu_formatted: str = "",
    ) -> np.ndarray:
        """
        Draw full annotation overlay on image.
        Returns annotated copy.
        """
        # Work on BGR copy
        if len(image.shape) == 3 and image.shape[2] == 3:
            annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            annotated = image.copy()

        # Draw petri dish circle
        if petri_center and petri_radius:
            self._draw_petri_circle(annotated, petri_center, petri_radius)
            self._draw_quadrant_lines(annotated, petri_center, petri_radius)

        # Draw colonies
        for colony in colonies:
            self._draw_colony(annotated, colony)

        # Draw info overlay
        self._draw_info_overlay(
            annotated,
            total_count=total_count,
            cfu_formatted=cfu_formatted,
            qc_grade=qc_grade,
        )

        return annotated

    def _draw_petri_circle(
        self,
        img: np.ndarray,
        center: tuple[float, float],
        radius: float,
    ):
        """Draw petri dish boundary circle."""
        cv2.circle(
            img,
            (int(center[0]), int(center[1])),
            int(radius),
            (200, 200, 200),  # Light gray
            2,
            cv2.LINE_AA,
        )

    def _draw_quadrant_lines(
        self,
        img: np.ndarray,
        center: tuple[float, float],
        radius: float,
    ):
        """Draw quadrant dividing lines (crosshair)."""
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        color = (150, 150, 150)  # Gray
        thickness = 1

        # Horizontal line
        cv2.line(img, (cx - r, cy), (cx + r, cy), color, thickness, cv2.LINE_AA)
        # Vertical line
        cv2.line(img, (cx, cy - r), (cx, cy + r), color, thickness, cv2.LINE_AA)

        # Quadrant labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        offset = r // 2
        for label, pos in [
            ("Q1", (cx + offset // 2, cy - offset // 2)),
            ("Q2", (cx - offset, cy - offset // 2)),
            ("Q3", (cx - offset, cy + offset // 2)),
            ("Q4", (cx + offset // 2, cy + offset // 2)),
        ]:
            cv2.putText(img, label, pos, font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_colony(self, img: np.ndarray, colony: ColonyMorphologyData):
        """Draw bounding box and index for a colony."""
        x1, y1, x2, y2 = [int(v) for v in colony.bbox]
        color_bgr = SIZE_COLORS.get(colony.size_category, (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)

        # Draw centroid dot
        cx, cy = int(colony.centroid_x), int(colony.centroid_y)
        cv2.circle(img, (cx, cy), 2, color_bgr, -1)

        # Colony index label
        label = str(colony.colony_index + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Background for label
        cv2.rectangle(
            img,
            (x1, y1 - th - 4),
            (x1 + tw + 4, y1),
            color_bgr,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_info_overlay(
        self,
        img: np.ndarray,
        total_count: int,
        cfu_formatted: str,
        qc_grade: str,
    ):
        """Draw info panel in top-left corner."""
        h, w = img.shape[:2]

        # Semi-transparent background
        overlay = img.copy()
        panel_w = min(320, w // 3)
        panel_h = 90
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)

        lines = [
            f"MicroCount v1.0",
            f"Colonies: {total_count}",
            f"CFU: {cfu_formatted[:25] if cfu_formatted else 'N/A'}",
            f"QC: {qc_grade}",
        ]

        qc_color = QC_GRADE_COLORS.get(qc_grade, (200, 200, 200))

        for i, line in enumerate(lines):
            text_color = qc_color if i == 3 else color
            cv2.putText(
                img,
                line,
                (10, 22 + i * 18),
                font,
                0.42,
                text_color,
                1,
                cv2.LINE_AA,
            )

    def save(self, image: np.ndarray, output_path: str):
        """Save annotated image to disk."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.debug(f"Annotated image saved: {output_path}")

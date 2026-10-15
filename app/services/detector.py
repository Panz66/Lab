"""
MicroCount - YOLO Colony Detection Module
YOLOv11-based colony detection with confidence filtering and NMS.
Falls back to classical CV when model weights not found.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


@dataclass
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    def __post_init__(self):
        self.centroid_x = (self.x1 + self.x2) / 2
        self.centroid_y = (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class YOLODetectionResult:
    boxes: list[DetectionBox] = field(default_factory=list)
    total_detected: int = 0
    model_version: str = "yolo11"
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0


class ColonyDetector:
    """
    YOLOv11-based colony detector.
    Supports automatic fallback to classical blob detection when model is absent.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._model_path = settings.yolo_model_path
        self._conf_threshold = settings.yolo_confidence_threshold
        self._iou_threshold = settings.yolo_nms_iou_threshold
        self._img_size = settings.yolo_img_size
        self._device = settings.device
        self._load_model()

    def _load_model(self):
        """Load YOLOv11 model. Graceful fallback if not found."""
        model_path = Path(self._model_path)

        if not model_path.exists():
            logger.warning(
                f"YOLO model not found at '{model_path}'. "
                "Using classical blob detection fallback. "
                "Train a model using: python scripts/train_yolo.py"
            )
            self._model_loaded = False
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
            self._model_loaded = True
            logger.info(f"YOLO model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model_loaded = False

    def detect(
        self,
        image: np.ndarray,
        petri_mask: Optional[np.ndarray] = None,
    ) -> YOLODetectionResult:
        """
        Detect colonies in image.
        Uses YOLO if loaded, else falls back to classical blob detection.
        """
        import time
        t0 = time.perf_counter()

        h, w = image.shape[:2]

        if self._model_loaded:
            result = self._yolo_detect(image)
        else:
            result = self._classical_detect(image)

        # Apply petri dish mask — remove detections outside petri area
        if petri_mask is not None and len(result.boxes) > 0:
            result.boxes = self._filter_by_mask(result.boxes, petri_mask)

        # Remove duplicates via NMS
        result.boxes = self._apply_nms(result.boxes, self._iou_threshold)

        result.total_detected = len(result.boxes)
        result.inference_time_ms = (time.perf_counter() - t0) * 1000
        result.image_width = w
        result.image_height = h

        logger.info(
            f"Detection complete: {result.total_detected} colonies | "
            f"method={'yolo' if self._model_loaded else 'classical'} | "
            f"{result.inference_time_ms:.1f}ms"
        )

        return result

    def _yolo_detect(self, image: np.ndarray) -> YOLODetectionResult:
        """Run YOLOv11 inference."""
        # Convert RGB to BGR for YOLO
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = self._model(
            bgr_image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            imgsz=self._img_size,
            device=self._device,
            verbose=False,
        )

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                if conf >= self._conf_threshold:
                    boxes.append(
                        DetectionBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                            class_id=cls,
                        )
                    )

        return YOLODetectionResult(boxes=boxes, model_version="yolo11")

    def _classical_detect(self, image: np.ndarray) -> YOLODetectionResult:
        """
        Classical blob detection fallback using SimpleBlobDetector + contour analysis.
        Used when YOLO weights are not available.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=5,
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Contours found: {len(contours)}")

        h, w = gray.shape
        min_area = (w * h) * 0.00005  # Minimum colony area relative to image
        max_area = (w * h) * 0.05

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            logger.info(f"Contour area: {area}")
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / max(bh, 1)

            # Reject very elongated shapes
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue

            # Estimate confidence from circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                conf = 0.4 + circularity * 0.3  # [0.40, 0.70] range
            else:
                conf = 0.40

            boxes.append(
                DetectionBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + bw),
                    y2=float(y + bh),
                    confidence=min(conf, 1.0),
                    class_id=0,
                )
            )

            logger.info(f"Boxes detected before NMS: {len(boxes)}")
        return YOLODetectionResult(boxes=boxes, model_version="classical_blob")

    def _filter_by_mask(
        self, boxes: list[DetectionBox], mask: np.ndarray
    ) -> list[DetectionBox]:
        """Keep only boxes whose centroid is inside the petri dish mask."""
        filtered = []
        for box in boxes:
            cx, cy = int(box.centroid_x), int(box.centroid_y)
            h, w = mask.shape[:2]
            if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] > 0:
                filtered.append(box)
        return filtered

    def _apply_nms(self, boxes: list[DetectionBox], iou_threshold: float) -> list[DetectionBox]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(boxes) == 0:
            return boxes

        # Sort by confidence descending
        boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)

        kept = []
        while boxes_sorted:
            best = boxes_sorted.pop(0)
            kept.append(best)
            boxes_sorted = [
                b for b in boxes_sorted
                if self._iou(best, b) < iou_threshold
            ]

        return kept

    def _iou(self, a: DetectionBox, b: DetectionBox) -> float:
        """Compute Intersection over Union of two boxes."""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)

        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union_area = a.area + b.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def reload_model(self):
        """Reload YOLO model (e.g., after training)."""
        self._load_model()


# Singleton
_detector: Optional[ColonyDetector] = None


def get_detector() -> ColonyDetector:
    global _detector
    if _detector is None:
        _detector = ColonyDetector()
    return _detector

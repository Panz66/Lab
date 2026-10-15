"""
MicroCount - Image Preprocessing Module
OpenCV-based image preprocessing pipeline for petri dish analysis.
Implements: CLAHE, Gaussian blur, noise reduction, illumination correction,
adaptive thresholding, and petri dish detection via Hough circles + contour analysis.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scipy import ndimage


@dataclass
class PetriDishDetectionResult:
    detected: bool
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0
    area_px: float = 0.0
    detection_method: str = "none"
    confidence: float = 0.0


@dataclass
class PreprocessingResult:
    original: np.ndarray
    preprocessed: np.ndarray
    petri_cropped: Optional[np.ndarray]
    petri_mask: Optional[np.ndarray]
    petri_info: PetriDishDetectionResult
    grayscale: np.ndarray
    enhanced: np.ndarray
    processing_time_ms: float = 0.0


@dataclass
class QualityMetrics:
    focus_score: float = 0.0
    blur_score: float = 0.0
    glare_score: float = 0.0
    shadow_score: float = 0.0
    visibility_score: float = 0.0
    overall_score: float = 0.0
    grade: str = "Poor"
    issues: list[str] = field(default_factory=list)


class ImagePreprocessor:
    """
    Full preprocessing pipeline for petri dish images.
    Applies: grayscale, CLAHE, Gaussian blur, noise reduction,
    illumination correction, adaptive thresholding, petri detection.
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image_path: str) -> PreprocessingResult:
        """Run the full preprocessing pipeline on an image."""
        t0 = time.perf_counter()

        # Load image
        original = self._load_image(image_path)
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")

        logger.info(f"Processing image: {Path(image_path).name} | shape={original.shape}")

        # Step 1: Convert to grayscale
        gray = self._to_grayscale(original)

        # Step 2: Illumination correction
        illumination_corrected = self._correct_illumination(gray)

        # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_applied = self._apply_clahe(illumination_corrected)

        # Step 4: Gaussian blur (noise reduction)
        blurred = self._apply_gaussian_blur(clahe_applied)

        # Step 5: Non-local means denoising
        denoised = self._denoise(blurred)

        # Step 6: Detect petri dish
        petri_info = self._detect_petri_dish(original, gray)

        # Step 7: Create mask and crop
        petri_mask = None
        petri_cropped = None
        if petri_info.detected:
            petri_mask = self._create_petri_mask(original.shape[:2], petri_info)
            petri_cropped = self._crop_petri(original, petri_info)

        # Step 8: Final preprocessed image for YOLO (original with enhancements)
        preprocessed = self._prepare_for_detection(original, petri_mask)

        processing_time_ms = (time.perf_counter() - t0) * 1000

        return PreprocessingResult(
            original=original,
            preprocessed=preprocessed,
            petri_cropped=petri_cropped,
            petri_mask=petri_mask,
            petri_info=petri_info,
            grayscale=gray,
            enhanced=clahe_applied,
            processing_time_ms=processing_time_ms,
        )

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image with error handling."""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            # Convert BGR to RGB for consistent processing
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Image load error: {e}")
            return None

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def _correct_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination using morphological top-hat transform.
        This removes the background gradient/shadow effect.
        """
        # Use large structuring element to estimate background
        kernel_size = max(gray.shape) // 8
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 101)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Background estimation via morphological opening
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Subtract background
        corrected = cv2.subtract(gray, background)

        # Normalize to full range
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        return corrected.astype(np.uint8)

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE for local contrast enhancement."""
        return self.clahe.apply(gray)

    def _apply_gaussian_blur(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Apply light Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        return cv2.fastNlMeansDenoising(image, h=7, templateWindowSize=7, searchWindowSize=21)

    def _detect_petri_dish(
        self, original: np.ndarray, gray: np.ndarray
    ) -> PetriDishDetectionResult:
        """
        Detect petri dish using Hough Circle Detection + Contour Analysis.
        Tries Hough first, falls back to contour analysis.
        """
        h, w = gray.shape

        # Method 1: Hough Circle Detection
        result = self._hough_circle_detection(gray, w, h)
        if result.detected and result.confidence > 0.6:
            logger.info(
                f"Petri dish detected via Hough: center=({result.center_x:.0f},{result.center_y:.0f}), r={result.radius:.0f}"
            )
            return result

        # Method 2: Contour Analysis
        result = self._contour_detection(gray, w, h)
        if result.detected:
            logger.info(
                f"Petri dish detected via Contour: center=({result.center_x:.0f},{result.center_y:.0f}), r={result.radius:.0f}"
            )
            return result

        # Fallback: assume entire image is petri dish
        logger.warning("Petri dish not detected. Using full image as petri area.")
        radius = min(w, h) * 0.45
        return PetriDishDetectionResult(
            detected=True,
            center_x=w / 2,
            center_y=h / 2,
            radius=radius,
            area_px=np.pi * radius ** 2,
            detection_method="fallback",
            confidence=0.3,
        )

    def _hough_circle_detection(
        self, gray: np.ndarray, w: int, h: int
    ) -> PetriDishDetectionResult:
        """Hough Circle Transform for petri dish detection."""
        min_dist = min(w, h) // 4
        min_radius = int(min(w, h) * 0.25)
        max_radius = int(min(w, h) * 0.52)

        # Pre-process for Hough
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=80,
            param2=40,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None:
            # Try with more permissive params
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=min_dist,
                param1=50,
                param2=25,
                minRadius=min_radius,
                maxRadius=max_radius,
            )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # Pick circle closest to image center
            img_cx, img_cy = w / 2, h / 2
            best = min(
                circles,
                key=lambda c: ((c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2) ** 0.5,
            )
            cx, cy, r = float(best[0]), float(best[1]), float(best[2])

            # Confidence based on center proximity and size
            center_dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
            max_dist = (w ** 2 + h ** 2) ** 0.5 / 2
            conf = 1.0 - (center_dist / max_dist) * 0.5

            return PetriDishDetectionResult(
                detected=True,
                center_x=cx,
                center_y=cy,
                radius=r,
                area_px=np.pi * r ** 2,
                detection_method="hough",
                confidence=conf,
            )

        return PetriDishDetectionResult(detected=False)

    def _contour_detection(
        self, gray: np.ndarray, w: int, h: int
    ) -> PetriDishDetectionResult:
        """Contour-based petri dish detection as fallback."""
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return PetriDishDetectionResult(detected=False)

        # Filter by area: petri dish should be large
        min_area = (min(w, h) * 0.3) ** 2
        valid = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid:
            return PetriDishDetectionResult(detected=False)

        # Pick largest contour
        largest = max(valid, key=cv2.contourArea)

        # Fit enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest)

        # Check circularity
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.5:
                return PetriDishDetectionResult(detected=False)

        return PetriDishDetectionResult(
            detected=True,
            center_x=float(cx),
            center_y=float(cy),
            radius=float(radius),
            area_px=np.pi * radius ** 2,
            detection_method="contour",
            confidence=0.7,
        )

    def _create_petri_mask(
        self, shape: tuple[int, int], petri: PetriDishDetectionResult
    ) -> np.ndarray:
        """Create binary mask for petri dish area."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(
            mask,
            (int(petri.center_x), int(petri.center_y)),
            int(petri.radius),
            255,
            -1,
        )
        return mask

    def _crop_petri(
        self, image: np.ndarray, petri: PetriDishDetectionResult
    ) -> np.ndarray:
        """Crop and extract petri dish area with background removed."""
        h, w = image.shape[:2]
        cx, cy, r = int(petri.center_x), int(petri.center_y), int(petri.radius)

        # Crop bounding box with padding
        pad = int(r * 0.05)
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(w, cx + r + pad)
        y2 = min(h, cy + r + pad)

        cropped = image[y1:y2, x1:x2].copy()

        # Apply circular mask
        crop_h, crop_w = cropped.shape[:2]
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.circle(
            mask,
            (cx - x1, cy - y1),
            r,
            255,
            -1,
        )

        # Black out background
        if len(cropped.shape) == 3:
            result = cv2.bitwise_and(cropped, cropped, mask=mask)
        else:
            result = cv2.bitwise_and(cropped, cropped, mask=mask)

        return result

    def _prepare_for_detection(self, original: np.ndarray, petri_mask: Optional[np.ndarray]) -> np.ndarray:
        """Prepare final image for YOLO detection — JANGAN ubah area dalam petri dish."""
    # Kirim original apa adanya ke YOLO, tanpa memodifikasi warna koloni
    # Hanya mask out background agar YOLO fokus ke area petri
        prepared = original.copy()
        if petri_mask is not None:
        # Validasi dulu: apakah mask cukup besar? (minimal 10% area gambar)
          mask_area = np.sum(petri_mask > 0)
        total_area = petri_mask.shape[0] * petri_mask.shape[1]
        if mask_area < total_area * 0.10:
            # Mask terlalu kecil, kemungkinan deteksi petri dish salah — abaikan mask
            return prepared
        
        if len(original.shape) == 3:
            mask_3ch = np.stack([petri_mask, petri_mask, petri_mask], axis=-1)
        else:
            mask_3ch = petri_mask
        background = np.full_like(prepared, 50)  # Abu gelap, bukan hitam total
        prepared = np.where(mask_3ch > 0, prepared, background).astype(np.uint8)
        return prepared

    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess image quality for laboratory standards.
        Returns scores for focus, blur, glare, shadow, visibility.
        """
        gray = self._to_grayscale(image)
        issues = []

        # 1. Focus score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        focus_score = min(1.0, laplacian_var / 500.0)
        if focus_score < 0.3:
            issues.append("Image appears out of focus")

        # 2. Blur score (inverse of focus)
        blur_score = 1.0 - focus_score
        if blur_score > 0.7:
            issues.append("Significant motion blur detected")

        # 3. Glare detection (overexposed regions)
        _, overexposed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(overexposed > 0) / gray.size
        glare_score = 1.0 - min(1.0, glare_ratio * 10)
        if glare_score < 0.6:
            issues.append("Glare or overexposure detected")

        # 4. Shadow detection (underexposed regions)
        _, underexposed = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        shadow_ratio = np.sum(underexposed > 0) / gray.size
        shadow_score = 1.0 - min(1.0, shadow_ratio * 5)
        if shadow_score < 0.6:
            issues.append("Shadow or underexposure detected")

        # 5. Petri dish visibility
        petri_info = self._detect_petri_dish(image, gray)
        visibility_score = petri_info.confidence if petri_info.detected else 0.2
        if not petri_info.detected:
            issues.append("Petri dish boundary not clearly visible")

        # Overall score (weighted)
        overall = (
            focus_score * 0.30
            + (1 - blur_score) * 0.20
            + glare_score * 0.20
            + shadow_score * 0.15
            + visibility_score * 0.15
        )

        # Grade
        if overall >= 0.85:
            grade = "Excellent"
        elif overall >= 0.70:
            grade = "Good"
        elif overall >= 0.50:
            grade = "Fair"
        else:
            grade = "Poor"

        return QualityMetrics(
            focus_score=round(focus_score, 3),
            blur_score=round(blur_score, 3),
            glare_score=round(glare_score, 3),
            shadow_score=round(shadow_score, 3),
            visibility_score=round(visibility_score, 3),
            overall_score=round(overall, 3),
            grade=grade,
            issues=issues,
        )


# Singleton instance
_preprocessor: Optional[ImagePreprocessor] = None


def get_preprocessor() -> ImagePreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor

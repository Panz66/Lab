"""
MicroCount - Analysis Orchestrator
Coordinates the full analysis pipeline:
Upload → Preprocess → Detect → Watershed → Morphology → CFU → Annotate
"""
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import (
    AnalysisParams,
    AnalysisResult,
    CFUResult,
    ColonyMorphology,
    ColonyResult,
    MorphologySummary,
    PetriDishInfo,
    QualityControlResult,
)
from app.services.annotator import ImageAnnotator
from app.services.cfu_calculator import CFUCalculator
from app.services.detector import get_detector
from app.services.morphology import MorphologyAnalyzer
from app.services.preprocessor import get_preprocessor
from app.services.watershed import get_separator

settings = get_settings()


class AnalysisPipeline:
    """
    Full end-to-end analysis pipeline for petri dish colony counting.
    """

    def __init__(self):
        self.preprocessor = get_preprocessor()
        self.detector = get_detector()
        self.separator = get_separator()
        self.morphology_analyzer = MorphologyAnalyzer()
        self.cfu_calculator = CFUCalculator()
        self.annotator = ImageAnnotator()

    async def analyze(
        self,
        image_path: str,
        params: AnalysisParams,
        analysis_id: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Run the complete analysis pipeline.

        Returns:
            AnalysisResult with all counts, morphology, CFU, QC data.
        """
        t0 = time.perf_counter()
        analysis_id = analysis_id or str(uuid.uuid4())
        warnings = []

        logger.info(f"[{analysis_id}] Starting analysis: {Path(image_path).name}")

        # ── Step 1: Preprocess Image ─────────────────────────────────────
        logger.info(f"[{analysis_id}] Step 1/6: Preprocessing...")
        prep_result = self.preprocessor.preprocess(image_path)

        # ── Step 2: Quality Control Assessment ─────────────────────────────
        logger.info(f"[{analysis_id}] Step 2/6: Quality assessment...")
        qc_metrics = self.preprocessor.assess_quality(prep_result.original)

        if qc_metrics.grade in ("Poor",):
            warnings.append(f"Poor image quality ({qc_metrics.grade}): {', '.join(qc_metrics.issues)}")

        # ── Step 3: Colony Detection (YOLO) ───────────────────────────────
        logger.info(f"[{analysis_id}] Step 3/6: Colony detection...")
        detection_result = self.detector.detect(
            prep_result.preprocessed,
            petri_mask=prep_result.petri_mask,
        )

        # ── Step 4: Watershed Separation ─────────────────────────────────
        logger.info(f"[{analysis_id}] Step 4/6: Watershed separation...")
        watershed_result = self.separator.separate(
            prep_result.preprocessed,
            detection_result.boxes,
            petri_mask=prep_result.petri_mask,
        )

        final_boxes = watershed_result.separated_boxes
        if watershed_result.separation_count > 0:
            logger.info(
                f"[{analysis_id}] Watershed separated {watershed_result.separation_count} clusters"
            )

        # ── Step 5: Morphology Analysis ──────────────────────────────────
        logger.info(f"[{analysis_id}] Step 5/6: Morphology analysis...")
        petri = prep_result.petri_info
        colonies_morph = self.morphology_analyzer.analyze_all(
            prep_result.original,
            final_boxes,
            petri_radius_px=petri.radius if petri.detected else None,
        )

        # Assign quadrants
        if petri.detected:
            colonies_morph = self.morphology_analyzer.compute_quadrants(
                colonies_morph,
                center_x=petri.center_x,
                center_y=petri.center_y,
            )

        # ── Step 6: CFU Calculation ─────────────────────────────────────
        logger.info(f"[{analysis_id}] Step 6/6: CFU calculation...")
        total_colonies = len(final_boxes)
        cfu = self.cfu_calculator.calculate(
            total_colonies=total_colonies,
            dilution_factor=params.dilution_factor,
            inoculation_volume_ml=params.inoculation_volume_ml,
        )

        if cfu.count_out_of_range:
            warnings.append(cfu.count_range_message)

        # ── Build Results ────────────────────────────────────────────────
        # Colony density (colonies per cm²)
        colony_density = 0.0
        if petri.detected and petri.area_px > 0:
            # Approximate px/cm: assume 96 dpi = 37.8 px/cm
            # Use petri dish as calibration: 9cm diameter standard
            px_per_cm = (2 * petri.radius) / 9.0
            area_cm2 = petri.area_px / (px_per_cm ** 2)
            colony_density = round(total_colonies / area_cm2, 2) if area_cm2 > 0 else 0.0

        # Per-quadrant counts
        q_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for c in colonies_morph:
            if c.quadrant:
                q_counts[c.quadrant] = q_counts.get(c.quadrant, 0) + 1

        # Morphology summary
        morph_summary_dict = self.morphology_analyzer.summarize(colonies_morph)

        # ── Generate annotated image ─────────────────────────────────────
        annotated_path = None
        try:
            annotated = self.annotator.annotate(
                prep_result.original,
                colonies_morph,
                petri_center=(petri.center_x, petri.center_y) if petri.detected else None,
                petri_radius=petri.radius if petri.detected else None,
                qc_grade=qc_metrics.grade,
                total_count=total_colonies,
                cfu_formatted=cfu.cfu_per_ml_formatted,
            )
            annotated_dir = Path(settings.upload_dir) / "annotated"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = str(annotated_dir / f"{analysis_id}_annotated.jpg")
            self.annotator.save(annotated, annotated_path)
        except Exception as e:
            logger.warning(f"[{analysis_id}] Annotation failed: {e}")

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{analysis_id}] Analysis complete: {total_colonies} colonies, "
            f"CFU={cfu.cfu_per_ml_formatted}, QC={qc_metrics.grade} ({total_ms:.0f}ms)"
        )

        # ── Assemble Response ────────────────────────────────────────────
        return AnalysisResult(
            analysis_id=analysis_id,
            sample_id=params.sample_id,
            sample_name=params.sample_name,
            analyst=params.analyst,
            processing_time_ms=round(total_ms, 1),
            petri_dish=PetriDishInfo(
                detected=petri.detected,
                center_x=petri.center_x if petri.detected else None,
                center_y=petri.center_y if petri.detected else None,
                radius=petri.radius if petri.detected else None,
                area_px=petri.area_px if petri.detected else None,
                detection_method=petri.detection_method,
            ),
            total_colonies=total_colonies,
            colonies_per_quadrant={f"Q{k}": v for k, v in q_counts.items()},
            colony_density=colony_density,
            cfu=CFUResult(
                total_colonies=total_colonies,
                dilution_factor=params.dilution_factor,
                inoculation_volume_ml=params.inoculation_volume_ml,
                cfu_per_ml=cfu.cfu_per_ml,
                cfu_per_ml_formatted=cfu.cfu_per_ml_formatted,
                scientific_notation=cfu.scientific_notation,
            ),
            quality_control=QualityControlResult(
                score=qc_metrics.overall_score,
                grade=qc_metrics.grade,
                focus_score=qc_metrics.focus_score,
                blur_score=qc_metrics.blur_score,
                glare_score=qc_metrics.glare_score,
                shadow_score=qc_metrics.shadow_score,
                visibility_score=qc_metrics.visibility_score,
                issues=qc_metrics.issues,
            ),
            morphology_summary=MorphologySummary(**morph_summary_dict) if morph_summary_dict else MorphologySummary(),
            colonies=[
                ColonyResult(
                    colony_index=c.colony_index,
                    bbox=c.bbox,
                    centroid=(c.centroid_x, c.centroid_y),
                    confidence=c.confidence,
                    quadrant=c.quadrant,
                    morphology=ColonyMorphology(
                        diameter_px=c.diameter_px,
                        diameter_mm=c.diameter_mm,
                        area_px=c.area_px,
                        circularity=c.circularity,
                        mean_intensity=c.mean_intensity,
                        color_rgb=(c.color_r, c.color_g, c.color_b),
                        opacity=c.opacity,
                        edge_smoothness=c.edge_smoothness,
                        size_category=c.size_category,
                    ),
                    was_separated=c.was_separated,
                )
                for c in colonies_morph
            ],
            annotated_image_path=annotated_path,
            warnings=warnings,
        )

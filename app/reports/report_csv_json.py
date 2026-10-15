"""
MicroCount - CSV and JSON Report Generators
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import get_settings
from app.models.schemas import AnalysisResult

settings = get_settings()


class CSVReportGenerator:
    """Generates CSV reports for analysis results."""

    def generate(
        self,
        result: AnalysisResult,
        sample_name: str,
        analyst: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        if not output_path:
            output_dir = Path(settings.csv_report_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{result.analysis_id}_report.csv")

        # Sheet 1: Summary (header rows)
        summary_rows = [
            ["MicroCount - Colony Count Report"],
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            [""],
            ["SAMPLE INFORMATION"],
            ["Sample Name", sample_name],
            ["Analyst", analyst],
            ["Media Type", kwargs.get("media_type", "")],
            ["Organism Type", kwargs.get("organism_type", "")],
            ["Incubation Date", kwargs.get("incubation_date", "")],
            ["Observation Date", kwargs.get("observation_date", "")],
            ["Analysis ID", result.analysis_id],
            [""],
            ["ANALYSIS RESULTS"],
            ["Total Colonies", result.total_colonies],
            ["Colony Density (col/cm²)", result.colony_density],
            ["Q1 Colonies", result.colonies_per_quadrant.get("Q1", 0)],
            ["Q2 Colonies", result.colonies_per_quadrant.get("Q2", 0)],
            ["Q3 Colonies", result.colonies_per_quadrant.get("Q3", 0)],
            ["Q4 Colonies", result.colonies_per_quadrant.get("Q4", 0)],
            [""],
            ["CFU RESULTS"],
            ["CFU/mL", result.cfu.cfu_per_ml],
            ["CFU/mL Formatted", result.cfu.cfu_per_ml_formatted],
            ["Scientific Notation", result.cfu.scientific_notation],
            ["Dilution Factor", result.cfu.dilution_factor],
            ["Inoculation Volume (mL)", result.cfu.inoculation_volume_ml],
            [""],
            ["QUALITY CONTROL"],
            ["QC Grade", result.quality_control.grade],
            ["QC Score", result.quality_control.score],
            ["Focus Score", result.quality_control.focus_score],
            ["Blur Score", result.quality_control.blur_score],
            ["Glare Score", result.quality_control.glare_score],
            ["Shadow Score", result.quality_control.shadow_score],
            ["Visibility Score", result.quality_control.visibility_score],
            [""],
            ["MORPHOLOGY SUMMARY"],
            ["Pinpoint Colonies", result.morphology_summary.pinpoint_count],
            ["Small Colonies", result.morphology_summary.small_count],
            ["Medium Colonies", result.morphology_summary.medium_count],
            ["Large Colonies", result.morphology_summary.large_count],
            ["Mean Diameter (mm)", result.morphology_summary.mean_diameter_mm or ""],
            ["Mean Circularity", result.morphology_summary.mean_circularity or ""],
            ["Dominant Color", result.morphology_summary.dominant_color or ""],
            [""],
            ["INDIVIDUAL COLONIES"],
            [
                "Colony #", "Centroid X", "Centroid Y", "Quadrant",
                "Confidence", "Size Category", "Diameter (px)", "Diameter (mm)",
                "Area (px²)", "Circularity", "Mean Intensity",
                "Color R", "Color G", "Color B", "Opacity", "Edge Smoothness",
                "Was Separated", "BBox X1", "BBox Y1", "BBox X2", "BBox Y2",
            ],
        ]

        # Colony rows
        colony_rows = []
        for c in result.colonies:
            m = c.morphology
            colony_rows.append([
                c.colony_index + 1,
                round(c.centroid[0], 2),
                round(c.centroid[1], 2),
                c.quadrant or "",
                round(c.confidence, 4),
                m.size_category if m else "",
                round(m.diameter_px, 2) if m and m.diameter_px else "",
                round(m.diameter_mm, 3) if m and m.diameter_mm else "",
                round(m.area_px, 2) if m and m.area_px else "",
                round(m.circularity, 4) if m and m.circularity else "",
                round(m.mean_intensity, 2) if m and m.mean_intensity else "",
                round(m.color_rgb[0], 1) if m and m.color_rgb else "",
                round(m.color_rgb[1], 1) if m and m.color_rgb else "",
                round(m.color_rgb[2], 1) if m and m.color_rgb else "",
                round(m.opacity, 4) if m and m.opacity else "",
                round(m.edge_smoothness, 4) if m and m.edge_smoothness else "",
                "Yes" if c.was_separated else "No",
                round(c.bbox[0], 2),
                round(c.bbox[1], 2),
                round(c.bbox[2], 2),
                round(c.bbox[3], 2),
            ])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(summary_rows)
            writer.writerows(colony_rows)

        logger.info(f"CSV report generated: {output_path}")
        return output_path


class JSONReportGenerator:
    """Generates structured JSON reports."""

    def generate(
        self,
        result: AnalysisResult,
        sample_name: str,
        analyst: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        if not output_path:
            output_dir = Path(settings.json_report_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{result.analysis_id}_report.json")

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "MicroCount v1.0",
                "analysis_id": result.analysis_id,
                "standard": "ISO 7218 / USP / APHA",
            },
            "sample_information": {
                "sample_name": sample_name,
                "analyst": analyst,
                "media_type": kwargs.get("media_type"),
                "organism_type": kwargs.get("organism_type"),
                "incubation_date": kwargs.get("incubation_date"),
                "observation_date": kwargs.get("observation_date"),
                "notes": kwargs.get("notes"),
            },
            "petri_dish": {
                "detected": result.petri_dish.detected,
                "center_x": result.petri_dish.center_x,
                "center_y": result.petri_dish.center_y,
                "radius_px": result.petri_dish.radius,
                "area_px": result.petri_dish.area_px,
                "detection_method": result.petri_dish.detection_method,
            },
            "colony_count": {
                "total": result.total_colonies,
                "per_quadrant": result.colonies_per_quadrant,
                "density_per_cm2": result.colony_density,
            },
            "cfu_calculation": {
                "cfu_per_ml": result.cfu.cfu_per_ml,
                "cfu_per_ml_formatted": result.cfu.cfu_per_ml_formatted,
                "scientific_notation": result.cfu.scientific_notation,
                "dilution_factor": result.cfu.dilution_factor,
                "inoculation_volume_ml": result.cfu.inoculation_volume_ml,
                "formula": "CFU/mL = Total Colonies × Dilution Factor / Volume (mL)",
            },
            "quality_control": {
                "grade": result.quality_control.grade,
                "overall_score": result.quality_control.score,
                "metrics": {
                    "focus": result.quality_control.focus_score,
                    "blur": result.quality_control.blur_score,
                    "glare": result.quality_control.glare_score,
                    "shadow": result.quality_control.shadow_score,
                    "visibility": result.quality_control.visibility_score,
                },
                "issues": result.quality_control.issues,
            },
            "morphology_summary": {
                "size_distribution": {
                    "pinpoint": result.morphology_summary.pinpoint_count,
                    "small": result.morphology_summary.small_count,
                    "medium": result.morphology_summary.medium_count,
                    "large": result.morphology_summary.large_count,
                },
                "mean_diameter_mm": result.morphology_summary.mean_diameter_mm,
                "mean_circularity": result.morphology_summary.mean_circularity,
                "dominant_color": result.morphology_summary.dominant_color,
                "size_distribution_percent": result.morphology_summary.size_distribution,
            },
            "processing": {
                "processing_time_ms": result.processing_time_ms,
                "warnings": result.warnings,
                "status": result.status,
            },
            "colonies": [
                {
                    "index": c.colony_index + 1,
                    "centroid": {"x": round(c.centroid[0], 2), "y": round(c.centroid[1], 2)},
                    "quadrant": c.quadrant,
                    "confidence": round(c.confidence, 4),
                    "was_separated": c.was_separated,
                    "bbox": {
                        "x1": round(c.bbox[0], 2),
                        "y1": round(c.bbox[1], 2),
                        "x2": round(c.bbox[2], 2),
                        "y2": round(c.bbox[3], 2),
                    },
                    "morphology": {
                        "size_category": c.morphology.size_category if c.morphology else None,
                        "diameter_px": c.morphology.diameter_px if c.morphology else None,
                        "diameter_mm": c.morphology.diameter_mm if c.morphology else None,
                        "area_px": c.morphology.area_px if c.morphology else None,
                        "circularity": c.morphology.circularity if c.morphology else None,
                        "mean_intensity": c.morphology.mean_intensity if c.morphology else None,
                        "color_rgb": list(c.morphology.color_rgb) if c.morphology and c.morphology.color_rgb else None,
                        "opacity": c.morphology.opacity if c.morphology else None,
                        "edge_smoothness": c.morphology.edge_smoothness if c.morphology else None,
                    } if c.morphology else None,
                }
                for c in result.colonies
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report generated: {output_path}")
        return output_path

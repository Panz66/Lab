"""
MicroCount - Database Models (FIXED)
Perubahan:
  1. Report.analysis_id — tambahkan cascade ondelete="CASCADE" agar saat Analysis
     dihapus, Report ikut terhapus otomatis di level DB (defence-in-depth).
  2. Analysis.report relationship — tambahkan cascade="all, delete-orphan"
     agar SQLAlchemy ORM juga delete Report saat Analysis dihapus.
  3. Analysis.colonies relationship sudah ada cascade="all, delete-orphan" — OK.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Sample(Base):
    """Sampel yang bisa ditautkan ke satu atau lebih analisis."""
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    sample_name: Mapped[str] = mapped_column(String(255), nullable=False)
    analyst: Mapped[str] = mapped_column(String(255), nullable=False)
    incubation_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    observation_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    media_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    organism_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    analyses: Mapped[list["Analysis"]] = relationship(
        "Analysis", back_populates="sample", cascade="save-update, merge"
    )


class Analysis(Base):
    """Analysis result for one image processing run."""
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)

    # nullable agar analisis bisa berdiri sendiri meski sampel dihapus
    sample_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("samples.id", ondelete="SET NULL"), nullable=True
    )

    # Nama sampel & analis disimpan langsung agar riwayat tetap terbaca
    sample_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    analyst: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Image info
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Petri dish detection
    petri_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    petri_center_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    petri_center_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    petri_radius: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    petri_area_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Colony counting results
    total_colonies: Mapped[int] = mapped_column(Integer, default=0)
    colonies_q1: Mapped[int] = mapped_column(Integer, default=0)
    colonies_q2: Mapped[int] = mapped_column(Integer, default=0)
    colonies_q3: Mapped[int] = mapped_column(Integer, default=0)
    colonies_q4: Mapped[int] = mapped_column(Integer, default=0)
    colony_density: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # CFU calculation
    dilution_factor: Mapped[float] = mapped_column(Float, default=1.0)
    inoculation_volume_ml: Mapped[float] = mapped_column(Float, default=1.0)
    cfu_per_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cfu_per_ml_formatted: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Quality control
    qc_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_grade: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    qc_focus_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_blur_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_glare_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_shadow_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_visibility_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    qc_issues: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Morphology summary
    morphology_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Processing metadata
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Notes / free-text field
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Result image paths
    annotated_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    preprocessed_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    heatmap_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    sample: Mapped[Optional["Sample"]] = relationship("Sample", back_populates="analyses")
    colonies: Mapped[list["Colony"]] = relationship(
        "Colony", back_populates="analysis", cascade="all, delete-orphan"
    )
    # FIX: cascade delete-orphan agar Report ikut terhapus saat Analysis dihapus ORM
    report: Mapped[Optional["Report"]] = relationship(
        "Report", back_populates="analysis", uselist=False,
        cascade="all, delete-orphan"
    )


class Colony(Base):
    """Individual colony detected in an analysis."""
    __tablename__ = "colonies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id"), nullable=False
    )
    colony_index: Mapped[int] = mapped_column(Integer, nullable=False)

    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    centroid_x: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_y: Mapped[float] = mapped_column(Float, nullable=False)

    quadrant: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    diameter_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    diameter_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    area_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    circularity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    color_r: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    color_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    color_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opacity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_smoothness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    size_category: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    was_separated: Mapped[bool] = mapped_column(Boolean, default=False)

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="colonies")


class Report(Base):
    """Generated reports for an analysis."""
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    # FIX: tambahkan ondelete="CASCADE" agar hapus di level DB juga aman
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    pdf_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    csv_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    json_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="report")
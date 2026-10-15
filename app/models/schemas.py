"""
MicroCount - Pydantic Schemas
API request/response models
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────────
# Sample Schemas
# ─────────────────────────────────────────────
class SampleCreate(BaseModel):
    sample_name: str = Field(..., min_length=1, max_length=255)
    analyst: str = Field(..., min_length=1, max_length=255)
    incubation_date: Optional[str] = None
    observation_date: Optional[str] = None
    media_type: Optional[str] = None
    organism_type: Optional[str] = None
    notes: Optional[str] = None


class SampleResponse(SampleCreate):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    updated_at: datetime

    # Alias agar frontend bisa pakai s.name (backward compat)
    @property
    def name(self) -> str:
        return self.sample_name

    def model_post_init(self, __context):
        pass

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        # Tambahkan 'name' ke response JSON agar FE yang pakai s.name tetap berjalan
        d["name"] = d.get("sample_name", "")
        return d


# ─────────────────────────────────────────────
# Analysis Schemas
# ─────────────────────────────────────────────
class AnalysisParams(BaseModel):
    sample_id: Optional[str] = None
    sample_name: str = Field("Sample", min_length=1, max_length=255)
    analyst: str = Field("Analyst", min_length=1, max_length=255)
    dilution_factor: float = Field(1.0, gt=0)
    inoculation_volume_ml: float = Field(1.0, gt=0)
    incubation_date: Optional[str] = None
    observation_date: Optional[str] = None
    media_type: Optional[str] = None
    organism_type: Optional[str] = None
    notes: Optional[str] = None


class QualityControlResult(BaseModel):
    score: float
    grade: str
    focus_score: float
    blur_score: float
    glare_score: float
    shadow_score: float
    visibility_score: float
    issues: list[str]


class ColonyMorphology(BaseModel):
    diameter_px: Optional[float] = None
    diameter_mm: Optional[float] = None
    area_px: Optional[float] = None
    circularity: Optional[float] = None
    mean_intensity: Optional[float] = None
    color_rgb: Optional[tuple[float, float, float]] = None
    opacity: Optional[float] = None
    edge_smoothness: Optional[float] = None
    size_category: str = "medium"


class ColonyResult(BaseModel):
    colony_index: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    confidence: float
    quadrant: Optional[int] = None
    morphology: Optional[ColonyMorphology] = None
    was_separated: bool = False


class MorphologySummary(BaseModel):
    pinpoint_count: int = 0
    small_count: int = 0
    medium_count: int = 0
    large_count: int = 0
    mean_diameter_mm: Optional[float] = None
    mean_circularity: Optional[float] = None
    dominant_color: Optional[str] = None
    size_distribution: dict[str, float] = {}


class CFUResult(BaseModel):
    total_colonies: int
    dilution_factor: float
    inoculation_volume_ml: float
    cfu_per_ml: float
    cfu_per_ml_formatted: str
    scientific_notation: str


class PetriDishInfo(BaseModel):
    detected: bool
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    radius: Optional[float] = None
    area_px: Optional[float] = None
    detection_method: str = "hough"


class AnalysisResult(BaseModel):
    analysis_id: str
    sample_id: Optional[str] = None
    sample_name: str
    analyst: str
    processing_time_ms: float
    petri_dish: PetriDishInfo
    total_colonies: int
    colonies_per_quadrant: dict[str, int]
    colony_density: float
    cfu: CFUResult
    quality_control: QualityControlResult
    morphology_summary: MorphologySummary
    colonies: list[ColonyResult]
    annotated_image_path: Optional[str] = None
    preprocessed_image_path: Optional[str] = None
    status: str = "completed"
    warnings: list[str] = []


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    sample_id: Optional[str] = None
    sample_name: Optional[str] = None   # disimpan langsung di kolom analyses
    analyst: Optional[str] = None
    original_filename: str
    total_colonies: int
    cfu_per_ml: Optional[float] = None
    cfu_per_ml_formatted: Optional[str] = None
    qc_grade: Optional[str] = None
    status: str
    created_at: datetime


class ReportRequest(BaseModel):
    analysis_id: str
    formats: list[str] = Field(default=["pdf", "csv", "json"])


class ReportResponse(BaseModel):
    analysis_id: str
    pdf_url: Optional[str] = None
    csv_url: Optional[str] = None
    json_url: Optional[str] = None
    created_at: datetime


# ─────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────
class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    per_page: int
    pages: int

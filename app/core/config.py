"""
MicroCount - Application Configuration
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "MicroCount"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "change-this-to-random-secret-key"
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Database
    database_url: str = "sqlite+aiosqlite:///./microcount.db"

    # File storage
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 20
    allowed_extensions: list[str] = ["jpg", "jpeg", "png", "webp"]

    # YOLO Model
    yolo_model_path: str = "weights/colony_yolo11.pt"
    yolo_confidence_threshold: float = 0.05
    yolo_nms_iou_threshold: float = 0.45
    yolo_img_size: int = 640
    device: str = "cpu"

    # Reports
    report_dir: str = "reports"
    pdf_report_dir: str = "reports/pdf"
    csv_report_dir: str = "reports/csv"
    json_report_dir: str = "reports/json"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/microcount.log"

    # Gemini AI
    gemini_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self):
        """Create required directories if they don't exist."""
        dirs = [
            self.upload_dir,
            self.pdf_report_dir,
            self.csv_report_dir,
            self.json_report_dir,
            "logs",
            "weights",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
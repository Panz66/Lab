"""
MicroCount - FastAPI Application
FIXED VERSION — Semua bug diperbaiki:
  1. DELETE /api/samples/{id} — hapus Report terkait sebelum nullify sample_id
  2. DELETE /api/analyses/{id} — hapus Report sebelum hapus Analysis
  3. generate_reports — metadata (media_type, dates, notes) diteruskan ke semua generator
  4. POST /api/reports/generate-session — generate_dev_session_report
  5. PATCH /api/analyses/{id}/notes — update notes analisis
  6. download_report — header Content-Disposition yang benar
  7. FIX: GET /api/images/{id}/annotated|original|preprocessed — serve image files
  8. FIX: POST /api/development/sessions — bulk kuantitatif (proses 9 foto)
  9. FIX: GET/DELETE /api/development/sessions & /{id}/result & /{id}/report
  10. FIX: POST /api/development/qualitative-sessions — bulk kualitatif (proses 9 foto)
  11. FIX: GET/DELETE /api/development/qualitative-sessions & /{id}/report
"""
import json
import os
import time
import uuid
import statistics
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import aiofiles
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from sqlalchemy import delete as sql_delete, desc, func, select, update as sql_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db, init_db
from app.core.logging import setup_logging
from app.models import database as db_models
from app.models.schemas import (
    AnalysisParams,
    AnalysisResponse,
    AnalysisResult,
    PaginatedResponse,
    ReportRequest,
    ReportResponse,
    SampleCreate,
    SampleResponse,
)
SampleUpdate = SampleCreate
from app.services.pipeline import AnalysisPipeline
from app.reports.report_csv_json import CSVReportGenerator, JSONReportGenerator
from app.reports.report_generator import PDFReportGenerator

settings = get_settings()
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    settings.ensure_directories()
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down MicroCount")


app = FastAPI(
    title="MicroCount API",
    description="Microbial Colony Counter — Computer Vision & AI-powered petri dish analysis",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = AnalysisPipeline()
pdf_generator = PDFReportGenerator()
csv_generator = CSVReportGenerator()
json_generator = JSONReportGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# Frontend Routes
# ─────────────────────────────────────────────────────────────────────────────

async def _serve_html(filename: str, not_found_label: str) -> HTMLResponse:
    html_path = Path("frontend") / filename
    if html_path.exists():
        async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(await f.read())
    return HTMLResponse(f"<h1>{not_found_label} - Frontend not found</h1>", status_code=404)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_root():
    """Halaman pertama saat server dibuka: portal iLaboratories."""
    return await _serve_html("ilaboratories.html", "iLaboratories")


@app.get("/ilaboratories.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_ilaboratories():
    return await _serve_html("ilaboratories.html", "iLaboratories")


@app.get("/microcount.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_microcount():
    return await _serve_html("microcount_frontend.html", "MicroCount")


@app.get("/stockopname.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_stockopname():
    return await _serve_html("stockopname.html", "Stock Opname")

@app.get("/schedule.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_schedule():
    return await _serve_html("schedule.html", "Schedule")

@app.get("/instrument.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_instrument():
    return await _serve_html("instrument.html", "Instrument")

    
# ─────────────────────────────────────────────────────────────────────────────
# Image Serving
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/images/{analysis_id}/annotated")
async def get_annotated_image(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Serve gambar hasil anotasi."""
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    img_path = analysis.annotated_image_path
    if not img_path:
        raise HTTPException(status_code=404, detail="Annotated image not recorded")
    file_path = Path(img_path)
    if not file_path.exists():
        fallback = Path(settings.upload_dir) / "annotated" / f"{analysis_id}_annotated.jpg"
        if fallback.exists():
            file_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"Annotated image not found on disk: {img_path}")
    media_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=str(file_path), media_type=media_type,
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/images/{analysis_id}/original")
async def get_original_image(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Serve gambar original."""
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    stored = analysis.stored_filename
    if not stored:
        raise HTTPException(status_code=404, detail="Stored filename not recorded")
    file_path = Path(settings.upload_dir) / "originals" / stored
    if not file_path.exists():
        file_path = Path(settings.upload_dir) / stored
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Original image not found: {stored}")
    media_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=str(file_path), media_type=media_type,
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/images/{analysis_id}/preprocessed")
async def get_preprocessed_image(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Serve gambar preprocessed."""
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    img_path = analysis.preprocessed_image_path
    if not img_path:
        raise HTTPException(status_code=404, detail="Preprocessed image not recorded")
    file_path = Path(img_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Preprocessed image not found: {img_path}")
    media_type = "image/png" if file_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path=str(file_path), media_type=media_type,
                        headers={"Cache-Control": "public, max-age=3600"})


# ─────────────────────────────────────────────────────────────────────────────
# Health & System
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
    }


@app.get("/api/system/info")
async def system_info():
    model_path = Path(settings.yolo_model_path)
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model_loaded": model_path.exists(),
        "model_path": str(model_path),
        "device": settings.device,
        "confidence_threshold": settings.yolo_confidence_threshold,
        "max_upload_mb": settings.max_upload_size_mb,
        "allowed_formats": settings.allowed_extensions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sample Management
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/samples", response_model=SampleResponse, status_code=status.HTTP_201_CREATED)
async def create_sample(
    sample: SampleCreate,
    db: AsyncSession = Depends(get_db),
):
    db_sample = db_models.Sample(
        id=str(uuid.uuid4()),
        **sample.model_dump(),
    )
    db.add(db_sample)
    await db.flush()
    await db.refresh(db_sample)
    return db_sample


@app.get("/api/samples", response_model=PaginatedResponse)
async def list_samples(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * per_page
    total_result = await db.execute(select(func.count()).select_from(db_models.Sample))
    total = total_result.scalar() or 0
    result = await db.execute(
        select(db_models.Sample)
        .order_by(desc(db_models.Sample.created_at))
        .offset(offset)
        .limit(per_page)
    )
    samples = result.scalars().all()
    return PaginatedResponse(
        items=[SampleResponse.model_validate(s) for s in samples],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page,
    )


@app.get("/api/samples/{sample_id}", response_model=SampleResponse)
async def get_sample(sample_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(db_models.Sample).where(db_models.Sample.id == sample_id)
    )
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return sample


@app.put("/api/samples/{sample_id}", response_model=SampleResponse)
async def update_sample(
    sample_id: str,
    sample: SampleCreate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(db_models.Sample).where(db_models.Sample.id == sample_id)
    )
    db_sample = result.scalar_one_or_none()
    if not db_sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    for key, value in sample.model_dump(exclude_unset=False).items():
        setattr(db_sample, key, value)
    await db.flush()
    await db.refresh(db_sample)
    return db_sample


@app.delete("/api/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sample(sample_id: str, db: AsyncSession = Depends(get_db)):
    """
    FIX: Urutan penghapusan yang aman untuk mencegah NOT NULL constraint error.
    1. Hapus semua Report yang terkait dengan analyses milik sample ini
    2. Nullify analysis.sample_id (pertahankan data analisis di riwayat)
    3. Hapus Sample
    """
    result = await db.execute(
        select(db_models.Sample).where(db_models.Sample.id == sample_id)
    )
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    # Cari analysis_id yang terkait sample ini
    analyses_result = await db.execute(
        select(db_models.Analysis.id).where(db_models.Analysis.sample_id == sample_id)
    )
    analysis_ids = [row[0] for row in analyses_result.fetchall()]

    if analysis_ids:
        # FIX: Hapus Report records dulu agar tidak melanggar NOT NULL constraint
        await db.execute(
            sql_delete(db_models.Report).where(
                db_models.Report.analysis_id.in_(analysis_ids)
            )
        )
        await db.flush()

        # Nullify sample_id pada analyses
        await db.execute(
            sql_update(db_models.Analysis)
            .where(db_models.Analysis.sample_id == sample_id)
            .values(sample_id=None)
        )
        await db.flush()

    await db.delete(sample)
    await db.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis — Single Image
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/analyze", status_code=status.HTTP_201_CREATED)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_name: str = Form("Sample"),
    analyst: str = Form("Analyst"),
    dilution_factor: float = Form(1.0),
    inoculation_volume_ml: float = Form(1.0),
    media_type: Optional[str] = Form(None),
    organism_type: Optional[str] = Form(None),
    incubation_date: Optional[str] = Form(None),
    observation_date: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload dan analisis gambar petri dish.
    Jika sample_id tidak diberikan, otomatis membuat record Sample baru.
    """
    await _validate_upload(file)

    analysis_id = str(uuid.uuid4())

    upload_dir = Path(settings.upload_dir) / "originals"
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    stored_filename = f"{analysis_id}{ext}"
    file_path = upload_dir / stored_filename

    content = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    file_size = len(content)

    resolved_sample_name: str = sample_name.strip() or "Sample"
    resolved_analyst: str = analyst.strip() or "Analyst"
    validated_sample_id: Optional[str] = None

    if sample_id and sample_id.strip():
        sample_check = await db.execute(
            select(db_models.Sample).where(db_models.Sample.id == sample_id.strip())
        )
        db_sample_obj = sample_check.scalar_one_or_none()
        if db_sample_obj:
            validated_sample_id = sample_id.strip()
            resolved_sample_name = db_sample_obj.sample_name or resolved_sample_name
            resolved_analyst = db_sample_obj.analyst or resolved_analyst
        else:
            logger.warning(f"sample_id '{sample_id}' tidak ditemukan — akan dibuat otomatis")

    if not validated_sample_id:
        new_sample = db_models.Sample(
            id=str(uuid.uuid4()),
            sample_name=resolved_sample_name,
            analyst=resolved_analyst,
            media_type=media_type,
            organism_type=organism_type,
            incubation_date=incubation_date,
            observation_date=observation_date,
            notes=notes,
        )
        db.add(new_sample)
        await db.flush()
        validated_sample_id = new_sample.id
        logger.info(f"Auto-created Sample '{resolved_sample_name}' id={validated_sample_id}")

    params = AnalysisParams(
        sample_id=validated_sample_id,
        sample_name=resolved_sample_name,
        analyst=resolved_analyst,
        dilution_factor=dilution_factor,
        inoculation_volume_ml=inoculation_volume_ml,
        media_type=media_type,
        organism_type=organism_type,
        incubation_date=incubation_date,
        observation_date=observation_date,
        notes=notes,
    )

    db_analysis = db_models.Analysis(
        id=analysis_id,
        sample_id=validated_sample_id,
        sample_name=resolved_sample_name,
        analyst=resolved_analyst,
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_size_bytes=file_size,
        dilution_factor=dilution_factor,
        inoculation_volume_ml=inoculation_volume_ml,
        notes=notes,
        status="processing",
    )
    db.add(db_analysis)
    await db.flush()

    try:
        result: AnalysisResult = await pipeline.analyze(
            str(file_path),
            params,
            analysis_id=analysis_id,
        )
        await _save_analysis_result(db, db_analysis, result, params)
        await _save_colonies(db, result)

        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": result.model_dump(),
        }

    except Exception as e:
        logger.exception(f"Analysis failed for {analysis_id}: {e}")
        db_analysis.status = "failed"
        db_analysis.error_message = str(e)
        await db.flush()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Development / Bulk Analysis (Validasi Farmasi)
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_TYPES = {
    "positive_product":  {"label": "Kontrol Positif Produk",  "required": 3},
    "negative_product":  {"label": "Kontrol Negatif Produk",  "required": 2},
    "negative_media":    {"label": "Kontrol Negatif Media",   "required": 2},
    "positive_media":    {"label": "Kontrol Positif Media",   "required": 2},
}

_SESSION_TAG = "dev_session:"
_KUALI_TAG   = "kuali_session:"

# field name → control group key
FIELD_MAP = {
    "positive_product_1": "positive_product",
    "positive_product_2": "positive_product",
    "positive_product_3": "positive_product",
    "negative_product_1": "negative_product",
    "negative_product_2": "negative_product",
    "negative_media_1":   "negative_media",
    "negative_media_2":   "negative_media",
    "positive_media_1":   "positive_media",
    "positive_media_2":   "positive_media",
}


async def _process_bulk_images(
    db: AsyncSession,
    sample_id: str,
    files_by_field: dict[str, "UploadFile"],
    params_base: "AnalysisParams",
    analysis_id_prefix: str = "",
) -> list[db_models.Analysis]:
    """
    Proses semua file upload, simpan Analysis ke DB, kembalikan list Analysis.
    """
    results = []
    upload_dir = Path(settings.upload_dir) / "originals"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for field_name, file in files_by_field.items():
        ctrl_key = FIELD_MAP.get(field_name, "positive_product")
        analysis_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        stored_filename = f"{analysis_id}{ext}"
        file_path = upload_dir / stored_filename

        content = await file.read()
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        params = AnalysisParams(
            sample_id=sample_id,
            sample_name=params_base.sample_name,
            analyst=params_base.analyst,
            dilution_factor=params_base.dilution_factor,
            inoculation_volume_ml=params_base.inoculation_volume_ml,
            media_type=params_base.media_type,
            organism_type=params_base.organism_type,
            incubation_date=params_base.incubation_date,
            observation_date=params_base.observation_date,
            notes=params_base.notes,
        )

        db_analysis = db_models.Analysis(
            id=analysis_id,
            sample_id=sample_id,
            sample_name=params_base.sample_name,
            analyst=params_base.analyst,
            original_filename=file.filename or stored_filename,
            stored_filename=stored_filename,
            file_size_bytes=len(content),
            dilution_factor=params_base.dilution_factor,
            inoculation_volume_ml=params_base.inoculation_volume_ml,
            notes=f"control_group:{ctrl_key}",
            status="processing",
        )
        db.add(db_analysis)
        await db.flush()

        try:
            result = await pipeline.analyze(str(file_path), params, analysis_id=analysis_id)
            await _save_analysis_result(db, db_analysis, result, params)
            await _save_colonies(db, result)
            # Simpan control_group ke morphology_summary
            morph = db_analysis.morphology_summary or {}
            morph["control_group"] = ctrl_key
            db_analysis.morphology_summary = morph
            await db.flush()
        except Exception as e:
            logger.exception(f"Bulk analysis failed for {analysis_id}: {e}")
            db_analysis.status = "failed"
            db_analysis.error_message = str(e)
            await db.flush()

        results.append(db_analysis)

    return results


def _compute_quanti_stats(analyses: list[db_models.Analysis]) -> dict:
    """Hitung statistik kuantitatif (mean, RSD, % recovery) dari list analysis."""
    all_cfus: dict[str, list] = {k: [] for k in CONTROL_TYPES}
    analysis_ids: dict[str, list] = {k: [] for k in CONTROL_TYPES}

    for a in analyses:
        morph = a.morphology_summary or {}
        ctrl_key = morph.get("control_group", "positive_product")
        if a.cfu_per_ml and ctrl_key in all_cfus:
            all_cfus[ctrl_key].append(a.cfu_per_ml)
        if ctrl_key in analysis_ids:
            analysis_ids[ctrl_key].append(a.id)

    stats: dict = {}
    for ctrl_key, vals in all_cfus.items():
        if vals:
            mean_val = statistics.mean(vals)
            rsd_val = (statistics.stdev(vals) / mean_val * 100) if len(vals) > 1 and mean_val > 0 else 0.0
            stats[ctrl_key] = {"n": len(vals), "mean": mean_val, "rsd": rsd_val, "values": vals}
        else:
            stats[ctrl_key] = {"n": 0, "mean": None, "rsd": None, "values": []}

    pos_mean = stats.get("positive_product", {}).get("mean")
    neg_mean = stats.get("negative_product", {}).get("mean")
    recovery = None
    rec_pass = False
    if pos_mean and neg_mean and neg_mean > 0:
        recovery = (pos_mean / neg_mean) * 100
        rec_pass = 70 <= recovery <= 130
    elif pos_mean:
        recovery = 100.0
        rec_pass = True

    stats["percent_recovery"] = recovery
    stats["recovery_pass"] = rec_pass

    issues = []
    if recovery is not None:
        icon = "✓" if rec_pass else "✕"
        issues.append(f"{icon} % Recovery: {recovery:.2f}% ({'Memenuhi' if rec_pass else 'Tidak Memenuhi'} syarat 70-130%)")
    for ck, st in {k: v for k, v in stats.items() if isinstance(v, dict)}.items():
        lbl = CONTROL_TYPES.get(ck, {}).get("label", ck)
        if st["rsd"] is not None:
            rsd_ok = st["rsd"] <= 15
            icon = "✓" if rsd_ok else "⚠"
            issues.append(f"{icon} {lbl} RSD: {st['rsd']:.2f}% ({'≤15% OK' if rsd_ok else '>15% Periksa Ulang'})")

    interp = {
        "passed": rec_pass,
        "summary": "✓ Validasi Kuantitatif Lulus" if rec_pass else "✕ Validasi Kuantitatif Tidak Lulus",
        "issues": issues,
    }
    all_ids = [a.id for a in analyses]
    return {"statistics": stats, "interpretation": interp,
            "analysis_ids": analysis_ids, "all_analysis_ids": all_ids}


def _compute_kuali_stats(analyses: list[db_models.Analysis]) -> dict:
    """Hitung hasil kualitatif dari list analysis."""
    exp_map = {
        "positive_product": True, "negative_product": False,
        "negative_media": False, "positive_media": True,
    }
    groups: dict[str, dict] = {}
    analysis_ids: dict[str, list] = {k: [] for k in CONTROL_TYPES}

    for a in analyses:
        morph = a.morphology_summary or {}
        ctrl_key = morph.get("control_group", "positive_product")
        if ctrl_key not in groups:
            groups[ctrl_key] = {"n_positif": 0, "n_negatif": 0, "total_colonies_group": 0, "results": []}
        # Kualitatif: >0 koloni = Positif
        is_positive = a.total_colonies > 0
        result_str = "Positif" if is_positive else "Negatif"
        groups[ctrl_key]["n_positif" if is_positive else "n_negatif"] += 1
        groups[ctrl_key]["total_colonies_group"] += a.total_colonies
        groups[ctrl_key]["results"].append({
            "foto": len(groups[ctrl_key]["results"]) + 1,
            "filename": a.original_filename,
            "result": result_str,
            "colony_count": a.total_colonies,
            "confidence": "—",
        })
        if ctrl_key in analysis_ids:
            analysis_ids[ctrl_key].append(a.id)

    for ck, g in groups.items():
        total_n = g["n_positif"] + g["n_negatif"]
        if total_n == 0:
            g["mean_result"] = "—"
        elif g["n_positif"] == total_n:
            g["mean_result"] = "Positif"
        elif g["n_negatif"] == total_n:
            g["mean_result"] = "Negatif"
        else:
            g["mean_result"] = "Mixed"

    all_pass = all(
        (exp_map.get(ck, True) and g["mean_result"] == "Positif") or
        (not exp_map.get(ck, True) and g["mean_result"] == "Negatif")
        for ck, g in groups.items()
    ) if groups else False

    issues = []
    for ck, g in groups.items():
        exp = "Positif" if exp_map.get(ck) else "Negatif"
        lbl = CONTROL_TYPES.get(ck, {}).get("label", ck)
        ok = g["mean_result"] == exp
        icon = "✓" if ok else ("⚠" if g["mean_result"] == "Mixed" else "✕")
        issues.append(f"{icon} {lbl}: {g['mean_result']} (Ekspektasi: {exp})")

    interp = {
        "passed": all_pass,
        "summary": "✓ Semua Kontrol Memenuhi Syarat" if all_pass else "✕ Terdapat Kontrol yang Tidak Memenuhi Syarat",
        "issues": issues,
    }
    all_ids = [a.id for a in analyses]
    return {"groups": groups, "interpretation": interp,
            "analysis_ids": analysis_ids, "all_analysis_ids": all_ids}


# ─────────────────────────────────────────────────────────────────────────────
# Development — Quantitative Sessions  (POST /api/development/sessions)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/development/sessions", status_code=status.HTTP_201_CREATED)
async def create_dev_session(
    sample_name: str = Form(...),
    analyst: str = Form(...),
    dilution_factor: float = Form(1.0),
    inoculation_volume_ml: float = Form(1.0),
    media_type: Optional[str] = Form(None),
    organism_type: Optional[str] = Form(None),
    incubation_date: Optional[str] = Form(None),
    observation_date: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    # 9 foto — 3+2+2+2
    positive_product_1: Optional[UploadFile] = File(None),
    positive_product_2: Optional[UploadFile] = File(None),
    positive_product_3: Optional[UploadFile] = File(None),
    negative_product_1: Optional[UploadFile] = File(None),
    negative_product_2: Optional[UploadFile] = File(None),
    negative_media_1:   Optional[UploadFile] = File(None),
    negative_media_2:   Optional[UploadFile] = File(None),
    positive_media_1:   Optional[UploadFile] = File(None),
    positive_media_2:   Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Buat sesi kuantitatif + proses semua foto sekaligus."""
    session_id = str(uuid.uuid4())
    session_tag = f"{_SESSION_TAG}{session_id[:8]}"
    full_notes = f"{session_tag} {notes}".strip() if notes else session_tag

    new_sample = db_models.Sample(
        id=str(uuid.uuid4()),
        sample_name=sample_name, analyst=analyst,
        media_type=media_type, organism_type=organism_type,
        incubation_date=incubation_date, observation_date=observation_date,
        notes=full_notes,
    )
    db.add(new_sample)
    await db.flush()

    files_by_field = {
        k: v for k, v in {
            "positive_product_1": positive_product_1,
            "positive_product_2": positive_product_2,
            "positive_product_3": positive_product_3,
            "negative_product_1": negative_product_1,
            "negative_product_2": negative_product_2,
            "negative_media_1":   negative_media_1,
            "negative_media_2":   negative_media_2,
            "positive_media_1":   positive_media_1,
            "positive_media_2":   positive_media_2,
        }.items() if v is not None
    }

    params_base = AnalysisParams(
        sample_id=new_sample.id, sample_name=sample_name, analyst=analyst,
        dilution_factor=dilution_factor, inoculation_volume_ml=inoculation_volume_ml,
        media_type=media_type, organism_type=organism_type,
        incubation_date=incubation_date, observation_date=observation_date, notes=notes,
    )

    analyses = await _process_bulk_images(db, new_sample.id, files_by_field, params_base)
    computed = _compute_quanti_stats(analyses)

    return {
        "session_id": session_id,
        "session_type": "quantitative",
        "sample_id": new_sample.id,
        "sample_name": sample_name,
        "analyst": analyst,
        "statistics": computed["statistics"],
        "interpretation": computed["interpretation"],
        "analysis_ids": computed["analysis_ids"],
        "all_analysis_ids": computed["all_analysis_ids"],
        "created_at": new_sample.created_at.isoformat() if new_sample.created_at else None,
    }


@app.get("/api/development/sessions")
async def list_dev_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List semua sesi kuantitatif."""
    offset = (page - 1) * per_page
    base_filter = db_models.Sample.notes.contains(_SESSION_TAG)
    total = (await db.execute(
        select(func.count()).select_from(db_models.Sample).where(base_filter)
    )).scalar() or 0
    result = await db.execute(
        select(db_models.Sample).where(base_filter)
        .order_by(desc(db_models.Sample.created_at))
        .offset(offset).limit(per_page)
    )
    samples = result.scalars().all()
    items = []
    for s in samples:
        n_analyses = (await db.execute(
            select(func.count()).select_from(db_models.Analysis)
            .where(db_models.Analysis.sample_id == s.id)
        )).scalar() or 0
        items.append({
            "id": s.id, "sample_name": s.sample_name, "analyst": s.analyst,
            "media_type": s.media_type, "organism_type": s.organism_type,
            "incubation_date": s.incubation_date, "observation_date": s.observation_date,
            "n_analyses": n_analyses,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })
    return PaginatedResponse(
        items=items, total=total, page=page, per_page=per_page,
        pages=(total + per_page - 1) // per_page,
    )


@app.get("/api/development/sessions/{sample_id}/result")
async def get_dev_session_result(sample_id: str, db: AsyncSession = Depends(get_db)):
    """Ambil hasil lengkap sesi kuantitatif."""
    result = await db.execute(
        select(db_models.Sample).where(db_models.Sample.id == sample_id)
    )
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")

    analyses_result = await db.execute(
        select(db_models.Analysis)
        .where(db_models.Analysis.sample_id == sample_id)
        .order_by(db_models.Analysis.created_at)
    )
    analyses = analyses_result.scalars().all()
    computed = _compute_quanti_stats(analyses)

    return {
        "session_id": sample_id,
        "session_type": "quantitative",
        "sample_id": sample.id,
        "sample_name": sample.sample_name,
        "analyst": sample.analyst,
        "statistics": computed["statistics"],
        "interpretation": computed["interpretation"],
        "analysis_ids": computed["analysis_ids"],
        "all_analysis_ids": computed["all_analysis_ids"],
        "created_at": sample.created_at.isoformat() if sample.created_at else None,
    }


@app.delete("/api/development/sessions/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dev_session(sample_id: str, db: AsyncSession = Depends(get_db)):
    """Hapus sesi kuantitatif dan semua analisis terkait."""
    result = await db.execute(
        select(db_models.Sample).where(db_models.Sample.id == sample_id)
    )
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")

    analyses_result = await db.execute(
        select(db_models.Analysis.id).where(db_models.Analysis.sample_id == sample_id)
    )
    analysis_ids = [row[0] for row in analyses_result.fetchall()]
    if analysis_ids:
        await db.execute(sql_delete(db_models.Report).where(
            db_models.Report.analysis_id.in_(analysis_ids)))
        await db.flush()
        await db.execute(sql_delete(db_models.Analysis).where(
            db_models.Analysis.sample_id == sample_id))
        await db.flush()
    await db.delete(sample)
    await db.flush()


@app.post("/api/development/sessions/{sample_id}/report")
async def generate_dev_session_report_endpoint(
    sample_id: str,
    formats: list[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate laporan sesi kuantitatif (PDF/CSV/JSON)."""
    if formats is None:
        formats = ["pdf"]

    result = await db.execute(select(db_models.Sample).where(db_models.Sample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")

    analyses_result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.sample_id == sample_id)
        .order_by(db_models.Analysis.created_at)
    )
    analyses = analyses_result.scalars().all()
    computed = _compute_quanti_stats(analyses)

    session_data = {
        "session_id": sample_id,
        "sample_id": sample.id,
        "sample_name": sample.sample_name,
        "analyst": sample.analyst,
        "media_type": sample.media_type,
        "organism_type": sample.organism_type,
        "incubation_date": sample.incubation_date,
        "observation_date": sample.observation_date,
        "notes": sample.notes,
        "created_at": sample.created_at.isoformat() if sample.created_at else None,
        "statistics": computed["statistics"],
        "interpretation": computed["interpretation"],
    }

    report_paths = {}
    errors = []

    if "pdf" in formats:
        try:
            path = pdf_generator.generate_dev_session_report(session_data, "quantitative")
            report_paths["pdf"] = path
        except Exception as e:
            logger.exception(f"PDF session report failed: {e}")
            errors.append(f"PDF: {e}")

    if "csv" in formats:
        try:
            path = csv_generator.generate_dev_session_report(session_data, "quantitative")                 if hasattr(csv_generator, "generate_dev_session_report")                 else csv_generator.generate(
                    _reconstruct_analysis_result(analyses[0], []) if analyses else None,
                    sample_name=sample.sample_name, analyst=sample.analyst or "")
            report_paths["csv"] = path
        except Exception as e:
            errors.append(f"CSV: {e}")

    if "json" in formats:
        try:
            import json as _json
            report_dir = Path("reports/json")
            report_dir.mkdir(parents=True, exist_ok=True)
            json_path = str(report_dir / f"session_{sample_id[:8]}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                _json.dump(session_data, jf, ensure_ascii=False, indent=2, default=str)
            report_paths["json"] = json_path
        except Exception as e:
            errors.append(f"JSON: {e}")

    download_urls = {
        fmt: f"/api/development/sessions/{sample_id}/report/download/{fmt}"
        for fmt in report_paths
    }
    return {"sample_id": sample_id, "generated": list(report_paths.keys()),
            "download_urls": download_urls, "errors": errors}


@app.get("/api/development/sessions/{sample_id}/report/download/{fmt}")
async def download_dev_session_report(sample_id: str, fmt: str):
    """Download laporan sesi kuantitatif."""
    if fmt not in ("pdf", "csv", "json"):
        raise HTTPException(status_code=400, detail="Format harus pdf, csv, atau json")
    ext_map = {"pdf": "pdf", "csv": "csv", "json": "json"}
    search_dir = Path(f"reports/{fmt}")
    matches = list(search_dir.glob(f"*session*{sample_id[:8]}*.{ext_map[fmt]}")) +               list(search_dir.glob(f"*{sample_id[:8]}*.{ext_map[fmt]}"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Laporan {fmt.upper()} tidak ditemukan. Generate dulu.")
    latest = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    media_map = {"pdf": "application/pdf", "csv": "text/csv; charset=utf-8", "json": "application/json"}
    return FileResponse(path=str(latest), media_type=media_map[fmt], filename=latest.name,
                        headers={"Content-Disposition": f'attachment; filename="{latest.name}"'})


# ─────────────────────────────────────────────────────────────────────────────
# Development — Qualitative Sessions  (POST /api/development/qualitative-sessions)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/development/qualitative-sessions", status_code=status.HTTP_201_CREATED)
async def create_kuali_session(
    sample_name: str = Form(...),
    analyst: str = Form(...),
    media_type: Optional[str] = Form(None),
    organism_type: Optional[str] = Form(None),
    incubation_date: Optional[str] = Form(None),
    observation_date: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    positive_product_1: Optional[UploadFile] = File(None),
    positive_product_2: Optional[UploadFile] = File(None),
    positive_product_3: Optional[UploadFile] = File(None),
    negative_product_1: Optional[UploadFile] = File(None),
    negative_product_2: Optional[UploadFile] = File(None),
    negative_media_1:   Optional[UploadFile] = File(None),
    negative_media_2:   Optional[UploadFile] = File(None),
    positive_media_1:   Optional[UploadFile] = File(None),
    positive_media_2:   Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Buat sesi kualitatif + proses semua foto sekaligus."""
    session_id = str(uuid.uuid4())
    session_tag = f"{_KUALI_TAG}{session_id[:8]}"
    full_notes = f"{session_tag} {notes}".strip() if notes else session_tag

    new_sample = db_models.Sample(
        id=str(uuid.uuid4()),
        sample_name=sample_name, analyst=analyst,
        media_type=media_type, organism_type=organism_type,
        incubation_date=incubation_date, observation_date=observation_date,
        notes=full_notes,
    )
    db.add(new_sample)
    await db.flush()

    files_by_field = {
        k: v for k, v in {
            "positive_product_1": positive_product_1,
            "positive_product_2": positive_product_2,
            "positive_product_3": positive_product_3,
            "negative_product_1": negative_product_1,
            "negative_product_2": negative_product_2,
            "negative_media_1":   negative_media_1,
            "negative_media_2":   negative_media_2,
            "positive_media_1":   positive_media_1,
            "positive_media_2":   positive_media_2,
        }.items() if v is not None
    }

    params_base = AnalysisParams(
        sample_id=new_sample.id, sample_name=sample_name, analyst=analyst,
        dilution_factor=1.0, inoculation_volume_ml=1.0,
        media_type=media_type, organism_type=organism_type,
        incubation_date=incubation_date, observation_date=observation_date, notes=notes,
    )

    analyses = await _process_bulk_images(db, new_sample.id, files_by_field, params_base)
    computed = _compute_kuali_stats(analyses)

    return {
        "session_id": session_id,
        "session_type": "qualitative",
        "sample_id": new_sample.id,
        "sample_name": sample_name,
        "analyst": analyst,
        "groups": computed["groups"],
        "interpretation": computed["interpretation"],
        "analysis_ids": computed["analysis_ids"],
        "all_analysis_ids": computed["all_analysis_ids"],
        "created_at": new_sample.created_at.isoformat() if new_sample.created_at else None,
    }


@app.get("/api/development/qualitative-sessions")
async def list_kuali_sessions(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List semua sesi kualitatif."""
    offset = (page - 1) * per_page
    base_filter = db_models.Sample.notes.contains(_KUALI_TAG)
    total = (await db.execute(
        select(func.count()).select_from(db_models.Sample).where(base_filter)
    )).scalar() or 0
    result = await db.execute(
        select(db_models.Sample).where(base_filter)
        .order_by(desc(db_models.Sample.created_at))
        .offset(offset).limit(per_page)
    )
    samples = result.scalars().all()
    items = []
    for s in samples:
        n_analyses = (await db.execute(
            select(func.count()).select_from(db_models.Analysis)
            .where(db_models.Analysis.sample_id == s.id)
        )).scalar() or 0
        items.append({
            "id": s.id, "sample_name": s.sample_name, "analyst": s.analyst,
            "media_type": s.media_type, "organism_type": s.organism_type,
            "n_analyses": n_analyses,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })
    return PaginatedResponse(
        items=items, total=total, page=page, per_page=per_page,
        pages=(total + per_page - 1) // per_page,
    )


@app.get("/api/development/qualitative-sessions/{sample_id}/result")
async def get_kuali_session_result(sample_id: str, db: AsyncSession = Depends(get_db)):
    """Ambil hasil lengkap sesi kualitatif."""
    result = await db.execute(select(db_models.Sample).where(db_models.Sample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")
    analyses_result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.sample_id == sample_id)
        .order_by(db_models.Analysis.created_at)
    )
    analyses = analyses_result.scalars().all()
    computed = _compute_kuali_stats(analyses)
    return {
        "session_id": sample_id,
        "session_type": "qualitative",
        "sample_id": sample.id,
        "sample_name": sample.sample_name,
        "analyst": sample.analyst,
        "groups": computed["groups"],
        "interpretation": computed["interpretation"],
        "analysis_ids": computed["analysis_ids"],
        "all_analysis_ids": computed["all_analysis_ids"],
        "created_at": sample.created_at.isoformat() if sample.created_at else None,
    }


@app.delete("/api/development/qualitative-sessions/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kuali_session(sample_id: str, db: AsyncSession = Depends(get_db)):
    """Hapus sesi kualitatif dan semua analisis terkait."""
    result = await db.execute(select(db_models.Sample).where(db_models.Sample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")
    analyses_result = await db.execute(
        select(db_models.Analysis.id).where(db_models.Analysis.sample_id == sample_id)
    )
    analysis_ids = [row[0] for row in analyses_result.fetchall()]
    if analysis_ids:
        await db.execute(sql_delete(db_models.Report).where(
            db_models.Report.analysis_id.in_(analysis_ids)))
        await db.flush()
        await db.execute(sql_delete(db_models.Analysis).where(
            db_models.Analysis.sample_id == sample_id))
        await db.flush()
    await db.delete(sample)
    await db.flush()


@app.post("/api/development/qualitative-sessions/{sample_id}/report")
async def generate_kuali_session_report_endpoint(
    sample_id: str,
    formats: list[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Generate laporan sesi kualitatif (PDF/CSV/JSON)."""
    if formats is None:
        formats = ["pdf"]

    result = await db.execute(select(db_models.Sample).where(db_models.Sample.id == sample_id))
    sample = result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail="Session not found")

    analyses_result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.sample_id == sample_id)
        .order_by(db_models.Analysis.created_at)
    )
    analyses = analyses_result.scalars().all()
    computed = _compute_kuali_stats(analyses)

    session_data = {
        "session_id": sample_id,
        "sample_id": sample.id,
        "sample_name": sample.sample_name,
        "analyst": sample.analyst,
        "media_type": sample.media_type,
        "organism_type": sample.organism_type,
        "incubation_date": sample.incubation_date,
        "observation_date": sample.observation_date,
        "notes": sample.notes,
        "created_at": sample.created_at.isoformat() if sample.created_at else None,
        "groups": computed["groups"],
        "interpretation": computed["interpretation"],
    }

    report_paths = {}
    errors = []

    if "pdf" in formats:
        try:
            path = pdf_generator.generate_dev_session_report(session_data, "qualitative")
            report_paths["pdf"] = path
        except Exception as e:
            logger.exception(f"PDF kuali report failed: {e}")
            errors.append(f"PDF: {e}")

    if "json" in formats:
        try:
            import json as _json
            report_dir = Path("reports/json")
            report_dir.mkdir(parents=True, exist_ok=True)
            json_path = str(report_dir / f"kuali_session_{sample_id[:8]}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                _json.dump(session_data, jf, ensure_ascii=False, indent=2, default=str)
            report_paths["json"] = json_path
        except Exception as e:
            errors.append(f"JSON: {e}")

    if "csv" in formats:
        try:
            import csv as _csv
            report_dir = Path("reports/csv")
            report_dir.mkdir(parents=True, exist_ok=True)
            csv_path = str(report_dir / f"kuali_session_{sample_id[:8]}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = _csv.writer(cf)
                writer.writerow(["Kelompok Kontrol", "Hasil", "Jumlah Koloni", "Filename"])
                for ck, g in computed["groups"].items():
                    lbl = CONTROL_TYPES.get(ck, {}).get("label", ck)
                    for r in g.get("results", []):
                        writer.writerow([lbl, r["result"], r["colony_count"], r["filename"]])
            report_paths["csv"] = csv_path
        except Exception as e:
            errors.append(f"CSV: {e}")

    download_urls = {
        fmt: f"/api/development/qualitative-sessions/{sample_id}/report/download/{fmt}"
        for fmt in report_paths
    }
    return {"sample_id": sample_id, "generated": list(report_paths.keys()),
            "download_urls": download_urls, "errors": errors}


@app.get("/api/development/qualitative-sessions/{sample_id}/report/download/{fmt}")
async def download_kuali_session_report(sample_id: str, fmt: str):
    """Download laporan sesi kualitatif."""
    if fmt not in ("pdf", "csv", "json"):
        raise HTTPException(status_code=400, detail="Format harus pdf, csv, atau json")
    search_dir = Path(f"reports/{fmt}")
    matches = (
    list(search_dir.glob(f"*kuali*{sample_id[:8]}*.{fmt}")) +
    list(search_dir.glob(f"*quali*{sample_id[:8]}*.{fmt}")) +  # ← tambahan ini
    list(search_dir.glob(f"*dev*{sample_id[:8]}*.{fmt}")) +    # ← dan ini
    list(search_dir.glob(f"*{sample_id[:8]}*.{fmt}"))
)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Laporan {fmt.upper()} tidak ditemukan. Generate dulu.")
    latest = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    media_map = {"pdf": "application/pdf", "csv": "text/csv; charset=utf-8", "json": "application/json"}
    return FileResponse(path=str(latest), media_type=media_map[fmt], filename=latest.name,
                        headers={"Content-Disposition": f'attachment; filename="{latest.name}"'})


# ─────────────────────────────────────────────────────────────────────────────
# Analyses — List / Get / Delete
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/analyses", response_model=PaginatedResponse)
async def list_analyses(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sample_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * per_page
    query = select(db_models.Analysis).order_by(desc(db_models.Analysis.created_at))
    if sample_id:
        query = query.where(db_models.Analysis.sample_id == sample_id)
    count_q = select(func.count()).select_from(db_models.Analysis)
    if sample_id:
        count_q = count_q.where(db_models.Analysis.sample_id == sample_id)
    total = (await db.execute(count_q)).scalar() or 0
    result = await db.execute(query.offset(offset).limit(per_page))
    analyses = result.scalars().all()
    return PaginatedResponse(
        items=[AnalysisResponse.model_validate(a) for a in analyses],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page,
    )


@app.get("/api/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    colonies_result = await db.execute(
        select(db_models.Colony)
        .where(db_models.Colony.analysis_id == analysis_id)
        .order_by(db_models.Colony.colony_index)
    )
    colonies = colonies_result.scalars().all()
    return {
        "analysis": AnalysisResponse.model_validate(analysis),
        "morphology_summary": analysis.morphology_summary,
        "qc_issues": analysis.qc_issues,
        "colony_count": len(colonies),
        "colonies": [
            {
                "index": c.colony_index + 1,
                "centroid": {"x": c.centroid_x, "y": c.centroid_y},
                "quadrant": c.quadrant,
                "confidence": c.confidence,
                "size_category": c.size_category,
                "diameter_mm": c.diameter_mm,
                "circularity": c.circularity,
                "was_separated": c.was_separated,
            }
            for c in colonies
        ],
    }


@app.delete("/api/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """
    FIX: Hapus Report terlebih dahulu sebelum hapus Analysis.
    Tanpa ini, SQLAlchemy mencoba SET analysis_id=NULL pada Report saat flush,
    yang melanggar NOT NULL constraint.
    """
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # FIX: Hapus Report dulu
    await db.execute(
        sql_delete(db_models.Report).where(
            db_models.Report.analysis_id == analysis_id
        )
    )
    await db.flush()

    # Hapus Analysis (Colony ikut terhapus via cascade="all, delete-orphan")
    await db.delete(analysis)
    await db.flush()


# FIX: Endpoint tambahan untuk update notes analisis
@app.patch("/api/analyses/{analysis_id}/notes", status_code=status.HTTP_200_OK)
async def update_analysis_notes(
    analysis_id: str,
    notes: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    analysis.notes = notes
    await db.flush()
    return {"analysis_id": analysis_id, "notes": notes, "updated": True}


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/reports/generate")
async def generate_reports(
    report_req: ReportRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate PDF, CSV, dan/atau JSON report untuk satu analisis.
    FIX: Metadata sampel (media_type, dates, organism, notes) sekarang
    diteruskan ke semua generator.
    """
    result = await db.execute(
        select(db_models.Analysis).where(db_models.Analysis.id == report_req.analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    colonies_result = await db.execute(
        select(db_models.Colony)
        .where(db_models.Colony.analysis_id == report_req.analysis_id)
        .order_by(db_models.Colony.colony_index)
    )
    colonies = colonies_result.scalars().all()

    sample = None
    if analysis.sample_id:
        sample_result = await db.execute(
            select(db_models.Sample).where(db_models.Sample.id == analysis.sample_id)
        )
        sample = sample_result.scalar_one_or_none()

    analysis_result = _reconstruct_analysis_result(analysis, colonies)

    sample_name = (
        sample.sample_name if sample
        else getattr(analysis, "sample_name", None)
        or f"Sample-{report_req.analysis_id[:8]}"
    )
    analyst = (
        sample.analyst if sample
        else getattr(analysis, "analyst", None)
        or "Unknown"
    )

    # FIX: Kumpulkan metadata untuk diteruskan ke semua generator
    meta_kwargs = {
        "media_type": sample.media_type if sample else None,
        "incubation_date": sample.incubation_date if sample else None,
        "observation_date": sample.observation_date if sample else None,
        "organism_type": sample.organism_type if sample else None,
        "notes": (sample.notes if sample else None) or getattr(analysis, "notes", None),
    }

    report_paths = {}
    errors = []

    if "pdf" in report_req.formats:
        try:
            path = pdf_generator.generate(
                analysis_result,
                sample_name=sample_name,
                analyst=analyst,
                **meta_kwargs,
            )
            report_paths["pdf"] = path
        except Exception as e:
            logger.exception(f"PDF generation failed: {e}")
            errors.append(f"PDF generation failed: {e}")

    if "csv" in report_req.formats:
        try:
            # FIX: teruskan meta_kwargs ke CSV generator
            path = csv_generator.generate(
                analysis_result,
                sample_name=sample_name,
                analyst=analyst,
                **meta_kwargs,
            )
            report_paths["csv"] = path
        except Exception as e:
            logger.exception(f"CSV generation failed: {e}")
            errors.append(f"CSV generation failed: {e}")

    if "json" in report_req.formats:
        try:
            # FIX: teruskan meta_kwargs ke JSON generator
            path = json_generator.generate(
                analysis_result,
                sample_name=sample_name,
                analyst=analyst,
                **meta_kwargs,
            )
            report_paths["json"] = path
        except Exception as e:
            logger.exception(f"JSON generation failed: {e}")
            errors.append(f"JSON generation failed: {e}")

    # Upsert Report record
    existing_report = await db.execute(
        select(db_models.Report).where(db_models.Report.analysis_id == report_req.analysis_id)
    )
    db_report = existing_report.scalar_one_or_none()
    if db_report:
        if "pdf" in report_paths:
            db_report.pdf_path = report_paths["pdf"]
        if "csv" in report_paths:
            db_report.csv_path = report_paths["csv"]
        if "json" in report_paths:
            db_report.json_path = report_paths["json"]
    else:
        db_report = db_models.Report(
            id=str(uuid.uuid4()),
            analysis_id=report_req.analysis_id,
            pdf_path=report_paths.get("pdf"),
            csv_path=report_paths.get("csv"),
            json_path=report_paths.get("json"),
        )
        db.add(db_report)
    await db.flush()

    return {
        "analysis_id": report_req.analysis_id,
        "generated": list(report_paths.keys()),
        "download_urls": {
            fmt: f"/api/reports/{report_req.analysis_id}/download/{fmt}"
            for fmt in report_paths
        },
        "errors": errors,
    }


@app.post("/api/reports/generate-session")
async def generate_session_report(
    session_id: str = Form(...),
    session_type: str = Form("quantitative"),
    db: AsyncSession = Depends(get_db),
):
    """
    NEW: Generate laporan sesi pengembangan (kuantitatif atau kualitatif).
    Mengintegrasikan generate_dev_session_report dari report_generator_patch.py.
    session_type: 'quantitative' atau 'qualitative'
    """
    # Cari sample dengan session_id dalam notes
    sample_result = await db.execute(
        select(db_models.Sample).where(
            db_models.Sample.notes.contains(session_id[:8])
        )
    )
    sample = sample_result.scalar_one_or_none()
    if not sample:
        raise HTTPException(status_code=404, detail=f"Session '{session_id[:8]}' not found")

    # Cari semua analyses yang terkait sample ini
    analyses_result = await db.execute(
        select(db_models.Analysis)
        .where(db_models.Analysis.sample_id == sample.id)
        .order_by(db_models.Analysis.created_at)
    )
    analyses = analyses_result.scalars().all()

    is_quali = session_type == "qualitative"

    # Build session_data untuk generate_dev_session_report
    all_cfus: dict[str, list] = {
        "positive_product": [],
        "negative_product": [],
        "negative_media": [],
        "positive_media": [],
    }
    analyses_data = []

    for a in analyses:
        morph = a.morphology_summary or {}
        ctrl_key = morph.get("control_group", "positive_product")

        if is_quali:
            analyses_data.append({
                "analysis_id": a.id,
                "filename": a.original_filename,
                "control_group": ctrl_key,
                "qualitative_result": morph.get("qualitative_result", "—"),
                "colony_count": morph.get("qualitative_colony_count", a.total_colonies),
                "qc_grade": a.qc_grade,
                "confidence": morph.get("qualitative_confidence", "—"),
            })
        else:
            if a.cfu_per_ml and ctrl_key in all_cfus:
                all_cfus[ctrl_key].append(a.cfu_per_ml)
            analyses_data.append({
                "analysis_id": a.id,
                "filename": a.original_filename,
                "control_group": ctrl_key,
                "total_colonies": a.total_colonies,
                "cfu_per_ml": a.cfu_per_ml,
                "cfu_per_ml_formatted": a.cfu_per_ml_formatted,
                "qc_grade": a.qc_grade,
            })

    # Hitung statistik kuantitatif
    stats = {}
    if not is_quali:
        for ctrl_key, vals in all_cfus.items():
            if vals:
                mean_val = statistics.mean(vals)
                rsd_val = (statistics.stdev(vals) / mean_val * 100) if len(vals) > 1 and mean_val > 0 else 0.0
                stats[ctrl_key] = {"n": len(vals), "mean": mean_val, "rsd": rsd_val, "values": vals}
            else:
                stats[ctrl_key] = {"n": 0, "mean": None, "rsd": None, "values": []}

        pos_mean = stats.get("positive_product", {}).get("mean")
        neg_mean = stats.get("negative_product", {}).get("mean")
        recovery = None
        rec_pass = False
        if pos_mean and neg_mean and neg_mean > 0:
            recovery = (pos_mean / neg_mean) * 100
            rec_pass = 70 <= recovery <= 130
        elif pos_mean:
            recovery = 100.0
            rec_pass = True

        stats["percent_recovery"] = recovery
        stats["recovery_pass"] = rec_pass

    # Interpretasi
    if is_quali:
        # Kualitatif — evaluasi per grup
        groups: dict[str, dict] = {}
        exp_map = {
            "positive_product": True, "negative_product": False,
            "negative_media": False, "positive_media": True
        }
        for ad in analyses_data:
            ck = ad["control_group"]
            if ck not in groups:
                groups[ck] = {"n_positif": 0, "n_negatif": 0, "total_colonies_group": 0, "results": []}
            res = ad["qualitative_result"]
            if res == "Positif":
                groups[ck]["n_positif"] += 1
            else:
                groups[ck]["n_negatif"] += 1
            groups[ck]["total_colonies_group"] += ad["colony_count"]
            groups[ck]["results"].append({
                "foto": len(groups[ck]["results"]) + 1,
                "filename": ad["filename"],
                "result": res,
                "colony_count": ad["colony_count"],
                "confidence": ad["confidence"],
            })
        for ck, g in groups.items():
            total_n = g["n_positif"] + g["n_negatif"]
            if total_n == 0:
                g["mean_result"] = "—"
            elif g["n_positif"] == total_n:
                g["mean_result"] = "Positif"
            elif g["n_negatif"] == total_n:
                g["mean_result"] = "Negatif"
            else:
                g["mean_result"] = "Mixed"

        all_pass = all(
            (exp_map.get(ck, True) and g["mean_result"] == "Positif") or
            (not exp_map.get(ck, True) and g["mean_result"] == "Negatif")
            for ck, g in groups.items()
        )
        issues = []
        for ck, g in groups.items():
            exp = "Positif" if exp_map.get(ck) else "Negatif"
            lbl = CONTROL_TYPES.get(ck, {}).get("label", ck)
            ok = g["mean_result"] == exp
            icon = "✓" if ok else ("⚠" if g["mean_result"] == "Mixed" else "✕")
            issues.append(f"{icon} {lbl}: {g['mean_result']} (Ekspektasi: {exp})")

        interp = {
            "passed": all_pass,
            "summary": "✓ Semua Kontrol Memenuhi Syarat" if all_pass else "✕ Terdapat Kontrol yang Tidak Memenuhi Syarat",
            "issues": issues,
        }
        session_data = {
            "session_id": session_id,
            "sample_id": sample.id,
            "sample_name": sample.sample_name,
            "analyst": sample.analyst,
            "media_type": sample.media_type,
            "organism_type": sample.organism_type,
            "incubation_date": sample.incubation_date,
            "observation_date": sample.observation_date,
            "notes": sample.notes,
            "created_at": sample.created_at.isoformat() if sample.created_at else None,
            "groups": groups,
            "interpretation": interp,
            "analyses": analyses_data,
        }
    else:
        issues = []
        rec_pass = stats.get("recovery_pass", False)
        recovery = stats.get("percent_recovery")
        if recovery is not None:
            icon = "✓" if rec_pass else "✕"
            issues.append(f"{icon} % Recovery: {recovery:.2f}% ({'Memenuhi' if rec_pass else 'Tidak Memenuhi'} syarat 70-130%)")
        for ck, st in {k: v for k, v in stats.items() if isinstance(v, dict)}.items():
            lbl = CONTROL_TYPES.get(ck, {}).get("label", ck)
            if st["rsd"] is not None:
                rsd_ok = st["rsd"] <= 15
                icon = "✓" if rsd_ok else "⚠"
                issues.append(f"{icon} {lbl} RSD: {st['rsd']:.2f}% ({'≤15% OK' if rsd_ok else '>15% Periksa Ulang'})")

        interp = {
            "passed": rec_pass,
            "summary": "✓ Validasi Kuantitatif Lulus" if rec_pass else "✕ Validasi Kuantitatif Tidak Lulus",
            "issues": issues,
        }
        session_data = {
            "session_id": session_id,
            "sample_id": sample.id,
            "sample_name": sample.sample_name,
            "analyst": sample.analyst,
            "media_type": sample.media_type,
            "organism_type": sample.organism_type,
            "incubation_date": sample.incubation_date,
            "observation_date": sample.observation_date,
            "notes": sample.notes,
            "created_at": sample.created_at.isoformat() if sample.created_at else None,
            "statistics": stats,
            "interpretation": interp,
            "analyses": analyses_data,
        }

    try:
        pdf_path = pdf_generator.generate_dev_session_report(session_data, session_type)
        return {
            "session_id": session_id,
            "session_type": session_type,
            "pdf_path": pdf_path,
            "download_url": f"/api/reports/session/{session_id[:8]}/download",
            "interpretation": interp,
        }
    except AttributeError:
        # generate_dev_session_report belum ada di PDFReportGenerator
        # Tambahkan method dari report_generator_patch.py ke class PDFReportGenerator
        raise HTTPException(
            status_code=501,
            detail="Session report generator belum diintegrasikan. "
                   "Tambahkan method dari report_generator_patch.py ke class PDFReportGenerator."
        )
    except Exception as e:
        logger.exception(f"Session report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session report failed: {e}")


@app.get("/api/reports/session/{session_short_id}/download")
async def download_session_report(session_short_id: str):
    """Download laporan sesi pengembangan yang sudah digenerate."""
    pdf_dir = Path("reports/pdf")
    matches = list(pdf_dir.glob(f"*session_{session_short_id}*.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="Session report not found. Generate it first.")
    latest = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return FileResponse(
        path=str(latest),
        media_type="application/pdf",
        filename=latest.name,
        headers={"Content-Disposition": f'attachment; filename="{latest.name}"'},
    )


@app.get("/api/reports/{analysis_id}/download/{format}")
async def download_report(analysis_id: str, format: str, db: AsyncSession = Depends(get_db)):
    if format not in ("pdf", "csv", "json"):
        raise HTTPException(status_code=400, detail="Invalid format. Use: pdf, csv, json")

    result = await db.execute(
        select(db_models.Report).where(db_models.Report.analysis_id == analysis_id)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found. Generate it first via POST /api/reports/generate")

    path_map = {"pdf": report.pdf_path, "csv": report.csv_path, "json": report.json_path}
    file_path = path_map.get(format)
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"{format.upper()} report file not found on disk.")

    media_types = {
        "pdf": "application/pdf",
        "csv": "text/csv; charset=utf-8",
        "json": "application/json",
    }
    filename = f"microcount_{analysis_id[:8]}_{format}.{format}"
    return FileResponse(
        path=file_path,
        media_type=media_types[format],
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistics Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats/dashboard")
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    total_analyses = (await db.execute(select(func.count()).select_from(db_models.Analysis))).scalar() or 0
    total_samples = (await db.execute(select(func.count()).select_from(db_models.Sample))).scalar() or 0
    total_colonies = (await db.execute(select(func.sum(db_models.Analysis.total_colonies)))).scalar() or 0
    recent = await db.execute(
        select(db_models.Analysis)
        .order_by(desc(db_models.Analysis.created_at))
        .limit(5)
    )
    recent_analyses = recent.scalars().all()
    return {
        "total_analyses": total_analyses,
        "total_samples": total_samples,
        "total_colonies_counted": int(total_colonies),
        "recent_analyses": [
            {
                "id": a.id,
                "filename": a.original_filename,
                "total_colonies": a.total_colonies,
                "cfu_formatted": a.cfu_per_ml_formatted,
                "qc_grade": a.qc_grade,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in recent_analyses
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility / Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _validate_upload(file: UploadFile):
    """Validasi ukuran dan ekstensi file upload."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}",
        )
    # Cek ukuran file
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.max_upload_size_mb}MB",
        )
    # Reset pointer
    await file.seek(0)


async def _save_analysis_result(
    db: AsyncSession,
    db_analysis: db_models.Analysis,
    result: AnalysisResult,
    params: AnalysisParams,
):
    db_analysis.status = "completed"
    db_analysis.total_colonies = result.total_colonies
    db_analysis.colonies_q1 = result.colonies_per_quadrant.get("Q1", 0)
    db_analysis.colonies_q2 = result.colonies_per_quadrant.get("Q2", 0)
    db_analysis.colonies_q3 = result.colonies_per_quadrant.get("Q3", 0)
    db_analysis.colonies_q4 = result.colonies_per_quadrant.get("Q4", 0)
    db_analysis.colony_density = result.colony_density
    if result.petri_dish.detected:
        db_analysis.petri_detected = True
        db_analysis.petri_center_x = result.petri_dish.center_x
        db_analysis.petri_center_y = result.petri_dish.center_y
        db_analysis.petri_radius = result.petri_dish.radius
        db_analysis.petri_area_px = result.petri_dish.area_px
    db_analysis.cfu_per_ml = result.cfu.cfu_per_ml
    db_analysis.cfu_per_ml_formatted = result.cfu.cfu_per_ml_formatted
    qc = result.quality_control
    db_analysis.qc_score = qc.score
    db_analysis.qc_grade = qc.grade
    db_analysis.qc_focus_score = qc.focus_score
    db_analysis.qc_blur_score = qc.blur_score
    db_analysis.qc_glare_score = qc.glare_score
    db_analysis.qc_shadow_score = qc.shadow_score
    db_analysis.qc_visibility_score = qc.visibility_score
    db_analysis.qc_issues = qc.issues
    db_analysis.morphology_summary = result.morphology_summary.model_dump()
    db_analysis.annotated_image_path = result.annotated_image_path
    db_analysis.processing_time_ms = result.processing_time_ms
    db_analysis.model_version = "yolo11"
    await db.flush()


async def _save_colonies(db: AsyncSession, result: AnalysisResult):
    for colony in result.colonies:
        m = colony.morphology
        db_colony = db_models.Colony(
            id=str(uuid.uuid4()),
            analysis_id=result.analysis_id,
            colony_index=colony.colony_index,
            bbox_x1=colony.bbox[0],
            bbox_y1=colony.bbox[1],
            bbox_x2=colony.bbox[2],
            bbox_y2=colony.bbox[3],
            centroid_x=colony.centroid[0],
            centroid_y=colony.centroid[1],
            quadrant=colony.quadrant,
            confidence=colony.confidence,
            was_separated=colony.was_separated,
            diameter_px=m.diameter_px if m else None,
            diameter_mm=m.diameter_mm if m else None,
            area_px=m.area_px if m else None,
            circularity=m.circularity if m else None,
            mean_intensity=m.mean_intensity if m else None,
            color_r=m.color_rgb[0] if m and m.color_rgb else None,
            color_g=m.color_rgb[1] if m and m.color_rgb else None,
            color_b=m.color_rgb[2] if m and m.color_rgb else None,
            opacity=m.opacity if m else None,
            edge_smoothness=m.edge_smoothness if m else None,
            size_category=m.size_category if m else None,
        )
        db.add(db_colony)
    await db.flush()


def _reconstruct_analysis_result(
    analysis: db_models.Analysis,
    colonies: list[db_models.Colony],
) -> AnalysisResult:
    from app.models.schemas import (
        CFUResult, ColonyMorphology, ColonyResult,
        MorphologySummary, PetriDishInfo, QualityControlResult,
    )
    morph_dict = analysis.morphology_summary or {}
    return AnalysisResult(
        analysis_id=analysis.id,
        sample_id=analysis.sample_id,
        sample_name=getattr(analysis, "sample_name", None) or "Sample",
        analyst=getattr(analysis, "analyst", None) or "Analyst",
        processing_time_ms=analysis.processing_time_ms or 0,
        petri_dish=PetriDishInfo(
            detected=analysis.petri_detected,
            center_x=analysis.petri_center_x,
            center_y=analysis.petri_center_y,
            radius=analysis.petri_radius,
            area_px=analysis.petri_area_px,
        ),
        total_colonies=analysis.total_colonies,
        colonies_per_quadrant={
            "Q1": analysis.colonies_q1,
            "Q2": analysis.colonies_q2,
            "Q3": analysis.colonies_q3,
            "Q4": analysis.colonies_q4,
        },
        colony_density=analysis.colony_density or 0.0,
        cfu=CFUResult(
            total_colonies=analysis.total_colonies,
            dilution_factor=analysis.dilution_factor,
            inoculation_volume_ml=analysis.inoculation_volume_ml,
            cfu_per_ml=analysis.cfu_per_ml or 0.0,
            cfu_per_ml_formatted=analysis.cfu_per_ml_formatted or "0",
            scientific_notation=f"{analysis.cfu_per_ml:.2e}" if analysis.cfu_per_ml else "0",
        ),
        quality_control=QualityControlResult(
            score=analysis.qc_score or 0.0,
            grade=analysis.qc_grade or "Unknown",
            focus_score=analysis.qc_focus_score or 0.0,
            blur_score=analysis.qc_blur_score or 0.0,
            glare_score=analysis.qc_glare_score or 0.0,
            shadow_score=analysis.qc_shadow_score or 0.0,
            visibility_score=analysis.qc_visibility_score or 0.0,
            issues=analysis.qc_issues or [],
        ),
        morphology_summary=MorphologySummary(
            pinpoint_count=morph_dict.get("pinpoint_count", 0),
            small_count=morph_dict.get("small_count", 0),
            medium_count=morph_dict.get("medium_count", 0),
            large_count=morph_dict.get("large_count", 0),
            mean_diameter_mm=morph_dict.get("mean_diameter_mm"),
            mean_circularity=morph_dict.get("mean_circularity"),
            dominant_color=morph_dict.get("dominant_color"),
            size_distribution=morph_dict.get("size_distribution", {}),
        ),
        colonies=[
            ColonyResult(
                colony_index=c.colony_index,
                bbox=(c.bbox_x1, c.bbox_y1, c.bbox_x2, c.bbox_y2),
                centroid=(c.centroid_x, c.centroid_y),
                confidence=c.confidence,
                quadrant=c.quadrant,
                morphology=ColonyMorphology(
                    diameter_px=c.diameter_px,
                    diameter_mm=c.diameter_mm,
                    area_px=c.area_px,
                    circularity=c.circularity,
                    mean_intensity=c.mean_intensity,
                    color_rgb=(c.color_r, c.color_g, c.color_b) if c.color_r else None,
                    opacity=c.opacity,
                    edge_smoothness=c.edge_smoothness,
                    size_category=c.size_category or "medium",
                ),
                was_separated=c.was_separated,
            )
            for c in colonies
        ],
        annotated_image_path=analysis.annotated_image_path,
    )
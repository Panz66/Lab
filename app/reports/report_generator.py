"""
MicroCount - PDF Report Generator (FIXED + INTEGRATED)
Perubahan:
  1. generate_dev_session_report dari report_generator_patch.py
     diintegrasikan sebagai method resmi di class PDFReportGenerator.
  2. Menggunakan ReportLab (bukan WeasyPrint) agar konsisten dengan generate().
     Jika WeasyPrint tersedia, gunakan fallback HTML-to-PDF.
"""
import json
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.schemas import AnalysisResult
from app.core.config import get_settings

settings = get_settings()

# Brand colors
BRAND_GREEN = colors.HexColor("#2D7D46")
BRAND_DARK = colors.HexColor("#1A1A2E")
BRAND_LIGHT = colors.HexColor("#F0F7F2")
BRAND_ACCENT = colors.HexColor("#4CAF50")
BRAND_RED = colors.HexColor("#E53935")
BRAND_YELLOW = colors.HexColor("#FFA000")

QC_COLORS = {
    "Excellent": colors.HexColor("#2E7D32"),
    "Good": colors.HexColor("#388E3C"),
    "Fair": colors.HexColor("#F57C00"),
    "Poor": colors.HexColor("#C62828"),
}


class PDFReportGenerator:
    """Generates professional PDF laboratory reports."""

    def generate(
        self,
        result: AnalysisResult,
        sample_name: str,
        analyst: str,
        media_type: Optional[str] = None,
        incubation_date: Optional[str] = None,
        observation_date: Optional[str] = None,
        organism_type: Optional[str] = None,
        notes: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate PDF report untuk satu analisis, return file path."""

        if not output_path:
            output_dir = Path(settings.pdf_report_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{result.analysis_id}_report.pdf")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = self._build_styles()
        story = []

        story.extend(self._build_header(styles))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("MICROBIAL COLONY COUNT REPORT", styles["report_title"]))
        story.append(HRFlowable(width="100%", thickness=2, color=BRAND_GREEN))
        story.append(Spacer(1, 0.3 * cm))

        story.extend(
            self._build_sample_info(
                styles,
                result=result,
                sample_name=sample_name,
                analyst=analyst,
                media_type=media_type,
                incubation_date=incubation_date,
                observation_date=observation_date,
                organism_type=organism_type,
            )
        )
        story.append(Spacer(1, 0.5 * cm))
        story.extend(self._build_analysis_summary(styles, result))
        story.append(Spacer(1, 0.5 * cm))
        story.extend(self._build_cfu_section(styles, result))
        story.append(Spacer(1, 0.5 * cm))
        story.extend(self._build_qc_section(styles, result))
        story.append(Spacer(1, 0.5 * cm))
        story.extend(self._build_distribution_section(styles, result))
        story.append(Spacer(1, 0.5 * cm))
        story.extend(self._build_morphology_section(styles, result))
        story.append(Spacer(1, 0.5 * cm))

        if result.annotated_image_path and Path(result.annotated_image_path).exists():
            story.extend(self._build_image_section(styles, result.annotated_image_path))
            story.append(Spacer(1, 0.5 * cm))

        if notes:
            story.append(Paragraph("NOTES", styles["section_header"]))
            story.append(Paragraph(notes, styles["body"]))
            story.append(Spacer(1, 0.3 * cm))

        if result.warnings:
            story.extend(self._build_warnings(styles, result.warnings))
            story.append(Spacer(1, 0.3 * cm))

        story.extend(self._build_footer(styles, result))

        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────────────────
    # generate_dev_session_report — INTEGRATED dari report_generator_patch.py
    # ─────────────────────────────────────────────────────────────────────────

    def generate_dev_session_report(
        self,
        session_data: dict,
        session_type: str = "quantitative",
    ) -> str:
        """
        Generate satu PDF ringkasan untuk seluruh sesi pengembangan.
        Menggunakan WeasyPrint untuk HTML-to-PDF (layout lebih bebas).
        Fallback ke ReportLab jika WeasyPrint tidak tersedia.
        session_type: 'quantitative' atau 'qualitative'
        """
        try:
            from weasyprint import HTML as WeasyprintHTML
            return self._generate_dev_session_weasyprint(session_data, session_type)
        except ImportError:
            logger.warning("WeasyPrint tidak tersedia — fallback ke ReportLab")
            return self._generate_dev_session_reportlab(session_data, session_type)

    def _generate_dev_session_weasyprint(self, session_data: dict, session_type: str) -> str:
        """WeasyPrint HTML-to-PDF — versi lengkap dengan data analisis penuh."""
        from weasyprint import HTML

        is_quali = session_type == "qualitative"
        _sid_raw = session_data.get("session_id") or session_data.get("sample_id") or ""
        report_id = str(_sid_raw)[:8] if _sid_raw else str(_uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        sample_name = session_data.get("sample_name", "—")
        analyst     = session_data.get("analyst", "—")
        media_type  = session_data.get("media_type") or "—"
        organism    = session_data.get("organism_type") or "—"
        incubation  = session_data.get("incubation_date") or "—"
        observation = session_data.get("observation_date") or "—"
        notes       = session_data.get("notes") or ""
        created_at  = (session_data.get("created_at", timestamp)[:10]
                       if session_data.get("created_at") else timestamp)

        ctrl_order = [
            ("positive_product", "Kontrol Positif Produk", "#2D6A4F", True),
            ("negative_product", "Kontrol Negatif Produk", "#991B1B", False),
            ("negative_media",   "Kontrol Negatif Media",  "#92400E", False),
            ("positive_media",   "Kontrol Positif Media",  "#1E40AF", True),
        ]
        ctrl_labels_short = {
            "positive_product": "KP Produk",
            "negative_product": "KN Produk",
            "negative_media":   "KN Media",
            "positive_media":   "KP Media",
        }

        def _fmt_cfu(v):
            if v is None: return "—"
            try: return f"{float(v):.3e}"
            except: return str(v)

        def _fmt_rsd(v):
            if v is None: return "—"
            try: return f"{float(v):.2f}%"
            except: return str(v)

        # ══════════════════════════════════════════════════════════════════
        # KUANTITATIF — Statistik + Acuan Formula + Detail per Analisis
        # ══════════════════════════════════════════════════════════════════
        if not is_quali:
            stats  = session_data.get("statistics", {})
            interp = session_data.get("interpretation", {})
            recovery = stats.get("percent_recovery")
            rec_pass = stats.get("recovery_pass", False)
            analyses = session_data.get("analyses", [])

            # Ambil dilution_factor & volume dari analisis pertama (semua sama)
            first_a = analyses[0] if analyses else {}
            dilution = first_a.get("dilution_factor", 1)
            volume   = first_a.get("inoculation_volume_ml", 1.0)

            # ── Tabel statistik per kelompok ──
            stat_rows = ""
            for ctrl_key, ctrl_label, ctrl_color, _ in ctrl_order:
                st   = stats.get(ctrl_key, {})
                n    = st.get("n", 0)
                mean = _fmt_cfu(st.get("mean"))
                rsd  = _fmt_rsd(st.get("rsd"))
                rsd_val = st.get("rsd")
                rsd_ok  = (rsd_val is not None and rsd_val <= 15)
                rsd_color = "#2D6A4F" if rsd_ok else "#92400E"
                vals = "  |  ".join(_fmt_cfu(x) for x in (st.get("values") or [])) or "—"
                stat_rows += f"""
                <tr>
                  <td style="color:{ctrl_color};font-weight:600">{ctrl_label}</td>
                  <td style="text-align:center">{n}</td>
                  <td style="text-align:right;font-family:monospace;font-weight:600">{mean}</td>
                  <td style="text-align:right;color:{rsd_color};font-weight:600">{rsd}</td>
                  <td style="font-size:10px;color:#444;font-family:monospace">{vals}</td>
                </tr>"""

            # ── Recovery block ──
            rec_color  = "#2D6A4F" if rec_pass else "#991B1B"
            rec_label  = f"{recovery:.2f}%" if recovery is not None else "—"
            rec_status = ("✓ Memenuhi Syarat (70–130%)" if rec_pass
                          else ("✕ Di Luar Batas — Periksa Metode" if recovery is not None else "Tidak dapat dihitung"))

            # ── Acuan & Kriteria Kelulusan ──
            acuan_section = f"""
            <h2 class="section-title">Acuan Perhitungan &amp; Kriteria Penerimaan</h2>
            <div class="acuan-box">
              <div class="acuan-grid">
                <div class="acuan-item">
                  <div class="acuan-label">Rumus CFU/mL</div>
                  <div class="acuan-formula">CFU/mL = Jumlah Koloni × Faktor Pengenceran ÷ Volume Inokulasi (mL)</div>
                </div>
                <div class="acuan-item">
                  <div class="acuan-label">Parameter Sesi Ini</div>
                  <div style="font-size:11px;margin-top:4px">
                    Faktor Pengenceran: <strong>{dilution}×</strong>&nbsp;&nbsp;
                    Volume Inokulasi: <strong>{volume} mL</strong>
                  </div>
                </div>
              </div>
              <div style="height:1px;background:#e0e0e0;margin:10px 0"></div>
              <div class="acuan-criteria">
                <div class="criteria-item">
                  <span class="criteria-badge pass">% Recovery</span>
                  <span class="criteria-desc">70% – 130% &nbsp;(Acuan: USP &lt;1223&gt; / ISO 29981)</span>
                </div>
                <div class="criteria-item">
                  <span class="criteria-badge warn">RSD</span>
                  <span class="criteria-desc">≤ 15% per kelompok kontrol &nbsp;(Presisi antar-ulangan)</span>
                </div>
                <div class="criteria-item">
                  <span class="criteria-badge neutral">Rumus % Recovery</span>
                  <span class="criteria-desc">% Recovery = (Mean CFU KP Produk ÷ Mean CFU KN Produk) × 100</span>
                </div>
              </div>
            </div>"""

            # ── Issues interpretasi ──
            issues_html = "".join(
                f'<div class="issue-item" style="color:{"#2D6A4F" if i.startswith("✓") else "#991B1B" if i.startswith("✕") else "#92400E"}">{i}</div>'
                for i in (interp.get("issues") or [])
            )

            # ── Tabel detail per analisis individual ──
            detail_rows = ""
            for a in analyses:
                ctrl_lbl  = ctrl_labels_short.get(a.get("control_group", ""), a.get("control_group", "—"))
                col_count = a.get("total_colonies", 0)
                cfu_fmt   = a.get("cfu_per_ml_formatted") or _fmt_cfu(a.get("cfu_per_ml"))
                cfu_raw   = a.get("cfu_per_ml")
                qc        = a.get("qc_grade") or "—"
                fname     = a.get("filename", "—")
                aid       = a.get("analysis_id", "")[:8]
                dil_a     = a.get("dilution_factor", dilution)
                vol_a     = a.get("inoculation_volume_ml", volume)
                # Hitung manual untuk menunjukkan acuan
                formula_show = f"{col_count} × {dil_a} ÷ {vol_a}" if cfu_raw else "—"
                qc_color = "#2D6A4F" if qc in ("Excellent","Good") else "#92400E" if qc == "Fair" else "#991B1B"
                detail_rows += f"""
                <tr>
                  <td style="font-size:10px;color:#777;font-family:monospace">{aid}</td>
                  <td style="font-size:11px">{fname}</td>
                  <td style="font-weight:600;color:{ctrl_labels_short.get(a.get('control_group',''),'') and '#333'}">{ctrl_lbl}</td>
                  <td style="text-align:center;font-weight:600">{col_count}</td>
                  <td style="font-size:10px;color:#666;font-family:monospace">{formula_show}</td>
                  <td style="text-align:right;font-family:monospace;font-weight:600">{cfu_fmt}</td>
                  <td style="text-align:center;color:{qc_color};font-weight:600">{qc}</td>
                </tr>"""

            mode_label = "Kuantitatif (Hitung Koloni / CFU)"
            title_str  = "Laporan Validasi Sesi Pengembangan Kuantitatif"

            main_section = f"""
            {acuan_section}

            <h2 class="section-title">Statistik Hasil Analisis</h2>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Kelompok Kontrol</th><th style="text-align:center">n</th>
                  <th style="text-align:right">Mean CFU/mL</th>
                  <th style="text-align:right">RSD (%)</th>
                  <th>Nilai Individual CFU/mL</th>
                </tr>
              </thead>
              <tbody>{stat_rows}</tbody>
            </table>

            <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap">
              <div class="stat-block">
                <div class="stat-title">% Recovery</div>
                <div style="font-size:2rem;font-weight:700;color:{rec_color}">{rec_label}</div>
                <div style="color:{rec_color};font-size:12px;margin-top:4px">{rec_status}</div>
              </div>
              <div class="stat-block">
                <div class="stat-title">Rumus yang Digunakan</div>
                <div style="font-size:11px;margin-top:4px;font-family:monospace;color:#333">
                  CFU/mL = Koloni × {dilution} ÷ {volume}
                </div>
                <div style="font-size:10px;color:#888;margin-top:4px">FP={dilution}× | Vol={volume} mL</div>
              </div>
            </div>

            <h2 class="section-title">Kesimpulan Validasi</h2>
            <div class="interp-box {'pass-box' if interp.get('passed') else 'fail-box'}">
              <div class="interp-title">{interp.get('summary', '—')}</div>
              <div class="issues">{issues_html or '<em style="font-size:11px;color:#888">Tidak ada catatan tambahan.</em>'}</div>
            </div>

            <h2 class="section-title">Detail Per-Analisis (Acuan Perhitungan CFU)</h2>
            <table class="data-table">
              <thead>
                <tr>
                  <th>ID</th><th>File</th><th>Kelompok</th>
                  <th style="text-align:center">Koloni</th>
                  <th>Perhitungan (Koloni × FP ÷ Vol)</th>
                  <th style="text-align:right">CFU/mL</th>
                  <th style="text-align:center">QC</th>
                </tr>
              </thead>
              <tbody>{detail_rows or '<tr><td colspan="7" style="text-align:center;color:#aaa;padding:12px">Tidak ada data analisis</td></tr>'}</tbody>
            </table>"""

        # ══════════════════════════════════════════════════════════════════
        # KUALITATIF — Hasil Pengamatan Per Foto + Rekap + Kesimpulan
        # ══════════════════════════════════════════════════════════════════
        else:
            groups = session_data.get("groups", {})
            interp = session_data.get("interpretation", {})

            # ── Acuan kriteria kualitatif ──
            acuan_section = f"""
            <h2 class="section-title">Acuan Penilaian Kualitatif</h2>
            <div class="acuan-box">
              <div class="acuan-criteria">
                <div class="criteria-item">
                  <span class="criteria-badge pass">Positif</span>
                  <span class="criteria-desc">Terdeteksi pertumbuhan koloni (&gt; 0 koloni pada cawan) — Pertumbuhan mikroba terkonfirmasi</span>
                </div>
                <div class="criteria-item">
                  <span class="criteria-badge fail">Negatif</span>
                  <span class="criteria-desc">Tidak terdeteksi pertumbuhan koloni (0 koloni pada cawan) — Steril / tidak ada pertumbuhan</span>
                </div>
                <div class="criteria-item">
                  <span class="criteria-badge warn">Mixed</span>
                  <span class="criteria-desc">Hasil tidak konsisten antar ulangan foto dalam satu kelompok — Perlu investigasi</span>
                </div>
              </div>
              <div style="height:1px;background:#e0e0e0;margin:10px 0"></div>
              <div class="acuan-criteria">
                <div class="criteria-item" style="align-items:flex-start">
                  <span class="criteria-badge neutral" style="white-space:nowrap">Ekspektasi Kontrol</span>
                  <div style="font-size:11px">
                    <strong style="color:#2D6A4F">KP Produk &amp; KP Media</strong> → Positif (harus ada pertumbuhan)<br>
                    <strong style="color:#991B1B">KN Produk &amp; KN Media</strong> → Negatif (harus tidak ada pertumbuhan)
                  </div>
                </div>
              </div>
            </div>"""

            # ── Rekap ringkasan ──
            rekap_rows = ""
            for ctrl_key, ctrl_label, ctrl_color, expected_pos in ctrl_order:
                g      = groups.get(ctrl_key, {})
                mean_r = g.get("mean_result", "—")
                exp    = "Positif" if expected_pos else "Negatif"
                ok     = (mean_r == exp)
                status_icon  = "✓" if ok else ("⚠" if mean_r == "Mixed" else "✕")
                status_color = "#2D6A4F" if ok else ("#92400E" if mean_r == "Mixed" else "#991B1B")
                n_pos  = g.get("n_positif", 0)
                n_neg  = g.get("n_negatif", 0)
                n_tot  = n_pos + n_neg
                total_col = g.get("total_colonies_group", 0)
                rekap_rows += f"""
                <tr>
                  <td style="color:{ctrl_color};font-weight:600">{ctrl_label}</td>
                  <td style="text-align:center">{exp}</td>
                  <td style="text-align:center;color:{'#2D6A4F' if mean_r=='Positif' else '#991B1B' if mean_r=='Negatif' else '#92400E'};font-weight:700">{mean_r}</td>
                  <td style="text-align:center">{n_pos} Pos / {n_neg} Neg (n={n_tot})</td>
                  <td style="text-align:center">{total_col}</td>
                  <td style="text-align:center;color:{status_color};font-weight:700;font-size:13px">{status_icon}</td>
                </tr>"""

            # ── Hasil pengamatan per foto, per kelompok ──
            pengamatan_blocks = ""
            for ctrl_key, ctrl_label, ctrl_color, expected_pos in ctrl_order:
                g       = groups.get(ctrl_key, {})
                results = g.get("results", [])
                mean_r  = g.get("mean_result", "—")
                exp     = "Positif" if expected_pos else "Negatif"
                ok      = (mean_r == exp)
                group_status_color = "#2D6A4F" if ok else ("#92400E" if mean_r == "Mixed" else "#991B1B")
                group_status_icon  = "✓ Sesuai Ekspektasi" if ok else ("⚠ Mixed — Tidak Konsisten" if mean_r == "Mixed" else "✕ Tidak Sesuai Ekspektasi")

                foto_rows = ""
                for i, r in enumerate(results):
                    col_count = r.get("colony_count", 0)
                    result_str = r.get("result", "—")
                    fname      = r.get("filename", f"foto_{i+1}")
                    conf       = r.get("confidence") or "—"
                    res_color  = "#2D6A4F" if result_str == "Positif" else "#991B1B" if result_str == "Negatif" else "#92400E"
                    # Ikon visual koloni
                    col_bar = ("▰" * min(col_count, 10)) + ("▱" * max(0, 10 - min(col_count, 10)))
                    foto_rows += f"""
                    <tr>
                      <td style="text-align:center;font-weight:600;color:#555">{i+1}</td>
                      <td style="font-size:10px;color:#666;max-width:160px;word-break:break-all">{fname}</td>
                      <td style="text-align:center;font-weight:700;color:{res_color}">
                        {'⊕' if result_str=='Positif' else '⊖' if result_str=='Negatif' else '⚠'} {result_str}
                      </td>
                      <td style="text-align:center;font-weight:600">{col_count}</td>
                      <td style="font-size:9px;color:#888;letter-spacing:1px" title="{col_count} koloni">{col_bar}</td>
                      <td style="font-size:10px;color:#666">{conf}</td>
                    </tr>"""

                if not foto_rows:
                    foto_rows = '<tr><td colspan="6" style="text-align:center;color:#aaa;padding:8px">Tidak ada data foto</td></tr>'

                pengamatan_blocks += f"""
                <div class="obs-group">
                  <div class="obs-group-header" style="border-left:4px solid {ctrl_color}">
                    <span style="color:{ctrl_color};font-weight:700">{ctrl_label}</span>
                    <span style="font-size:11px;color:#888;margin-left:8px">Ekspektasi: {exp}</span>
                    <span style="margin-left:auto;color:{group_status_color};font-weight:700;font-size:12px">{group_status_icon}</span>
                  </div>
                  <table class="data-table" style="margin-bottom:0">
                    <thead>
                      <tr>
                        <th style="text-align:center;width:40px">Foto</th>
                        <th>Nama File</th>
                        <th style="text-align:center">Hasil</th>
                        <th style="text-align:center">Jml Koloni</th>
                        <th style="text-align:center">Distribusi</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>{foto_rows}</tbody>
                  </table>
                </div>"""

            # ── Issues ──
            issues_html = "".join(
                f'<div class="issue-item" style="color:{"#2D6A4F" if i.startswith("✓") else "#991B1B" if i.startswith("✕") else "#92400E"}">{i}</div>'
                for i in (interp.get("issues") or [])
            )

            mode_label = "Kualitatif (Positif / Negatif)"
            title_str  = "Laporan Validasi Sesi Pengembangan Kualitatif"

            main_section = f"""
            {acuan_section}

            <h2 class="section-title">Rekap Hasil per Kelompok Kontrol</h2>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Kelompok Kontrol</th><th style="text-align:center">Ekspektasi</th>
                  <th style="text-align:center">Hasil</th><th style="text-align:center">Distribusi (Pos/Neg)</th>
                  <th style="text-align:center">Total Koloni</th><th style="text-align:center">Status</th>
                </tr>
              </thead>
              <tbody>{rekap_rows}</tbody>
            </table>

            <h2 class="section-title">Kesimpulan Validasi</h2>
            <div class="interp-box {'pass-box' if interp.get('passed') else 'fail-box'}">
              <div class="interp-title">{interp.get('summary', '—')}</div>
              <div class="issues">{issues_html or '<em style="font-size:11px;color:#888">Tidak ada catatan tambahan.</em>'}</div>
            </div>

            <h2 class="section-title">Hasil Pengamatan Per Foto</h2>
            {pengamatan_blocks}"""

        # ══════════════════════════════════════════════════════════════════
        # HTML TEMPLATE LENGKAP
        # ══════════════════════════════════════════════════════════════════
        notes_clean = notes.split("kuali_session:")[0].split("dev_session:")[0].strip() if notes else ""

        html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px; color: #1a1a1a; background: #fff; padding: 28px; }}

  /* HEADER */
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; border-bottom: 3px solid #2D6A4F; padding-bottom: 14px; }}
  .logo {{ font-size: 22px; font-weight: 700; color: #2D6A4F; letter-spacing: -0.5px; }}
  .logo sub {{ font-size: 11px; color: #888; font-weight: 400; }}
  .title {{ font-size: 17px; font-weight: 700; margin-bottom: 4px; color: #111; }}
  .mode-badge {{ display: inline-block; background: #E8F5EE; color: #2D6A4F; padding: 2px 10px; border-radius: 20px; font-size: 10px; font-weight: 600; margin-bottom: 14px; }}
  .header-right {{ text-align: right; font-size: 10px; color: #888; line-height: 1.8; }}

  /* METADATA */
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5px 20px; background: #f8f8f8; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; border: 1px solid #eee; }}
  .meta-item {{ display: flex; gap: 6px; font-size: 11px; align-items: baseline; }}
  .meta-key {{ color: #888; min-width: 120px; flex-shrink: 0; }}
  .meta-val {{ font-weight: 600; color: #222; }}

  /* SECTION */
  .section-title {{ font-size: 12px; font-weight: 700; color: #2D6A4F; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; }}

  /* TABLES */
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 10.5px; margin-bottom: 12px; }}
  .data-table th {{ background: #2D6A4F; color: #fff; padding: 6px 8px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .data-table td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }}
  .data-table tr:nth-child(even) td {{ background: #fafafa; }}
  .data-table tr:hover td {{ background: #f0f7f2; }}

  /* ACUAN BOX */
  .acuan-box {{ background: #FAFDF8; border: 1px solid #B7DFC8; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }}
  .acuan-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 8px; }}
  .acuan-item {{ font-size: 11px; }}
  .acuan-label {{ font-weight: 700; color: #2D6A4F; margin-bottom: 4px; font-size: 10px; text-transform: uppercase; }}
  .acuan-formula {{ font-family: monospace; font-size: 11px; background: #fff; border: 1px solid #ddd; border-radius: 4px; padding: 5px 8px; color: #333; }}
  .acuan-criteria {{ display: flex; flex-direction: column; gap: 6px; }}
  .criteria-item {{ display: flex; align-items: center; gap: 10px; font-size: 11px; }}
  .criteria-badge {{ padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 700; white-space: nowrap; }}
  .criteria-badge.pass {{ background: #E8F5EE; color: #2D6A4F; border: 1px solid #2D6A4F; }}
  .criteria-badge.fail {{ background: #FEE2E2; color: #991B1B; border: 1px solid #991B1B; }}
  .criteria-badge.warn {{ background: #FEF3C7; color: #92400E; border: 1px solid #92400E; }}
  .criteria-badge.neutral {{ background: #DBEAFE; color: #1E40AF; border: 1px solid #1E40AF; }}
  .criteria-desc {{ color: #444; }}

  /* STAT BLOCKS */
  .stat-block {{ display: inline-block; background: #f8f8f8; border: 1px solid #e8e8e8; border-radius: 8px; padding: 12px 20px; margin: 4px 0 12px; min-width: 180px; }}
  .stat-title {{ font-size: 10px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; margin-bottom: 4px; }}

  /* INTERP BOX */
  .interp-box {{ border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  .pass-box {{ background: #E8F5EE; border: 1px solid #2D6A4F; }}
  .fail-box {{ background: #FEE2E2; border: 1px solid #991B1B; }}
  .interp-title {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
  .pass-box .interp-title {{ color: #2D6A4F; }}
  .fail-box .interp-title {{ color: #991B1B; }}
  .issue-item {{ padding: 4px 0; font-size: 11px; border-bottom: 1px solid rgba(0,0,0,0.06); }}

  /* PENGAMATAN GROUPS (kuali) */
  .obs-group {{ border: 1px solid #e8e8e8; border-radius: 6px; margin-bottom: 12px; overflow: hidden; }}
  .obs-group-header {{ display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #f8f8f8; border-bottom: 1px solid #eee; font-size: 12px; }}

  /* FOOTER */
  .footer {{ margin-top: 28px; border-top: 1px solid #eee; padding-top: 10px; font-size: 10px; color: #aaa; display: flex; justify-content: space-between; }}
  .std-ref {{ font-size: 9px; color: #bbb; margin-top: 4px; }}
</style>
</head>
<body>
  <!-- HEADER -->
  <div class="header">
    <div>
      <div class="logo">MicroCount <sub>by Colony Counter</sub></div>
      <div class="title">{title_str}</div>
      <div class="mode-badge">Mode: {mode_label}</div>
    </div>
    <div class="header-right">
      <div><strong>ID Laporan:</strong> {report_id}</div>
      <div><strong>Dicetak:</strong> {timestamp}</div>
      <div><strong>Standar:</strong> USP &lt;1223&gt; / ISO 29981 / FI Ed.VI</div>
    </div>
  </div>

  <!-- INFORMASI SAMPEL & SESI -->
  <div class="section-title" style="margin-top:0">Informasi Sampel &amp; Sesi</div>
  <div class="meta-grid">
    <div class="meta-item"><span class="meta-key">Nama Sampel</span><span class="meta-val">{sample_name}</span></div>
    <div class="meta-item"><span class="meta-key">Analis</span><span class="meta-val">{analyst}</span></div>
    <div class="meta-item"><span class="meta-key">Jenis Media</span><span class="meta-val">{media_type}</span></div>
    <div class="meta-item"><span class="meta-key">Jenis Organisme</span><span class="meta-val">{organism}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Inkubasi</span><span class="meta-val">{incubation}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Observasi</span><span class="meta-val">{observation}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Sesi</span><span class="meta-val">{created_at}</span></div>
    <div class="meta-item"><span class="meta-key">ID Sesi</span><span class="meta-val" style="font-family:monospace">{report_id}</span></div>
    {f'<div class="meta-item" style="grid-column:1/-1"><span class="meta-key">Catatan</span><span class="meta-val">{notes_clean[:200]}</span></div>' if notes_clean else ''}
  </div>

  <!-- KONTEN UTAMA (kuanti atau kuali) -->
  {main_section}

  <!-- FOOTER -->
  <div class="footer">
    <div>
      <div>MicroCount — Automated Microbial Colony Counter</div>
      <div class="std-ref">Acuan: USP &lt;1223&gt; Validation of Alternative Microbiological Methods | ISO 29981:2010 | FI Ed.VI Bab &lt;Uji Sterilitas&gt;</div>
    </div>
    <div style="text-align:right">
      <div>Session: {session_data.get('session_id', session_data.get('sample_id', '—'))[:16]}</div>
      <div>Generated: {timestamp}</div>
    </div>
  </div>
</body>
</html>"""

        pdf_dir = Path("reports/pdf")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        prefix = "quali" if is_quali else "dev"
        out_path = pdf_dir / f"{prefix}_session_{report_id}.pdf"
        HTML(string=html).write_pdf(str(out_path))
        logger.info(f"Session PDF report generated: {out_path}")
        return str(out_path)

    def _generate_dev_session_reportlab(self, session_data: dict, session_type: str) -> str:
        """
        ReportLab versi LENGKAP untuk generate_dev_session_report.
        Ini adalah path utama di Windows (WeasyPrint tidak support Windows).
        Konten sama dengan versi WeasyPrint: acuan formula, detail per-analisis,
        hasil pengamatan per foto.
        """
        from reportlab.platypus import KeepTogether
        from reportlab.lib.styles import ParagraphStyle

        is_quali = session_type == "qualitative"
        _sid_raw = session_data.get("session_id") or session_data.get("sample_id") or ""
        report_id = str(_sid_raw)[:8] if _sid_raw else str(_uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        interp    = session_data.get("interpretation", {})
        analyses  = session_data.get("analyses", [])

        # ── output path ──
        pdf_dir = Path("reports/pdf")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        prefix   = "quali" if is_quali else "dev"
        out_path = str(pdf_dir / f"{prefix}_session_{report_id}.pdf")

        doc  = SimpleDocTemplate(out_path, pagesize=A4,
                                 leftMargin=1.8*cm, rightMargin=1.8*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = self._build_styles()
        W = A4[0] - 3.6*cm   # usable width

        # ── style tambahan ──
        s_title2 = ParagraphStyle("title2", fontSize=14, fontName="Helvetica-Bold",
                                  textColor=BRAND_DARK, alignment=TA_CENTER, spaceAfter=4)
        s_sub    = ParagraphStyle("sub", fontSize=9, fontName="Helvetica",
                                  textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=2)
        s_hdr    = styles["section_header"]
        s_body   = styles["body"]
        s_small  = styles["small"]
        s_bold   = ParagraphStyle("bold9", fontSize=9, fontName="Helvetica-Bold")
        s_green  = ParagraphStyle("green9", fontSize=9, fontName="Helvetica-Bold", textColor=BRAND_GREEN)
        s_red    = ParagraphStyle("red9",   fontSize=9, fontName="Helvetica-Bold", textColor=BRAND_RED)
        s_amber  = ParagraphStyle("amber9", fontSize=9, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#92400E"))
        s_mono   = ParagraphStyle("mono9",  fontSize=8, fontName="Courier")

        def _p(txt, style=None): return Paragraph(str(txt), style or s_body)
        def _fmt_cfu(v):
            if v is None: return "—"
            try: return f"{float(v):.3e}"
            except: return str(v)
        def _fmt_rsd(v):
            if v is None: return "—"
            try: return f"{float(v):.2f}%"
            except: return str(v)

        ctrl_labels = {
            "positive_product": "KP Produk",
            "negative_product": "KN Produk",
            "negative_media":   "KN Media",
            "positive_media":   "KP Media",
        }
        ctrl_labels_full = {
            "positive_product": "Kontrol Positif Produk",
            "negative_product": "Kontrol Negatif Produk",
            "negative_media":   "Kontrol Negatif Media",
            "positive_media":   "Kontrol Positif Media",
        }
        ctrl_colors_rl = {
            "positive_product": BRAND_GREEN,
            "negative_product": BRAND_RED,
            "negative_media":   colors.HexColor("#92400E"),
            "positive_media":   colors.HexColor("#1E40AF"),
        }
        ctrl_order = ["positive_product", "negative_product", "negative_media", "positive_media"]
        ctrl_expected_pos = {"positive_product": True, "negative_product": False,
                              "negative_media": False, "positive_media": True}

        tbl_hdr_style = TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), BRAND_GREEN),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("FONTNAME",     (0,1), (0,-1), "Helvetica-Bold"),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.lightgrey),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
        ])

        def _make_table(rows, col_widths, extra_style=None):
            t = Table(rows, colWidths=col_widths)
            ts = TableStyle(tbl_hdr_style._cmds[:])
            if extra_style:
                for cmd in extra_style:
                    ts.add(*cmd)
            t.setStyle(ts)
            return t

        story = []

        # ════════════════════════════════════════════
        # HEADER
        # ════════════════════════════════════════════
        title_str = ("LAPORAN VALIDASI SESI PENGEMBANGAN KUALITATIF"
                     if is_quali else "LAPORAN VALIDASI SESI PENGEMBANGAN KUANTITATIF")
        story.append(_p(title_str, s_title2))
        mode_str  = ("Mode: Kualitatif (Positif / Negatif)"
                     if is_quali else "Mode: Kuantitatif (Hitung Koloni / CFU)")
        story.append(_p(f"{mode_str}  |  ID: {report_id}  |  {timestamp}", s_sub))
        story.append(_p("Standar Acuan: USP <1223> | ISO 29981:2010 | FI Ed.VI", s_sub))
        story.append(HRFlowable(width="100%", thickness=2, color=BRAND_GREEN, spaceAfter=8))

        # ════════════════════════════════════════════
        # INFORMASI SAMPEL
        # ════════════════════════════════════════════
        story.append(_p("INFORMASI SAMPEL & SESI", s_hdr))
        notes_raw   = session_data.get("notes") or ""
        notes_clean = notes_raw.split("kuali_session:")[0].split("dev_session:")[0].strip()
        meta_rows   = [
            [_p("Nama Sampel", s_bold), _p(session_data.get("sample_name") or "—"),
             _p("Analis", s_bold),      _p(session_data.get("analyst") or "—")],
            [_p("Jenis Media", s_bold), _p(session_data.get("media_type") or "—"),
             _p("Organisme", s_bold),   _p(session_data.get("organism_type") or "—")],
            [_p("Tgl Inkubasi", s_bold),_p(session_data.get("incubation_date") or "—"),
             _p("Tgl Observasi", s_bold),_p(session_data.get("observation_date") or "—")],
            [_p("ID Sesi", s_bold),     _p(report_id, s_mono),
             _p("Catatan", s_bold),     _p(notes_clean[:120] if notes_clean else "—")],
        ]
        story.append(_make_table(meta_rows, [W*0.18, W*0.32, W*0.18, W*0.32]))
        story.append(Spacer(1, 0.4*cm))

        # ════════════════════════════════════════════
        # KUANTITATIF
        # ════════════════════════════════════════════
        if not is_quali:
            stats    = session_data.get("statistics", {})
            recovery = stats.get("percent_recovery")
            rec_pass = stats.get("recovery_pass", False)

            # Ambil FP & volume dari analisis pertama
            first_a  = analyses[0] if analyses else {}
            dilution = first_a.get("dilution_factor", 1)
            volume   = first_a.get("inoculation_volume_ml", 1.0)

            # ── Acuan & Kriteria ──
            story.append(_p("ACUAN PERHITUNGAN & KRITERIA PENERIMAAN", s_hdr))
            acuan_rows = [
                [_p("Parameter", s_bold), _p("Keterangan", s_bold)],
                [_p("Rumus CFU/mL"),
                 _p("CFU/mL = Jumlah Koloni x Faktor Pengenceran / Volume Inokulasi (mL)", s_mono)],
                [_p("Parameter Sesi Ini"),
                 _p(f"Faktor Pengenceran: {dilution}x   |   Volume Inokulasi: {volume} mL")],
                [_p("% Recovery"),
                 _p("Syarat: 70% - 130%   (Acuan: USP <1223> / ISO 29981)")],
                [_p("RSD"),
                 _p("Syarat: <= 15% per kelompok   (Presisi antar-ulangan)")],
                [_p("Rumus % Recovery"),
                 _p("% Recovery = (Mean CFU KP Produk / Mean CFU KN Produk) x 100", s_mono)],
            ]
            story.append(_make_table(acuan_rows, [W*0.28, W*0.72],
                extra_style=[("BACKGROUND",(0,0),(-1,0), colors.HexColor("#E8F5EE")),
                             ("TEXTCOLOR",(0,0),(-1,0), BRAND_GREEN)]))
            story.append(Spacer(1, 0.4*cm))

            # ── Statistik per kelompok ──
            story.append(_p("STATISTIK HASIL ANALISIS", s_hdr))
            stat_rows_rl = [[
                _p("Kelompok Kontrol"), _p("n"), _p("Mean CFU/mL"),
                _p("RSD (%)"), _p("Nilai Individual CFU/mL"),
            ]]
            for ck in ctrl_order:
                st      = stats.get(ck, {})
                n       = st.get("n", 0)
                mean    = _fmt_cfu(st.get("mean"))
                rsd_val = st.get("rsd")
                rsd_str = _fmt_rsd(rsd_val)
                rsd_ok  = (rsd_val is not None and rsd_val <= 15)
                vals    = "  |  ".join(_fmt_cfu(x) for x in (st.get("values") or [])) or "—"
                clr     = ctrl_colors_rl.get(ck, colors.black)
                stat_rows_rl.append([
                    _p(ctrl_labels_full.get(ck, ck),
                       ParagraphStyle(f"c_{ck}", fontSize=9, fontName="Helvetica-Bold", textColor=clr)),
                    _p(str(n)),
                    _p(mean, s_mono),
                    _p(rsd_str, s_green if rsd_ok else s_amber),
                    _p(vals, s_mono),
                ])
            story.append(_make_table(stat_rows_rl,
                                     [W*0.27, W*0.05, W*0.16, W*0.12, W*0.40]))
            story.append(Spacer(1, 0.3*cm))

            # ── % Recovery block ──
            rec_color_rl = BRAND_GREEN if rec_pass else BRAND_RED
            rec_label    = f"{recovery:.2f}%" if recovery is not None else "—"
            rec_status   = ("Memenuhi Syarat (70-130%)" if rec_pass
                            else ("Di Luar Batas - Periksa Metode" if recovery is not None
                                  else "Tidak dapat dihitung"))
            rec_style = ParagraphStyle("rec", fontSize=20, fontName="Helvetica-Bold",
                                       textColor=rec_color_rl, alignment=TA_CENTER)
            rec_rows = [[
                _p("% Recovery", s_bold),
                _p(rec_label, rec_style),
                _p(rec_status, s_green if rec_pass else s_red),
                _p(f"Rumus yg dipakai:\nCFU/mL = Koloni x {dilution} / {volume}", s_mono),
            ]]
            story.append(_make_table(rec_rows, [W*0.18, W*0.18, W*0.34, W*0.30],
                extra_style=[("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#F8FDF8")),
                             ("BOX",(0,0),(-1,-1), 1, rec_color_rl)]))
            story.append(Spacer(1, 0.4*cm))

            # ── Kesimpulan ──
            story.append(_p("KESIMPULAN VALIDASI", s_hdr))
            passed  = interp.get("passed", False)
            summary = interp.get("summary", "—")
            box_color = colors.HexColor("#E8F5EE") if passed else colors.HexColor("#FEE2E2")
            box_border= BRAND_GREEN if passed else BRAND_RED
            conc_rows = [[_p(summary, s_green if passed else s_red)]]
            story.append(_make_table(conc_rows, [W],
                extra_style=[("BACKGROUND",(0,0),(-1,-1), box_color),
                             ("BOX",(0,0),(-1,-1), 1.5, box_border)]))
            story.append(Spacer(1, 0.15*cm))
            for issue in (interp.get("issues") or []):
                clr = s_green if issue.startswith("✓") else s_red if issue.startswith("✕") else s_amber
                story.append(_p(f"  {issue}", clr))
            story.append(Spacer(1, 0.4*cm))

            # ── Detail per analisis dengan acuan perhitungan ──
            story.append(_p("DETAIL PER-ANALISIS (ACUAN PERHITUNGAN CFU)", s_hdr))
            det_rows = [[
                _p("ID"), _p("File"), _p("Kelompok"),
                _p("Koloni"), _p("Perhitungan (K x FP / Vol)"), _p("CFU/mL"), _p("QC"),
            ]]
            for a in analyses:
                ck       = a.get("control_group", "—")
                col      = a.get("total_colonies", 0)
                cfu_fmt  = a.get("cfu_per_ml_formatted") or _fmt_cfu(a.get("cfu_per_ml"))
                dil_a    = a.get("dilution_factor", dilution)
                vol_a    = a.get("inoculation_volume_ml", volume)
                qc       = a.get("qc_grade") or "—"
                formula  = f"{col} x {dil_a} / {vol_a}" if a.get("cfu_per_ml") else "—"
                qc_style = s_green if qc in ("Excellent","Good") else s_amber if qc=="Fair" else s_red
                det_rows.append([
                    _p(a.get("analysis_id","")[:8], s_mono),
                    _p(a.get("filename","—")[:28]),
                    _p(ctrl_labels.get(ck, ck)),
                    _p(str(col)),
                    _p(formula, s_mono),
                    _p(cfu_fmt, s_mono),
                    _p(qc, qc_style),
                ])
            story.append(_make_table(det_rows,
                                     [W*0.10, W*0.22, W*0.13, W*0.07, W*0.20, W*0.17, W*0.11]))

        # ════════════════════════════════════════════
        # KUALITATIF
        # ════════════════════════════════════════════
        else:
            groups = session_data.get("groups", {})

            # ── Acuan penilaian ──
            story.append(_p("ACUAN PENILAIAN KUALITATIF", s_hdr))
            acuan_rows = [
                [_p("Hasil", s_bold), _p("Definisi", s_bold), _p("Ekspektasi Kontrol", s_bold)],
                [_p("Positif", s_green),
                 _p("Terdeteksi pertumbuhan koloni (> 0 koloni pada cawan)"),
                 _p("KP Produk & KP Media harus Positif", s_green)],
                [_p("Negatif", s_red),
                 _p("Tidak terdeteksi pertumbuhan koloni (0 koloni pada cawan)"),
                 _p("KN Produk & KN Media harus Negatif", s_red)],
                [_p("Mixed", s_amber),
                 _p("Hasil tidak konsisten antar ulangan foto dalam satu kelompok"),
                 _p("Perlu investigasi - tidak boleh ada Mixed", s_amber)],
            ]
            story.append(_make_table(acuan_rows, [W*0.13, W*0.52, W*0.35],
                extra_style=[("BACKGROUND",(0,0),(-1,0), colors.HexColor("#E8F5EE")),
                             ("TEXTCOLOR",(0,0),(-1,0), BRAND_GREEN)]))
            story.append(Spacer(1, 0.4*cm))

            # ── Rekap per kelompok ──
            story.append(_p("REKAP HASIL PER KELOMPOK KONTROL", s_hdr))
            rekap_rows = [[
                _p("Kelompok Kontrol"), _p("Ekspektasi"), _p("Hasil"),
                _p("Distribusi (Pos/Neg)"), _p("Total Koloni"), _p("Status"),
            ]]
            extra_rekap = []
            for i, ck in enumerate(ctrl_order):
                g       = groups.get(ck, {})
                mean_r  = g.get("mean_result", "—")
                exp     = "Positif" if ctrl_expected_pos.get(ck) else "Negatif"
                ok      = (mean_r == exp)
                n_pos   = g.get("n_positif", 0)
                n_neg   = g.get("n_negatif", 0)
                tot_col = g.get("total_colonies_group", 0)
                icon    = "OK" if ok else ("MIXED" if mean_r == "Mixed" else "TIDAK SESUAI")
                clr     = ctrl_colors_rl.get(ck, colors.black)
                res_s   = s_green if mean_r=="Positif" else s_red if mean_r=="Negatif" else s_amber
                sts_s   = s_green if ok else (s_amber if mean_r=="Mixed" else s_red)
                row_i   = i + 1
                rekap_rows.append([
                    _p(ctrl_labels_full.get(ck,ck),
                       ParagraphStyle(f"rk_{ck}", fontSize=9, fontName="Helvetica-Bold", textColor=clr)),
                    _p(exp),
                    _p(mean_r, res_s),
                    _p(f"{n_pos} Pos / {n_neg} Neg"),
                    _p(str(tot_col)),
                    _p(icon, sts_s),
                ])
                if not ok:
                    extra_rekap.append(("BACKGROUND",(0,row_i),(-1,row_i), colors.HexColor("#FFF5F5")))
            story.append(_make_table(rekap_rows,
                                     [W*0.27, W*0.13, W*0.12, W*0.20, W*0.13, W*0.15],
                                     extra_style=extra_rekap))
            story.append(Spacer(1, 0.4*cm))

            # ── Kesimpulan ──
            story.append(_p("KESIMPULAN VALIDASI", s_hdr))
            passed    = interp.get("passed", False)
            summary   = interp.get("summary", "—")
            box_color = colors.HexColor("#E8F5EE") if passed else colors.HexColor("#FEE2E2")
            box_border= BRAND_GREEN if passed else BRAND_RED
            conc_rows = [[_p(summary, s_green if passed else s_red)]]
            story.append(_make_table(conc_rows, [W],
                extra_style=[("BACKGROUND",(0,0),(-1,-1), box_color),
                             ("BOX",(0,0),(-1,-1), 1.5, box_border)]))
            story.append(Spacer(1, 0.15*cm))
            for issue in (interp.get("issues") or []):
                clr = s_green if issue.startswith("✓") else s_red if issue.startswith("✕") else s_amber
                story.append(_p(f"  {issue}", clr))
            story.append(Spacer(1, 0.4*cm))

            # ── Hasil pengamatan per foto, per kelompok ──
            story.append(_p("HASIL PENGAMATAN PER FOTO", s_hdr))
            for ck in ctrl_order:
                g       = groups.get(ck, {})
                results = g.get("results", [])
                if not results:
                    continue
                mean_r  = g.get("mean_result", "—")
                exp     = "Positif" if ctrl_expected_pos.get(ck) else "Negatif"
                ok      = (mean_r == exp)
                clr     = ctrl_colors_rl.get(ck, colors.black)
                sts_lbl = ("Sesuai Ekspektasi" if ok
                           else ("Mixed - Tidak Konsisten" if mean_r == "Mixed"
                                 else "TIDAK Sesuai Ekspektasi"))
                sts_s   = s_green if ok else (s_amber if mean_r=="Mixed" else s_red)

                # sub-header kelompok
                grp_hdr = [[
                    _p(ctrl_labels_full.get(ck,ck),
                       ParagraphStyle(f"gh_{ck}", fontSize=9, fontName="Helvetica-Bold", textColor=clr)),
                    _p(f"Ekspektasi: {exp}"),
                    _p(f"Hasil: {mean_r}", s_green if mean_r=="Positif" else s_red if mean_r=="Negatif" else s_amber),
                    _p(sts_lbl, sts_s),
                ]]
                story.append(_make_table(grp_hdr, [W*0.35, W*0.20, W*0.20, W*0.25],
                    extra_style=[("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#F0F4FF")),
                                 ("BOX",(0,0),(-1,-1), 1, clr)]))

                # baris per foto
                foto_rows_rl = [[
                    _p("No"), _p("Nama File"), _p("Hasil"),
                    _p("Jml Koloni"), _p("Confidence"),
                ]]
                extra_foto = []
                for fi, r in enumerate(results):
                    col_count  = r.get("colony_count", 0)
                    result_str = r.get("result", "—")
                    res_s      = s_green if result_str=="Positif" else s_red if result_str=="Negatif" else s_amber
                    row_idx    = fi + 1
                    # beri background merah muda jika hasil tidak sesuai ekspektasi
                    if result_str != exp:
                        extra_foto.append(("BACKGROUND",(0,row_idx),(-1,row_idx), colors.HexColor("#FFF5F5")))
                    foto_rows_rl.append([
                        _p(str(fi+1)),
                        _p(r.get("filename","—")[:35]),
                        _p(result_str, res_s),
                        _p(str(col_count)),
                        _p(r.get("confidence") or "—"),
                    ])
                story.append(_make_table(foto_rows_rl,
                                         [W*0.06, W*0.45, W*0.15, W*0.15, W*0.19],
                                         extra_style=extra_foto))
                story.append(Spacer(1, 0.3*cm))

        # ════════════════════════════════════════════
        # FOOTER
        # ════════════════════════════════════════════
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=4))
        story.append(_p(
            f"MicroCount — Automated Microbial Colony Counter  |  "
            f"Session: {session_data.get('session_id', session_data.get('sample_id','—'))[:16]}  |  "
            f"Dicetak: {timestamp}",
            s_small,
        ))
        story.append(_p(
            "Standar: USP <1223> Validation of Alternative Microbiological Methods  |  "
            "ISO 29981:2010  |  FI Ed.VI",
            s_small,
        ))

        doc.build(story)
        logger.info(f"Session PDF (ReportLab) generated: {out_path}")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers (tidak berubah dari versi asli)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_styles(self):
        base = getSampleStyleSheet()
        return {
            "report_title": ParagraphStyle(
                "report_title", fontSize=16, textColor=BRAND_DARK,
                spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold",
            ),
            "section_header": ParagraphStyle(
                "section_header", fontSize=11, textColor=BRAND_GREEN,
                spaceBefore=6, spaceAfter=4, fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "body", fontSize=9, leading=14, fontName="Helvetica",
            ),
            "small": ParagraphStyle(
                "small", fontSize=8, leading=12, textColor=colors.gray, fontName="Helvetica",
            ),
            "header_org": ParagraphStyle(
                "header_org", fontSize=14, textColor=BRAND_GREEN,
                fontName="Helvetica-Bold", alignment=TA_LEFT,
            ),
            "header_sub": ParagraphStyle(
                "header_sub", fontSize=9, textColor=colors.gray, fontName="Helvetica",
            ),
            "cfu_value": ParagraphStyle(
                "cfu_value", fontSize=18, textColor=BRAND_GREEN,
                fontName="Helvetica-Bold", alignment=TA_CENTER,
            ),
            "warning": ParagraphStyle(
                "warning", fontSize=9, leading=12,
                textColor=colors.HexColor("#E65100"), fontName="Helvetica",
            ),
        }

    def _build_header(self, styles) -> list:
        data = [[
            Paragraph("MicroCount", styles["header_org"]),
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ParagraphStyle("hr", fontSize=8, textColor=colors.gray,
                               alignment=TA_RIGHT, fontName="Helvetica"),
            ),
        ]]
        t = Table(data, colWidths=["70%", "30%"])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return [t]

    def _build_sample_info(self, styles, result, **kwargs) -> list:
        elements = [Paragraph("SAMPLE INFORMATION", styles["section_header"])]
        data = [
            ["Sample Name", kwargs.get("sample_name", "—"),
             "Analysis ID", result.analysis_id[:16] + "..."],
            ["Analyst", kwargs.get("analyst", "—"),
             "Processing Time", f"{result.processing_time_ms:.0f} ms"],
            ["Media Type", kwargs.get("media_type") or "—",
             "Organism Type", kwargs.get("organism_type") or "—"],
            ["Incubation Date", kwargs.get("incubation_date") or "—",
             "Observation Date", kwargs.get("observation_date") or "—"],
        ]
        t = Table(data, colWidths=["22%", "28%", "22%", "28%"])
        t.setStyle(self._info_table_style())
        elements.append(t)
        return elements

    def _build_analysis_summary(self, styles, result: AnalysisResult) -> list:
        elements = [Paragraph("ANALYSIS SUMMARY", styles["section_header"])]
        data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Total Colonies", str(result.total_colonies),
             "Colony Density", f"{result.colony_density:.1f} col/cm²"],
            ["Q1 (Top-Right)", str(result.colonies_per_quadrant.get("Q1", 0)),
             "Q2 (Top-Left)", str(result.colonies_per_quadrant.get("Q2", 0))],
            ["Q3 (Bot-Left)", str(result.colonies_per_quadrant.get("Q3", 0)),
             "Q4 (Bot-Right)", str(result.colonies_per_quadrant.get("Q4", 0))],
            ["Petri Detected", "Yes" if result.petri_dish.detected else "No",
             "Detection Method", result.petri_dish.detection_method.title()],
            ["QC Grade", result.quality_control.grade,
             "QC Score", f"{result.quality_control.score:.1%}"],
        ]
        t = Table(data, colWidths=["22%", "28%", "22%", "28%"])
        style = self._info_table_style()
        style.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        style.add("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN)
        style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
        t.setStyle(style)
        elements.append(t)
        return elements

    def _build_cfu_section(self, styles, result: AnalysisResult) -> list:
        elements = [Paragraph("CFU CALCULATION RESULTS", styles["section_header"])]
        cfu = result.cfu
        cfu_table_data = [[
            Paragraph("CFU/mL Result", styles["body"]),
            Paragraph(cfu.cfu_per_ml_formatted, styles["cfu_value"]),
        ]]
        cfu_main = Table(cfu_table_data, colWidths=["40%", "60%"])
        cfu_main.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 1, BRAND_GREEN),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(cfu_main)
        elements.append(Spacer(1, 0.2 * cm))
        data = [
            ["Parameter", "Value"],
            ["Total Colony Count", str(cfu.total_colonies)],
            ["Dilution Factor",
             f"1:{int(cfu.dilution_factor):,}" if cfu.dilution_factor >= 1 else str(cfu.dilution_factor)],
            ["Inoculation Volume", f"{cfu.inoculation_volume_ml} mL"],
            ["CFU/mL (Numeric)", f"{cfu.cfu_per_ml:,.2f}"],
            ["Scientific Notation", cfu.scientific_notation],
        ]
        t = Table(data, colWidths=["50%", "50%"])
        t.setStyle(self._info_table_style())
        elements.append(t)
        return elements

    def _build_qc_section(self, styles, result: AnalysisResult) -> list:
        elements = [Paragraph("IMAGE QUALITY CONTROL", styles["section_header"])]
        qc = result.quality_control
        data = [
            ["QC Parameter", "Score", "Status"],
            ["Focus", f"{qc.focus_score:.1%}", "✓" if qc.focus_score > 0.5 else "⚠"],
            ["Blur",  f"{qc.blur_score:.1%}",  "✓" if qc.blur_score < 0.5 else "⚠"],
            ["Glare", f"{qc.glare_score:.1%}", "✓" if qc.glare_score > 0.6 else "⚠"],
            ["Shadow", f"{qc.shadow_score:.1%}", "✓" if qc.shadow_score > 0.6 else "⚠"],
            ["Petri Visibility", f"{qc.visibility_score:.1%}", "✓" if qc.visibility_score > 0.5 else "⚠"],
            ["Overall Score", f"{qc.score:.1%}", qc.grade],
        ]
        t = Table(data, colWidths=["50%", "25%", "25%"])
        style = self._info_table_style()
        style.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        style.add("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN)
        style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
        t.setStyle(style)
        elements.append(t)
        if qc.issues:
            elements.append(Spacer(1, 0.2 * cm))
            for issue in qc.issues:
                elements.append(Paragraph(f"⚠ {issue}", styles["warning"]))
        return elements

    def _build_distribution_section(self, styles, result: AnalysisResult) -> list:
        elements = [Paragraph("COLONY DISTRIBUTION BY QUADRANT", styles["section_header"])]
        total = result.total_colonies or 1
        data = [
            ["Quadrant", "Colony Count", "Percentage"],
            ["Q1 (Top-Right)", str(result.colonies_per_quadrant.get("Q1", 0)),
             f"{result.colonies_per_quadrant.get('Q1', 0)/total*100:.1f}%"],
            ["Q2 (Top-Left)", str(result.colonies_per_quadrant.get("Q2", 0)),
             f"{result.colonies_per_quadrant.get('Q2', 0)/total*100:.1f}%"],
            ["Q3 (Bottom-Left)", str(result.colonies_per_quadrant.get("Q3", 0)),
             f"{result.colonies_per_quadrant.get('Q3', 0)/total*100:.1f}%"],
            ["Q4 (Bottom-Right)", str(result.colonies_per_quadrant.get("Q4", 0)),
             f"{result.colonies_per_quadrant.get('Q4', 0)/total*100:.1f}%"],
            ["TOTAL", str(result.total_colonies), "100%"],
        ]
        t = Table(data, colWidths=["40%", "30%", "30%"])
        style = self._info_table_style()
        style.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        style.add("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
        style.add("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN)
        style.add("TEXTCOLOR", (0, 0), (-1, 0), colors.white)
        style.add("BACKGROUND", (0, -1), (-1, -1), BRAND_LIGHT)
        t.setStyle(style)
        elements.append(t)
        return elements

    def _build_morphology_section(self, styles, result: AnalysisResult) -> list:
        elements = [Paragraph("COLONY MORPHOLOGY SUMMARY", styles["section_header"])]
        m = result.morphology_summary
        data = [
            ["Morphology Parameter", "Value"],
            ["Pinpoint Colonies (< 0.5mm)", str(m.pinpoint_count)],
            ["Small Colonies (0.5–2mm)", str(m.small_count)],
            ["Medium Colonies (2–5mm)", str(m.medium_count)],
            ["Large Colonies (> 5mm)", str(m.large_count)],
            ["Mean Diameter", f"{m.mean_diameter_mm:.2f} mm" if m.mean_diameter_mm else "N/A"],
            ["Mean Circularity", f"{m.mean_circularity:.3f}" if m.mean_circularity else "N/A"],
            ["Dominant Color", str(m.dominant_color or "N/A")],
        ]
        t = Table(data, colWidths=["60%", "40%"])
        t.setStyle(self._info_table_style())
        elements.append(t)
        return elements

    def _build_image_section(self, styles, image_path: str) -> list:
        elements = [Paragraph("ANNOTATED IMAGE", styles["section_header"])]
        try:
            img = Image(image_path, width=12 * cm, height=12 * cm, kind="proportional")
            elements.append(img)
        except Exception as e:
            logger.warning(f"Could not embed image in PDF: {e}")
            elements.append(Paragraph(f"[Image: {image_path}]", styles["small"]))
        return elements

    def _build_warnings(self, styles, warnings: list[str]) -> list:
        elements = [Paragraph("WARNINGS & NOTES", styles["section_header"])]
        for w in warnings:
            elements.append(Paragraph(f"• {w}", styles["warning"]))
        return elements

    def _build_footer(self, styles, result) -> list:
        analysis_id = getattr(result, "analysis_id", "—")
        return [
            HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
            Spacer(1, 0.2 * cm),
            Paragraph(
                f"Generated by MicroCount v1.0 | Analysis ID: {analysis_id} | "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["small"],
            ),
            Paragraph(
                "This report is for laboratory use only. Results should be interpreted "
                "by qualified personnel in accordance with applicable standards (ISO 7218, USP, APHA).",
                styles["small"],
            ),
        ]

    def _info_table_style(self) -> TableStyle:
        return TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F5E9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ])
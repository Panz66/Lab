"""
MicroCount — Patch: generate_dev_session_report
Tambahkan method ini ke class PDFReportGenerator di app/reports/report_generator.py
"""

# ─── Tempelkan method ini ke dalam class PDFReportGenerator ─────────────────

def generate_dev_session_report(self, session_data: dict, session_type: str = "quantitative") -> str:
    """
    Generate satu PDF ringkasan untuk seluruh sesi pengembangan.
    session_type: 'quantitative' atau 'qualitative'
    """
    import uuid
    from pathlib import Path
    from datetime import datetime

    try:
        from weasyprint import HTML, CSS
    except ImportError:
        raise RuntimeError("weasyprint tidak terinstal. Jalankan: pip install weasyprint")

    is_quali = session_type == "qualitative"
    report_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    sample_name  = session_data.get("sample_name", "—")
    analyst      = session_data.get("analyst", "—")
    media_type   = session_data.get("media_type") or "—"
    organism     = session_data.get("organism_type") or "—"
    incubation   = session_data.get("incubation_date") or "—"
    observation  = session_data.get("observation_date") or "—"
    notes        = session_data.get("notes") or ""
    created_at   = session_data.get("created_at", timestamp)[:10] if session_data.get("created_at") else timestamp

    # ── Blok statistik (kuantitatif) ──────────────────────────────────────
    ctrl_order = [
        ("positive_product", "Kontrol Positif Produk", "#2D6A4F"),
        ("negative_product", "Kontrol Negatif Produk", "#991B1B"),
        ("negative_media",   "Kontrol Negatif Media",  "#92400E"),
        ("positive_media",   "Kontrol Positif Media",  "#1E40AF"),
    ]

    if not is_quali:
        stats = session_data.get("statistics", {})
        recovery    = stats.get("percent_recovery")
        rec_pass    = stats.get("recovery_pass")
        interp      = session_data.get("interpretation", {})

        def _fmt(v):
            if v is None: return "—"
            try: return f"{float(v):.2e}"
            except: return str(v)

        def _rsd(v):
            if v is None: return "—"
            try: return f"{float(v):.2f}%"
            except: return str(v)

        stat_rows = ""
        for ctrl_key, ctrl_label, ctrl_color in ctrl_order:
            st = stats.get(ctrl_key, {})
            vals = "; ".join(_fmt(x) for x in (st.get("values") or []))
            stat_rows += f"""
            <tr>
              <td style="color:{ctrl_color};font-weight:600">{ctrl_label}</td>
              <td style="text-align:center">{st.get('n', 0)}</td>
              <td style="text-align:right;font-family:monospace">{_fmt(st.get('mean'))}</td>
              <td style="text-align:right">{_rsd(st.get('rsd'))}</td>
              <td style="font-size:10px;color:#555">{vals or '—'}</td>
            </tr>"""

        rec_color = "#2D6A4F" if rec_pass else "#991B1B"
        rec_label = f"{recovery:.2f}%" if recovery is not None else "—"
        rec_status = "✓ Memenuhi Syarat (70–130%)" if rec_pass else ("✕ Tidak Memenuhi Syarat" if recovery is not None else "—")

        recovery_html = f"""
        <div class="stat-block">
          <div class="stat-title">% Recovery</div>
          <div style="font-size:2rem;font-weight:700;color:{rec_color}">{rec_label}</div>
          <div style="color:{rec_color};font-size:13px;margin-top:4px">{rec_status}</div>
        </div>"""

        issues_html = "".join(
            f'<div class="issue-item" style="color:{"#2D6A4F" if i.startswith("✓") else "#991B1B" if i.startswith("✕") else "#92400E"}">{i}</div>'
            for i in (interp.get("issues") or [])
        )

        stat_section = f"""
        <h2 class="section-title">Statistik Hasil Analisis</h2>
        <table class="data-table">
          <thead>
            <tr>
              <th>Kelompok Kontrol</th><th>n</th>
              <th>Mean CFU/mL</th><th>RSD</th><th>Nilai Individual</th>
            </tr>
          </thead>
          <tbody>{stat_rows}</tbody>
        </table>
        {recovery_html}
        <h2 class="section-title">Interpretasi</h2>
        <div class="interp-box {'pass-box' if interp.get('passed') else 'fail-box'}">
          <div class="interp-title">{interp.get('summary', '—')}</div>
          <div class="issues">{issues_html}</div>
        </div>"""

    else:
        # ── Kualitatif ────────────────────────────────────────────────────
        groups   = session_data.get("groups", {})
        interp   = session_data.get("interpretation", {})
        exp_map  = {"positive_product": True, "negative_product": False,
                    "negative_media": False, "positive_media": True}

        group_rows = ""
        for ctrl_key, ctrl_label, ctrl_color in ctrl_order:
            g = groups.get(ctrl_key, {})
            mean_r = g.get("mean_result", "—")
            exp    = "Positif" if exp_map.get(ctrl_key) else "Negatif"
            ok     = (exp == mean_r)
            status_icon = "✓" if ok else ("⚠" if mean_r == "Mixed" else "✕")
            status_color = "#2D6A4F" if ok else ("#92400E" if mean_r == "Mixed" else "#991B1B")
            n_pos  = g.get("n_positif", 0)
            n_neg  = g.get("n_negatif", 0)
            total_c = g.get("total_colonies_group", 0)

            # sub-rows foto
            foto_rows = ""
            for r in g.get("results", []):
                res_color = "#2D6A4F" if r.get("result") == "Positif" else "#991B1B"
                foto_rows += f"""
                <tr style="font-size:10px;color:#555">
                  <td style="padding-left:16px">↳ Foto {r.get('foto', '?')}: {r.get('filename','')}</td>
                  <td colspan="4" style="color:{res_color};font-weight:600">
                    {r.get('result','—')} — {r.get('colony_count',0)} koloni ({r.get('confidence','—')})
                  </td>
                </tr>"""

            group_rows += f"""
            <tr>
              <td style="color:{ctrl_color};font-weight:600">{ctrl_label}</td>
              <td style="text-align:center">{exp}</td>
              <td style="text-align:center;color:{"#2D6A4F" if mean_r=="Positif" else "#991B1B" if mean_r=="Negatif" else "#92400E"};font-weight:600">{mean_r}</td>
              <td style="text-align:center">{n_pos} / {n_neg}</td>
              <td style="text-align:center;color:{status_color};font-weight:700">{status_icon}</td>
              <td style="text-align:right">{total_c}</td>
            </tr>
            {foto_rows}"""

        issues_html = "".join(
            f'<div class="issue-item" style="color:{"#2D6A4F" if i.startswith("✓") else "#991B1B" if i.startswith("✕") else "#92400E"}">{i}</div>'
            for i in (interp.get("issues") or [])
        )

        stat_section = f"""
        <h2 class="section-title">Hasil Analisis Kualitatif</h2>
        <table class="data-table">
          <thead>
            <tr>
              <th>Kelompok Kontrol</th><th>Ekspektasi</th>
              <th>Hasil</th><th>Pos/Neg</th><th>Status</th><th>Total Koloni</th>
            </tr>
          </thead>
          <tbody>{group_rows}</tbody>
        </table>
        <h2 class="section-title">Interpretasi</h2>
        <div class="interp-box {'pass-box' if interp.get('passed') else 'fail-box'}">
          <div class="interp-title">{interp.get('summary', '—')}</div>
          <div class="issues">{issues_html}</div>
        </div>"""

    # ── Tabel detail analisis individual ──────────────────────────────────
    detail_rows = ""
    for a in session_data.get("analyses", []):
        if is_quali:
            res    = a.get("qualitative_result", "—")
            col    = a.get("colony_count", 0)
            grade  = a.get("qc_grade") or "—"
            extra  = f'<td style="color:{"#2D6A4F" if res=="Positif" else "#991B1B"};font-weight:600">{res}</td><td>{col}</td><td>{grade}</td>'
        else:
            cfu    = a.get("cfu_per_ml_formatted") or "—"
            col    = a.get("total_colonies", 0)
            grade  = a.get("qc_grade") or "—"
            extra  = f'<td>{col}</td><td style="font-family:monospace">{cfu}</td><td>{grade}</td>'
        ctrl_lbl = dict((k, v["label"]) for k, v in {
            "positive_product": {"label":"KP Produk"},
            "negative_product": {"label":"KN Produk"},
            "negative_media":   {"label":"KN Media"},
            "positive_media":   {"label":"KP Media"},
        }.items()).get(a.get("control_group", ""), a.get("control_group", "—"))
        detail_rows += f"""
        <tr>
          <td style="font-size:10px;color:#777">{a.get('analysis_id','')[:8]}</td>
          <td style="font-size:11px">{a.get('filename','—')}</td>
          <td>{ctrl_lbl}</td>
          {extra}
        </tr>"""

    detail_headers = (
        "<th>Hasil</th><th>Koloni</th><th>QC</th>"
        if is_quali else
        "<th>Koloni</th><th>CFU/mL</th><th>QC</th>"
    )

    mode_label = "Kualitatif (Positif/Negatif)" if is_quali else "Kuantitatif (Hitung Koloni)"
    title_str  = "Laporan Sesi Pengembangan Kualitatif" if is_quali else "Laporan Sesi Pengembangan Kuantitatif"

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px; color: #1a1a1a; background: #fff; padding: 24px; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; border-bottom: 2px solid #2D6A4F; padding-bottom: 14px; }}
  .logo {{ font-size: 22px; font-weight: 700; color: #2D6A4F; letter-spacing: -0.5px; }}
  .logo sub {{ font-size: 11px; color: #888; font-weight: 400; }}
  .header-right {{ text-align: right; font-size: 10px; color: #888; }}
  .title {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; color: #111; }}
  .mode-badge {{ display: inline-block; background: #E8F5EE; color: #2D6A4F; padding: 2px 10px; border-radius: 20px; font-size: 10px; font-weight: 600; margin-bottom: 14px; }}
  .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 20px; background: #f8f8f8; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; }}
  .meta-item {{ display: flex; gap: 6px; font-size: 11px; }}
  .meta-key {{ color: #888; min-width: 110px; }}
  .meta-val {{ font-weight: 600; }}
  .section-title {{ font-size: 13px; font-weight: 700; color: #2D6A4F; margin: 18px 0 8px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-bottom: 12px; }}
  .data-table th {{ background: #2D6A4F; color: #fff; padding: 6px 8px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .data-table td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  .data-table tr:nth-child(even) td {{ background: #fafafa; }}
  .stat-block {{ display: inline-block; background: #f8f8f8; border-radius: 8px; padding: 12px 20px; margin: 8px 0 14px; min-width: 180px; }}
  .stat-title {{ font-size: 10px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; margin-bottom: 4px; }}
  .interp-box {{ border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; }}
  .pass-box {{ background: #E8F5EE; border: 1px solid #2D6A4F; }}
  .fail-box {{ background: #FEE2E2; border: 1px solid #991B1B; }}
  .interp-title {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; }}
  .pass-box .interp-title {{ color: #2D6A4F; }}
  .fail-box .interp-title {{ color: #991B1B; }}
  .issue-item {{ padding: 3px 0; font-size: 11px; border-bottom: 1px solid rgba(0,0,0,0.06); }}
  .footer {{ margin-top: 24px; border-top: 1px solid #eee; padding-top: 10px; font-size: 10px; color: #aaa; display: flex; justify-content: space-between; }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <div class="logo">MicroCount <sub>by Colony Counter</sub></div>
      <div class="title">{title_str}</div>
      <div class="mode-badge">Mode: {mode_label}</div>
    </div>
    <div class="header-right">
      <div>ID Laporan: {report_id}</div>
      <div>Dicetak: {timestamp}</div>
    </div>
  </div>

  <div class="meta-grid">
    <div class="meta-item"><span class="meta-key">Nama Sampel</span><span class="meta-val">{sample_name}</span></div>
    <div class="meta-item"><span class="meta-key">Analis</span><span class="meta-val">{analyst}</span></div>
    <div class="meta-item"><span class="meta-key">Jenis Media</span><span class="meta-val">{media_type}</span></div>
    <div class="meta-item"><span class="meta-key">Organisme</span><span class="meta-val">{organism}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Inkubasi</span><span class="meta-val">{incubation}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Observasi</span><span class="meta-val">{observation}</span></div>
    <div class="meta-item"><span class="meta-key">Tanggal Sesi</span><span class="meta-val">{created_at}</span></div>
    {f'<div class="meta-item"><span class="meta-key">Catatan</span><span class="meta-val">{notes[:120]}</span></div>' if notes else ''}
  </div>

  {stat_section}

  <h2 class="section-title">Detail Analisis Individual</h2>
  <table class="data-table">
    <thead>
      <tr><th>ID</th><th>File</th><th>Kelompok</th>{detail_headers}</tr>
    </thead>
    <tbody>{detail_rows or '<tr><td colspan="6" style="text-align:center;color:#aaa;padding:12px">Tidak ada data</td></tr>'}</tbody>
  </table>

  <div class="footer">
    <div>MicroCount — Automated Microbial Colony Counter</div>
    <div>Session ID: {session_data.get('session_id', session_data.get('sample_id', '—'))[:16]}</div>
  </div>
</body>
</html>"""

    pdf_dir = Path("reports/pdf")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    prefix = "quali" if is_quali else "dev"
    out_path = pdf_dir / f"{prefix}_session_{report_id}.pdf"
    HTML(string=html).write_pdf(str(out_path))
    return str(out_path)

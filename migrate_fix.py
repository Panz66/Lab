"""
MicroCount — Database Migration Script
Jalankan SEKALI untuk memperbaiki database yang sudah ada.

Apa yang dilakukan:
  1. Hapus orphan Report records (reports.analysis_id yang tidak ada di analyses)
  2. Rekonstruksi foreign key referential integrity

Cara pakai:
    python migrate_fix.py
    
(Jalankan dari root folder MicroCount, dengan venv aktif)
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main():
    from app.core.database import AsyncSessionLocal, init_db  # ← ganti ini
    from app.models import database as db_models
    from sqlalchemy import select, delete, text

    print("Connecting to database...")
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. Hapus orphan Report records
        print("Checking for orphan Report records...")
        orphan_reports = await session.execute(
            text("""
                SELECT r.id, r.analysis_id 
                FROM reports r
                LEFT JOIN analyses a ON r.analysis_id = a.id
                WHERE a.id IS NULL
            """)
        )
        orphans = orphan_reports.fetchall()
        if orphans:
            print(f"Found {len(orphans)} orphan Report(s). Deleting...")
            orphan_ids = [row[0] for row in orphans]
            from sqlalchemy import delete as sql_delete
            await session.execute(
                sql_delete(db_models.Report).where(
                    db_models.Report.id.in_(orphan_ids)
                )
            )
            await session.commit()
            print(f"Deleted {len(orphans)} orphan report(s).")
        else:
            print("No orphan reports found.")

        # 2. Verifikasi konsistensi
        total_reports = (await session.execute(
            text("SELECT COUNT(*) FROM reports")
        )).scalar()
        total_analyses = (await session.execute(
            text("SELECT COUNT(*) FROM analyses")
        )).scalar()
        print(f"Database stats: {total_analyses} analyses, {total_reports} reports.")
        print("Migration complete.")


if __name__ == "__main__":
    asyncio.run(main())

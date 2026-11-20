"""Build the relational DuckDB file used by --sources duckdb and duckdb_faiss.

Usage:
    python scripts/build_duckdb.py --tables-dir data/mimic-iv-demo --out data/clinrag.duckdb

--tables-dir can be the hosp folder or any folder above it. Keep --out under
data/ (git-ignored): the file holds MIMIC-derived rows and must never be committed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from clinrag.sources.duckdb_relational import build_database


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", type=Path, default=Path("data/mimic-iv-demo"))
    parser.add_argument("--out", type=Path, default=Path("data/clinrag.duckdb"))
    parser.add_argument("--force", action="store_true", help="replace --out if it exists")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    out = build_database(args.tables_dir, args.out, overwrite=args.force)
    con = duckdb.connect(str(out), read_only=True)
    try:
        for table in ("admissions", "labevents", "diagnoses", "dx_codes", "d_labitems", "d_icd_diagnoses"):
            print(f"{table}: {con.execute(f'SELECT count(*) FROM {table}').fetchall()[0][0]} rows")
    finally:
        con.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - start:.1f} s")
    return out


if __name__ == "__main__":
    main()

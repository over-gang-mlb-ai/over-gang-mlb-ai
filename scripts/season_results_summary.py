#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize WIN / LOSS / PUSH / PENDING counts for OU_Result and ML_Result across
archived predictions CSVs for a calendar season (year). Read-only; no ROI/profit.

Expects filenames like predictions_YYYYMMDD_HHMM.csv (slate date in filename).

Usage:
  python scripts/season_results_summary.py --season 2026
  python scripts/season_results_summary.py --season 2026 --archive-dir archive
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def _year_from_predictions_filename(path: Path) -> Optional[int]:
    """Year from predictions_YYYYMMDD_HHMM.csv; None if pattern does not match."""
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", path.name, re.IGNORECASE)
    if not m:
        return None
    yyyymmdd = m.group(1)
    return int(yyyymmdd[:4])


def _bucket_result(val: Optional[str]) -> str:
    if val is None:
        return "PENDING"
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return "PENDING"
    u = s.upper()
    if u == "PENDING":
        return "PENDING"
    if u in ("WIN", "LOSS", "PUSH"):
        return u
    return "PENDING"


def _tally_rows(rows: list[dict[str, str]], col: str) -> dict[str, int]:
    out = {"WIN": 0, "LOSS": 0, "PUSH": 0, "PENDING": 0}
    for row in rows:
        raw = row.get(col)
        b = _bucket_result(raw if raw is not None else None)
        out[b] += 1
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _win_rate_ex_push_pending(counts: dict[str, int]) -> Optional[float]:
    w, l = counts["WIN"], counts["LOSS"]
    denom = w + l
    if denom == 0:
        return None
    return w / denom


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Season summary of OU_Result and ML_Result from archived predictions CSVs."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=datetime.now().year,
        metavar="YEAR",
        help="Calendar year to include (default: current year)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "archive",
        help="Directory containing predictions_*.csv (default: <repo>/archive)",
    )
    args = parser.parse_args()
    season_year: int = args.season
    archive_dir: Path = args.archive_dir.resolve()

    if not archive_dir.is_dir():
        print(f"Archive directory not found: {archive_dir}", file=sys.stderr)
        return 1

    files = sorted(archive_dir.glob("predictions_*.csv"))
    matched: list[Path] = []
    skipped_name = 0
    for f in files:
        y = _year_from_predictions_filename(f)
        if y is None:
            skipped_name += 1
            continue
        if y == season_year:
            matched.append(f)

    if not matched:
        print(f"No predictions_*.csv files for season {season_year} under {archive_dir}")
        if skipped_name:
            print(f"({skipped_name} file(s) skipped: filename not predictions_YYYYMMDD_HHMM.csv)")
        return 0

    ou_tot = {"WIN": 0, "LOSS": 0, "PUSH": 0, "PENDING": 0}
    ml_tot = {"WIN": 0, "LOSS": 0, "PUSH": 0, "PENDING": 0}
    total_rows = 0
    read_errors = 0

    for f in matched:
        try:
            rows = _read_csv_rows(f)
        except Exception:
            read_errors += 1
            continue
        total_rows += len(rows)
        ou = _tally_rows(rows, "OU_Result")
        ml = _tally_rows(rows, "ML_Result")
        for k in ou_tot:
            ou_tot[k] += ou[k]
        for k in ml_tot:
            ml_tot[k] += ml[k]

    print("==============================")
    print(f"SEASON RESULTS SUMMARY ({season_year})")
    print("==============================")
    print(f"Archive: {archive_dir}")
    print(f"Files included: {len(matched)}")
    print(f"Rows included: {total_rows}")
    if read_errors:
        print(f"Files failed to read: {read_errors}", file=sys.stderr)
    if skipped_name:
        print(f"Files skipped (bad filename pattern): {skipped_name}")

    def _print_block(title: str, counts: dict[str, int]) -> None:
        print()
        print(title)
        print(f"  WIN:     {counts['WIN']}")
        print(f"  LOSS:    {counts['LOSS']}")
        print(f"  PUSH:    {counts['PUSH']}")
        print(f"  PENDING: {counts['PENDING']}")
        wr = _win_rate_ex_push_pending(counts)
        if wr is None:
            print("  Win rate (excl. push & pending): n/a (no wins/losses)")
        else:
            print(f"  Win rate (excl. push & pending): {wr * 100:.1f}%")

    _print_block("OU_Result", ou_tot)
    _print_block("ML_Result", ml_tot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

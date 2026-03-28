#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append graded O/U and ML bets from a graded predictions CSV into a dedicated season tracker:
  tracking/season_results_YYYY.csv

One row per graded bet; separate rows for O/U and ML when both are graded.
Skips PENDING; does not re-append the same (Source_File, Game, Bet_Type) if already in the tracker.

Does not modify prediction generation or archive files (read-only on source CSV).

Usage:
  python scripts/append_season_tracker.py archive/predictions_20260327_0257.csv
  python scripts/append_season_tracker.py path/to/predictions.csv --date 2026-03-27
  python scripts/append_season_tracker.py path/to/predictions.csv --season 2026
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "tracking"

TRACKER_COLUMNS = [
    "Date",
    "Game",
    "Bet_Type",
    "Pick",
    "Result",
    "Confidence",
    "Confidence_Bucket",
    "Bet_Line",
    "Final_Total",
    "Source_File",
]


def _parse_date_from_filename(csv_path: Path) -> Optional[str]:
    """YYYY-MM-DD from predictions_YYYYMMDD_HHMM.csv."""
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", csv_path.name, re.IGNORECASE)
    if not m:
        return None
    y = m.group(1)
    return f"{y[:4]}-{y[4:6]}-{y[6:8]}"


def _year_from_date(date_yyyy_mm_dd: str) -> int:
    return int(date_yyyy_mm_dd[:4])


def _is_graded_result(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ("WIN", "LOSS", "PUSH")


def _cell(row: dict[str, str], key: str) -> str:
    if key not in row or row[key] is None:
        return ""
    return str(row[key]).strip()


def _ou_pick(row: dict[str, str]) -> str:
    pred = _cell(row, "Prediction")
    if pred:
        return pred
    side = _cell(row, "Side")
    line = _cell(row, "Bet_Line")
    if side and line:
        return f"{side} {line}".strip()
    if side:
        return side
    return line


def _ou_confidence(row: dict[str, str]) -> str:
    c = _cell(row, "Confidence")
    if c:
        return c
    cv = _cell(row, "Confidence_Value")
    if cv:
        try:
            v = float(cv)
            if 0 <= v <= 1:
                return f"{v * 100:.0f}%"
            return f"{v:.0f}%"
        except ValueError:
            return cv
    return ""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig strips BOM so exports from Excel/PowerShell still match column names
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(r) for r in reader]


def _load_tracker_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return _read_csv_rows(path)


def _dedupe_key(source_file: str, game: str, bet_type: str) -> tuple[str, str, str]:
    return (source_file.strip(), game.strip(), bet_type.strip())


def _existing_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for r in rows:
        sf = _cell(r, "Source_File")
        g = _cell(r, "Game")
        bt = _cell(r, "Bet_Type")
        if sf and g and bt:
            out.add(_dedupe_key(sf, g, bt))
    return out


def _write_tracker(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACKER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in TRACKER_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append graded bets from a predictions CSV into tracking/season_results_YYYY.csv."
    )
    parser.add_argument("predictions_csv", type=Path, help="Graded predictions CSV path")
    parser.add_argument(
        "--date",
        help="Slate date YYYY-MM-DD (default: from predictions_YYYYMMDD_HHMM filename)",
    )
    parser.add_argument(
        "--season",
        type=int,
        metavar="YEAR",
        help="Season year for tracker file (default: from --date or filename)",
    )
    args = parser.parse_args()
    src: Path = args.predictions_csv.resolve()

    if not src.is_file():
        print(f"File not found: {src}", file=sys.stderr)
        return 1

    if args.date:
        slate_date = args.date.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", slate_date):
            print("--date must be YYYY-MM-DD", file=sys.stderr)
            return 1
    else:
        slate_date = _parse_date_from_filename(src)
        if not slate_date:
            print(
                "Could not infer date from filename; use --date YYYY-MM-DD",
                file=sys.stderr,
            )
            return 1

    year = args.season if args.season is not None else _year_from_date(slate_date)
    tracker_path = TRACKING_DIR / f"season_results_{year}.csv"
    source_basename = src.name

    try:
        pred_rows = _read_csv_rows(src)
    except Exception as e:
        print(f"Failed to read predictions CSV: {e}", file=sys.stderr)
        return 1

    existing = _load_tracker_rows(tracker_path)
    keys = _existing_keys(existing)
    new_rows: list[dict[str, str]] = []

    for row in pred_rows:
        game = _cell(row, "Game")
        if not game:
            continue

        if _is_graded_result(row.get("OU_Result")):
            k = _dedupe_key(source_basename, game, "O/U")
            if k in keys:
                continue
            keys.add(k)
            new_rows.append(
                {
                    "Date": slate_date,
                    "Game": game,
                    "Bet_Type": "O/U",
                    "Pick": _ou_pick(row),
                    "Result": _cell(row, "OU_Result").upper(),
                    "Confidence": _ou_confidence(row),
                    "Confidence_Bucket": _cell(row, "OU_Confidence_Bucket"),
                    "Bet_Line": _cell(row, "Bet_Line"),
                    "Final_Total": _cell(row, "Final_Total"),
                    "Source_File": source_basename,
                }
            )

        if _is_graded_result(row.get("ML_Result")):
            k = _dedupe_key(source_basename, game, "ML")
            if k in keys:
                continue
            keys.add(k)
            new_rows.append(
                {
                    "Date": slate_date,
                    "Game": game,
                    "Bet_Type": "ML",
                    "Pick": _cell(row, "ML_Pick"),
                    "Result": _cell(row, "ML_Result").upper(),
                    "Confidence": _cell(row, "ML_Confidence"),
                    "Confidence_Bucket": _cell(row, "ML_Confidence_Bucket"),
                    "Bet_Line": "",
                    "Final_Total": _cell(row, "Final_Total"),
                    "Source_File": source_basename,
                }
            )

    if not new_rows:
        print(f"No new graded bet rows to append (tracker: {tracker_path})")
        return 0

    merged = existing + new_rows
    try:
        _write_tracker(tracker_path, merged)
    except Exception as e:
        print(f"Failed to write tracker: {e}", file=sys.stderr)
        return 1

    print(f"Appended {len(new_rows)} row(s) -> {tracker_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a graded archive/predictions_*.csv and write a narrow client-facing CSV:
one row per graded bet (O/U and/or ML), long form.
Does not modify the source file.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

OUT_COLUMNS = [
    "Game",
    "Market",
    "Pick",
    "Result",
    "Confidence",
    "Bet_Line",
    "Closing_Line",
    "CLV",
    "CLV_Result",
]


def _is_graded(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ("WIN", "LOSS", "PUSH")


def _cell(row: Dict[str, str], key: str) -> str:
    if key not in row or row[key] is None:
        return ""
    return str(row[key]).strip()


def _ou_pick(row: Dict[str, str]) -> str:
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


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(r) for r in reader]


def _build_output_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        game = _cell(row, "Game")
        if not game:
            continue

        if _is_graded(row.get("OU_Result")):
            out.append(
                {
                    "Game": game,
                    "Market": "O/U",
                    "Pick": _ou_pick(row),
                    "Result": _cell(row, "OU_Result").upper(),
                    "Confidence": _cell(row, "Confidence"),
                    "Bet_Line": _cell(row, "Bet_Line"),
                    "Closing_Line": _cell(row, "Closing_Line"),
                    "CLV": _cell(row, "CLV"),
                    "CLV_Result": _cell(row, "CLV_Result"),
                }
            )

        if _is_graded(row.get("ML_Result")):
            out.append(
                {
                    "Game": game,
                    "Market": "ML",
                    "Pick": _cell(row, "ML_Pick"),
                    "Result": _cell(row, "ML_Result").upper(),
                    "Confidence": _cell(row, "ML_Confidence"),
                    "Bet_Line": "",
                    "Closing_Line": "",
                    "CLV": "",
                    "CLV_Result": "",
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export client-facing graded report (long form) from archive CSV."
    )
    parser.add_argument(
        "archive_csv",
        type=Path,
        help="Path to graded archive/predictions_*.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same directory as input, client_<basename>)",
    )
    args = parser.parse_args()
    src = args.archive_csv.resolve()
    if not src.is_file():
        print(f"File not found: {src}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        out_path = src.parent / f"client_{src.name}"
    else:
        out_path = out_path.resolve()

    try:
        rows = _read_rows(src)
    except Exception as e:
        print(f"Failed to read archive: {e}", file=sys.stderr)
        return 1

    built = _build_output_rows(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in built:
                w.writerow(r)
    except Exception as e:
        print(f"Failed to write output: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {len(built)} row(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

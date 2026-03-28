#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only summary of the dedicated season ledger (tracking/season_results_YYYY.csv).

Separate O/U and ML totals, win rate excluding pushes, optional combined block.

Usage:
  python scripts/season_tracker_summary.py
  python scripts/season_tracker_summary.py --season 2026
  python scripts/season_tracker_summary.py --file tracking/season_results_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "tracking"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(r) for r in reader]


def _cell(row: dict[str, str], key: str) -> str:
    if key not in row or row[key] is None:
        return ""
    return str(row[key]).strip()


def _canonical_bet_type(raw: str) -> Optional[str]:
    """Map ledger Bet_Type to O/U or ML."""
    b = (raw or "").strip().upper()
    if b in ("O/U", "OU"):
        return "O/U"
    if b == "ML":
        return "ML"
    return None


def _canonical_result(raw: str) -> Optional[str]:
    s = (raw or "").strip().upper()
    if s in ("WIN", "LOSS", "PUSH"):
        return s
    return None


def _tally_for_bet_type(
    rows: list[dict[str, str]], want: str
) -> tuple[int, int, int, int]:
    """Returns wins, losses, pushes, skipped (unknown type or result)."""
    w = l = p = skipped = 0
    for row in rows:
        if _canonical_bet_type(_cell(row, "Bet_Type")) != want:
            continue
        res = _canonical_result(_cell(row, "Result"))
        if res == "WIN":
            w += 1
        elif res == "LOSS":
            l += 1
        elif res == "PUSH":
            p += 1
        else:
            skipped += 1
    return w, l, p, skipped


def _tally_combined(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
    w = l = p = skipped = 0
    for row in rows:
        if _canonical_bet_type(_cell(row, "Bet_Type")) not in ("O/U", "ML"):
            continue
        res = _canonical_result(_cell(row, "Result"))
        if res == "WIN":
            w += 1
        elif res == "LOSS":
            l += 1
        elif res == "PUSH":
            p += 1
        else:
            skipped += 1
    return w, l, p, skipped


def _win_rate(wins: int, losses: int) -> Optional[float]:
    d = wins + losses
    if d == 0:
        return None
    return wins / d


def _print_block(title: str, wins: int, losses: int, pushes: int, skipped: int) -> None:
    print()
    print(f"--- {title} ---")
    print(f"  WIN:   {wins}")
    print(f"  LOSS:  {losses}")
    print(f"  PUSH:  {pushes}")
    wr = _win_rate(wins, losses)
    if wr is None:
        print("  Win rate (excl. push): n/a (no wins or losses)")
    else:
        print(f"  Win rate (excl. push): {wr * 100:.1f}%")
    if skipped:
        print(f"  (Rows skipped — missing/unknown Result: {skipped})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Season summary from tracking/season_results_YYYY.csv (official ledger)."
    )
    parser.add_argument(
        "--season",
        type=int,
        metavar="YEAR",
        help=f"Season year (default: {datetime.now().year}); uses tracking/season_results_YEAR.csv",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Explicit path to season_results CSV (overrides --season)",
    )
    args = parser.parse_args()

    if args.file is not None:
        ledger_path = args.file.resolve()
        m = re.match(r"season_results_(\d{4})$", ledger_path.stem, re.IGNORECASE)
        season_label = m.group(1) if m else ledger_path.stem
    else:
        year = args.season if args.season is not None else datetime.now().year
        ledger_path = (TRACKING_DIR / f"season_results_{year}.csv").resolve()
        season_label = str(year)

    if not ledger_path.is_file():
        print(f"Ledger not found: {ledger_path}", file=sys.stderr)
        print(
            "Create it with scripts/append_season_tracker.py or pass --file to an existing CSV.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = _read_rows(ledger_path)
    except Exception as e:
        print(f"Failed to read ledger: {e}", file=sys.stderr)
        return 1

    ou_w, ou_l, ou_p, ou_sk = _tally_for_bet_type(rows, "O/U")
    ml_w, ml_l, ml_p, ml_sk = _tally_for_bet_type(rows, "ML")
    cb_w, cb_l, cb_p, cb_sk = _tally_combined(rows)

    print("==============================")
    print(f"SEASON TRACKER SUMMARY ({season_label})")
    print("==============================")
    print(f"Ledger: {ledger_path}")
    print(f"Total rows in file: {len(rows)}")

    _print_block("O/U", ou_w, ou_l, ou_p, ou_sk)
    _print_block("ML", ml_w, ml_l, ml_p, ml_sk)
    _print_block("Combined (O/U + ML)", cb_w, cb_l, cb_p, cb_sk)

    return 0


if __name__ == "__main__":
    sys.exit(main())

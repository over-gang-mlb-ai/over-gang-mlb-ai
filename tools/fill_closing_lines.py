#!/usr/bin/env python3
"""
Fills Closing_Line in an Over Gang predictions CSV using SportsDataIO odds for the same date.
Does not compute CLV; use update_clv.py after filling closing lines if needed.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ["Game", "Bet_Line", "Closing_Line"]


def _ensure_core_on_path() -> None:
    """Ensure project root is on sys.path so core.sportsdataio can be imported."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_date_from_filename(csv_path: str) -> Optional[str]:
    """
    Extract slate date from filename like predictions_YYYYMMDD_HHMM.csv.
    Returns YYYY-MM-DD or None if pattern does not match.
    """
    name = Path(csv_path).name
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", name, re.IGNORECASE)
    if not m:
        return None
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def game_to_odds_key(game_str: str) -> Optional[str]:
    """Convert Game column value to the key format used by odds_map (normalized 'away @ home')."""
    if not game_str or not isinstance(game_str, str):
        return None
    parts = game_str.strip().split(" @ ", 1)
    if len(parts) != 2:
        return None
    try:
        from core.public_betting_loader import normalize_team_name
        return f"{normalize_team_name(parts[0].strip())} @ {normalize_team_name(parts[1].strip())}"
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill Closing_Line in a predictions CSV using SportsDataIO odds."
    )
    parser.add_argument("csv_path", help="Path to the predictions CSV file")
    args = parser.parse_args()
    csv_path = args.csv_path

    _ensure_core_on_path()

    path = Path(csv_path)
    if not path.is_file():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    target_date = parse_date_from_filename(csv_path)
    if not target_date:
        print("Error: Could not parse date from filename (expected predictions_YYYYMMDD_HHMM.csv)", file=sys.stderr)
        sys.exit(1)

    try:
        from core.sportsdataio import fetch_mlb_odds_by_date
        odds_map = fetch_mlb_odds_by_date(target_date)
    except Exception as e:
        print(f"Error fetching SportsDataIO odds: {e}", file=sys.stderr)
        sys.exit(1)

    filled = 0
    for i in range(len(df)):
        game_val = df.at[i, "Game"]
        key = game_to_odds_key(game_val)
        if not key or key not in odds_map:
            continue
        row_odds = odds_map[key]
        total_line = row_odds.get("total_line")
        try:
            if total_line is not None and total_line != "":
                val = float(total_line)
                df.at[i, "Closing_Line"] = val
                filled += 1
        except (TypeError, ValueError):
            continue

    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Filled Closing_Line for {filled} rows in {csv_path}")


if __name__ == "__main__":
    main()

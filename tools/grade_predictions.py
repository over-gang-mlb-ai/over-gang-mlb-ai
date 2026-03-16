#!/usr/bin/env python3
"""
Grades finished Over Gang totals bets in a predictions CSV.
Only grades rows where Final_Total already has a numeric value.
"""
import argparse
import sys

import pandas as pd


REQUIRED_COLUMNS = ["Game", "Side", "Bet_Line", "Units"]
OUTPUT_COLUMNS = ["Final_Total", "Bet_Result", "Units_Won", "ROI"]


def _ensure_columns(df: pd.DataFrame) -> None:
    """Add OUTPUT_COLUMNS to df if they do not exist."""
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""


def _grade_row(final_total: float, bet_line: float, side: str, units: float) -> tuple:
    """
    Return (Bet_Result, Units_Won) for one row.
    side: "over" or "under"; units: positive number.
    """
    side = (side or "").strip().lower()
    if side not in ("over", "under"):
        return ("", None)
    try:
        ft = float(final_total)
        bl = float(bet_line)
        u = float(units) if units is not None and units != "" else 0.0
        u = max(0.0, u)
    except (TypeError, ValueError):
        return ("", None)
    if side == "over":
        if ft > bl:
            return ("win", u)
        if ft < bl:
            return ("loss", -u)
        return ("push", 0.0)
    # under
    if ft < bl:
        return ("win", u)
    if ft > bl:
        return ("loss", -u)
    return ("push", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade finished totals bets in a predictions CSV."
    )
    parser.add_argument("csv_path", help="Path to the predictions CSV file")
    args = parser.parse_args()
    csv_path = args.csv_path

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    _ensure_columns(df)

    graded = 0
    for i in range(len(df)):
        final_total = df.at[i, "Final_Total"]
        try:
            if final_total is None or final_total == "" or (isinstance(final_total, str) and not final_total.strip()):
                continue
            ft = float(final_total)
        except (TypeError, ValueError):
            continue
        bet_line = df.at[i, "Bet_Line"]
        side = df.at[i, "Side"]
        units = df.at[i, "Units"]
        result, units_won = _grade_row(ft, bet_line, side, units)
        if result == "":
            continue
        df.at[i, "Bet_Result"] = result
        df.at[i, "Units_Won"] = units_won
        df.at[i, "ROI"] = units_won
        graded += 1

    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Graded {graded} rows in {csv_path}")


if __name__ == "__main__":
    main()

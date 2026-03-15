#!/usr/bin/env python3
"""
CLV updater for Over Gang predictions CSV.
Updates Bet_Line, Closing_Line, CLV, and CLV_Result in place.
"""
import argparse
import sys

import pandas as pd


REQUIRED_COLUMNS = ["Bet_Line", "Closing_Line", "CLV", "CLV_Result", "Side"]


def compute_clv(bet_line: float, closing_line: float, side: str) -> float:
    """
    Compute Closing Line Value.
    - Over: CLV = Closing_Line - Bet_Line (higher close favors over)
    - Under: CLV = Bet_Line - Closing_Line (lower close favors under)
    """
    if not side or side not in ("over", "under"):
        return float("nan")
    if side == "over":
        return float(closing_line - bet_line)
    return float(bet_line - closing_line)


def clv_result(clv: float) -> str:
    """Return 'positive', 'neutral', or 'negative' from CLV value."""
    if pd.isna(clv):
        return ""
    if clv > 0:
        return "positive"
    if clv < 0:
        return "negative"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update CLV fields in a predictions CSV.")
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

    # Coerce to numeric; non-numeric become NaN
    df["Bet_Line"] = pd.to_numeric(df["Bet_Line"], errors="coerce")
    df["Closing_Line"] = pd.to_numeric(df["Closing_Line"], errors="coerce")

    updated = 0
    for i in range(len(df)):
        bet = df.at[i, "Bet_Line"]
        close = df.at[i, "Closing_Line"]
        side = (df.at[i, "Side"] or "").strip().lower()
        if pd.notna(bet) and pd.notna(close) and side in ("over", "under"):
            clv_val = compute_clv(float(bet), float(close), side)
            df.at[i, "CLV"] = clv_val if not pd.isna(clv_val) else ""
            df.at[i, "CLV_Result"] = clv_result(clv_val)
            updated += 1

    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Updated CLV for {updated} rows in {csv_path}")


if __name__ == "__main__":
    main()

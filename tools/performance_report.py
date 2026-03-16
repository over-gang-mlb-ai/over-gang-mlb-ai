#!/usr/bin/env python3
"""
Generate a simple performance summary from an Over Gang predictions CSV.
Read-only: does not modify the CSV.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ["Bet_Result", "Units_Won", "Edge", "CLV"]


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce column to numeric; invalid/blank become NaN."""
    return pd.to_numeric(df[col], errors="coerce")


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute performance metrics. Uses only rows where Units_Won is not blank (numeric).
    Returns dict with Total_Bets, Wins, Losses, Pushes, Win_Rate, Units_Won_Total,
    ROI_Percent, Average_Edge, Average_CLV.
    """
    units = _numeric_series(df, "Units_Won")
    mask = units.notna()
    graded = df.loc[mask].copy()
    if graded.empty:
        return {
            "Total_Bets": 0,
            "Wins": 0,
            "Losses": 0,
            "Pushes": 0,
            "Win_Rate": 0.0,
            "Units_Won_Total": 0.0,
            "ROI_Percent": 0.0,
            "Average_Edge": 0.0,
            "Average_CLV": 0.0,
        }
    total_bets = len(graded)
    result = graded["Bet_Result"].fillna("").astype(str).str.strip().str.lower()
    wins = int((result == "win").sum())
    losses = int((result == "loss").sum())
    pushes = int((result == "push").sum())
    units_total = float(graded["Units_Won"].astype(float).sum())
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    roi = (units_total / total_bets * 100) if total_bets > 0 else 0.0
    edge_vals = _numeric_series(graded, "Edge")
    clv_vals = _numeric_series(graded, "CLV")
    avg_edge = float(edge_vals.mean()) if edge_vals.notna().any() else 0.0
    avg_clv = float(clv_vals.mean()) if clv_vals.notna().any() else 0.0
    return {
        "Total_Bets": total_bets,
        "Wins": wins,
        "Losses": losses,
        "Pushes": pushes,
        "Win_Rate": win_rate,
        "Units_Won_Total": units_total,
        "ROI_Percent": roi,
        "Average_Edge": avg_edge,
        "Average_CLV": avg_clv,
    }


def report_as_string(metrics: dict) -> str:
    """Return the performance summary as a string."""
    lines = [
        "==============================",
        "OVER GANG PERFORMANCE REPORT",
        "==============================",
        "",
        f"Total Bets: {metrics['Total_Bets']}",
        f"Wins: {metrics['Wins']}",
        f"Losses: {metrics['Losses']}",
        f"Pushes: {metrics['Pushes']}",
        f"Win Rate: {metrics['Win_Rate'] * 100:.0f}%",
        "",
        f"Units Won: {metrics['Units_Won_Total']:.1f}",
        f"ROI: {metrics['ROI_Percent']:.1f}%",
        "",
        f"Average Edge: {metrics['Average_Edge']:.1f}",
        f"Average CLV: {metrics['Average_CLV']:.1f}",
    ]
    return "\n".join(lines)


def print_report(metrics: dict) -> None:
    """Print the performance summary to stdout."""
    print(report_as_string(metrics))


def _date_from_csv_filename(csv_path: str) -> Optional[str]:
    """Extract YYYYMMDD from filename like predictions_YYYYMMDD_HHMM.csv. None if no match."""
    name = Path(csv_path).name
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", name, re.IGNORECASE)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a performance summary from an Over Gang predictions CSV."
    )
    parser.add_argument("csv_path", help="Path to the predictions CSV file")
    parser.add_argument("--save", action="store_true", help="Also save report to reports/daily_performance_YYYYMMDD.txt")
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

    metrics = compute_metrics(df)
    print_report(metrics)

    if args.save:
        yyyymmdd = _date_from_csv_filename(csv_path)
        filename = f"daily_performance_{yyyymmdd}.txt" if yyyymmdd else "daily_performance_unknown_date.txt"
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / filename
        out_path.write_text(report_as_string(metrics), encoding="utf-8")
        print(f"\nSaved report to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

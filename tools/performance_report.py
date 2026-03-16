#!/usr/bin/env python3
"""
Generate a simple performance summary from an Over Gang predictions CSV.
Read-only: does not modify the CSV.
"""
import argparse
import sys

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


def print_report(metrics: dict) -> None:
    """Print the performance summary to stdout."""
    print("==============================")
    print("OVER GANG PERFORMANCE REPORT")
    print("==============================")
    print()
    print(f"Total Bets: {metrics['Total_Bets']}")
    print(f"Wins: {metrics['Wins']}")
    print(f"Losses: {metrics['Losses']}")
    print(f"Pushes: {metrics['Pushes']}")
    print(f"Win Rate: {metrics['Win_Rate'] * 100:.0f}%")
    print()
    print(f"Units Won: {metrics['Units_Won_Total']:.1f}")
    print(f"ROI: {metrics['ROI_Percent']:.1f}%")
    print()
    print(f"Average Edge: {metrics['Average_Edge']:.1f}")
    print(f"Average CLV: {metrics['Average_CLV']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a performance summary from an Over Gang predictions CSV."
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

    metrics = compute_metrics(df)
    print_report(metrics)


if __name__ == "__main__":
    main()

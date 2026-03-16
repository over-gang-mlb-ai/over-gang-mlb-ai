#!/usr/bin/env python3
"""
Generate an aggregate Over Gang performance report across all archived predictions CSVs.
Read-only: does not modify any CSV.
"""
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Bet_Result", "Units_Won", "Edge", "CLV"]


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce column to numeric; invalid/blank become NaN."""
    return pd.to_numeric(df[col], errors="coerce")


def _load_valid_frames(archive_dir: Path) -> tuple[list[pd.DataFrame], int]:
    """
    Find all archive/predictions_*.csv, read each, keep only those with REQUIRED_COLUMNS.
    Returns (list of DataFrames that have graded rows, files_processed_count).
    """
    files = sorted(archive_dir.glob("predictions_*.csv"))
    frames = []
    files_processed = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not all(c in df.columns for c in REQUIRED_COLUMNS):
            continue
        files_processed += 1
        units = _numeric_series(df, "Units_Won")
        graded = df.loc[units.notna()].copy()
        if not graded.empty:
            frames.append(graded)
    return frames, files_processed


def compute_aggregate_metrics(frames: list[pd.DataFrame], files_processed: int) -> dict:
    """
    Aggregate metrics across all provided DataFrames (each already filtered to numeric Units_Won).
    """
    if not frames:
        return {
            "Files_Processed": files_processed,
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
    combined = pd.concat(frames, ignore_index=True)
    total_bets = len(combined)
    result = combined["Bet_Result"].fillna("").astype(str).str.strip().str.lower()
    wins = int((result == "win").sum())
    losses = int((result == "loss").sum())
    pushes = int((result == "push").sum())
    units_total = float(combined["Units_Won"].astype(float).sum())
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    roi = (units_total / total_bets * 100) if total_bets > 0 else 0.0
    edge_vals = _numeric_series(combined, "Edge")
    clv_vals = _numeric_series(combined, "CLV")
    avg_edge = float(edge_vals.mean()) if edge_vals.notna().any() else 0.0
    avg_clv = float(clv_vals.mean()) if clv_vals.notna().any() else 0.0
    return {
        "Files_Processed": files_processed,
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
    """Print the season summary to stdout."""
    print("===========================")
    print("OVER GANG SEASON REPORT")
    print("===========================")
    print()
    print(f"Files Processed: {metrics['Files_Processed']}")
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
    # Run from project root; archive is archive/
    root = Path(__file__).resolve().parent.parent
    archive_dir = root / "archive"
    if not archive_dir.is_dir():
        print("Error: archive/ directory not found.", file=sys.stderr)
        sys.exit(1)

    frames, files_processed = _load_valid_frames(archive_dir)
    metrics = compute_aggregate_metrics(frames, files_processed)
    print_report(metrics)


if __name__ == "__main__":
    main()

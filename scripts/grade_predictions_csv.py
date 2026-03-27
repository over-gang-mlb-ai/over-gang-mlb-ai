#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manually set OU_Result and/or ML_Result on rows in an exported predictions CSV.

Valid values: WIN, LOSS, PUSH, PENDING. All other columns are preserved.

Usage:
  python scripts/grade_predictions_csv.py archive/predictions_x.csv \\
    --game "New York Yankees @ San Francisco Giants" --ou WIN --ml LOSS
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

VALID_RESULTS = {"WIN", "LOSS", "PUSH", "PENDING"}


def _write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set OU_Result and/or ML_Result for rows matching --game (exact Game column)."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Predictions CSV to update in place.",
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Exact Game cell value (same string as in the CSV Game column).",
    )
    parser.add_argument(
        "--ou",
        choices=sorted(VALID_RESULTS),
        default=None,
        metavar="RESULT",
        help="Outcome for O/U (OU_Result).",
    )
    parser.add_argument(
        "--ml",
        choices=sorted(VALID_RESULTS),
        default=None,
        metavar="RESULT",
        help="Outcome for moneyline (ML_Result).",
    )
    args = parser.parse_args()

    if args.ou is None and args.ml is None:
        parser.error("Provide at least one of --ou or --ml.")

    path = args.csv_path
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    if "Game" not in df.columns:
        print("CSV missing required column: Game", file=sys.stderr)
        return 1
    if args.ou is not None and "OU_Result" not in df.columns:
        print("CSV missing required column: OU_Result", file=sys.stderr)
        return 1
    if args.ml is not None and "ML_Result" not in df.columns:
        print("CSV missing required column: ML_Result", file=sys.stderr)
        return 1

    key = args.game.strip()
    mask = df["Game"].astype(str).str.strip() == key
    idx = df.index[mask]
    if len(idx) == 0:
        print(f"No row with Game == {key!r}", file=sys.stderr)
        return 1

    print(f"Updating {len(idx)} row(s) matching Game == {key!r}\n")

    for i in idx:
        ou_before = df.at[i, "OU_Result"] if "OU_Result" in df.columns else None
        ml_before = df.at[i, "ML_Result"] if "ML_Result" in df.columns else None
        ou_after = args.ou if args.ou is not None else ou_before
        ml_after = args.ml if args.ml is not None else ml_before
        print(f"  row {i}:")
        print(f"    OU_Result: {ou_before!r} -> {ou_after!r}")
        print(f"    ML_Result: {ml_before!r} -> {ml_after!r}")
        if args.ou is not None:
            df.at[i, "OU_Result"] = args.ou
        if args.ml is not None:
            df.at[i, "ML_Result"] = args.ml

    _write_csv_atomic(path, df)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

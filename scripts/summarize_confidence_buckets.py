#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize WIN / LOSS / PUSH / PENDING counts by OU_Confidence_Bucket and ML_Confidence_Bucket
from an exported predictions CSV. Does not modify the file.

Usage:
  python scripts/summarize_confidence_buckets.py path/to/predictions_YYYYMMDD_HHMM.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _norm_result(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "BLANK"
    s = str(val).strip()
    if not s:
        return "BLANK"
    u = s.upper()
    if u in ("WIN", "LOSS", "PUSH", "PENDING"):
        return u
    return "OTHER"


def _nonblank_bucket(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return series.notna() & (s != "") & (s != "nan")


def _bucket_sort_key(label) -> tuple:
    s = str(label).strip()
    if "-" in s:
        try:
            lo = int(s.split("-", 1)[0])
            return (0, lo)
        except ValueError:
            pass
    return (1, s)


def _print_section(title: str, df: pd.DataFrame, bucket_col: str, result_col: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    if bucket_col not in df.columns:
        print(f"  Missing column: {bucket_col}")
        return
    if result_col not in df.columns:
        print(f"  Missing column: {result_col}")
        return

    sub = df.loc[_nonblank_bucket(df[bucket_col])].copy()
    if sub.empty:
        print(f"  No rows with non-blank {bucket_col}.")
        return

    sub["_out"] = sub[result_col].map(_norm_result)
    buckets = sorted(sub[bucket_col].unique(), key=_bucket_sort_key)

    header = (
        f"{'bucket':<12} {'n':>5} {'WIN':>5} {'LOSS':>5} {'PUSH':>5} "
        f"{'PEND':>5} {'BLANK':>5} {'OTHER':>5} {'hit%*':>8}"
    )
    print(header)
    print("-" * len(header))

    for b in buckets:
        grp = sub[sub[bucket_col] == b]
        n = len(grp)
        vc = grp["_out"].value_counts()
        w = int(vc.get("WIN", 0))
        l = int(vc.get("LOSS", 0))
        p = int(vc.get("PUSH", 0))
        pend = int(vc.get("PENDING", 0))
        blank = int(vc.get("BLANK", 0))
        oth = int(vc.get("OTHER", 0))
        decided = w + l
        hit_pct = (100.0 * w / decided) if decided > 0 else float("nan")
        hit_str = f"{hit_pct:.1f}%" if decided > 0 else "n/a"
        print(
            f"{str(b):<12} {n:5d} {w:5d} {l:5d} {p:5d} {pend:5d} {blank:5d} {oth:5d} {hit_str:>8}"
        )

    print(
        "\n  * hit% = WIN / (WIN + LOSS) only; PUSH, PENDING, BLANK, OTHER excluded from denominator."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize outcomes by OU_Confidence_Bucket and ML_Confidence_Bucket."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Predictions export CSV (e.g. data/archive/predictions_*.csv)",
    )
    args = parser.parse_args()
    path: Path = args.csv_path
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} row(s) from {path}")

    _print_section("O/U by OU_Confidence_Bucket (OU_Result)", df, "OU_Confidence_Bucket", "OU_Result")
    _print_section("ML by ML_Confidence_Bucket (ML_Result)", df, "ML_Confidence_Bucket", "ML_Result")

    return 0


if __name__ == "__main__":
    sys.exit(main())

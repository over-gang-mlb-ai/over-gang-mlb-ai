#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read archived predictions CSVs and print cohort-based validation summaries using
the current OU_Result / ML_Result grading path.

Default input is archive/predictions_*.csv. Optionally filter by season year from
the filename pattern predictions_YYYYMMDD_HHMM.csv.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _year_from_predictions_filename(path: Path) -> Optional[int]:
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)[:4])


def _collect_files(paths: list[Path], archive_dir: Path, season: Optional[int]) -> list[Path]:
    if paths:
        out: list[Path] = []
        for p in paths:
            rp = p.resolve()
            if rp.is_dir():
                out.extend(sorted(rp.glob("predictions_*.csv")))
            elif rp.is_file():
                out.append(rp)
        files = out
    else:
        files = sorted(archive_dir.resolve().glob("predictions_*.csv"))

    if season is None:
        return files
    return [f for f in files if _year_from_predictions_filename(f) == season]


def _load_frames(files: list[Path]) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    read_errors = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            read_errors += 1
            continue
        df["__source_file"] = f.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(), read_errors
    return pd.concat(frames, ignore_index=True, sort=False), read_errors


def _norm_result_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.upper()
    return s.where(s.isin(["WIN", "LOSS", "PUSH"]), "")


def _bool_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return txt.isin(["true", "1", "yes", "y"])


def _numeric_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _avg_clv_str(df: pd.DataFrame) -> str:
    clv = _numeric_series(df, "CLV")
    if clv is None:
        return "n/a"
    clv = clv.dropna()
    if clv.empty:
        return "n/a"
    return f"{clv.mean():.2f}"


def _result_summary(df: pd.DataFrame, result_col: str) -> tuple[int, int, int, int, str]:
    if result_col not in df.columns:
        return 0, 0, 0, 0, "n/a"
    res = _norm_result_series(df[result_col])
    wins = int((res == "WIN").sum())
    losses = int((res == "LOSS").sum())
    pushes = int((res == "PUSH").sum())
    rows = wins + losses + pushes
    decided = wins + losses
    hit = f"{(100.0 * wins / decided):.1f}%" if decided > 0 else "n/a"
    return rows, wins, losses, pushes, hit


def _print_table(title: str, headers: list[str], rows: Iterable[list[str]]) -> None:
    rows = list(rows)
    print(f"\n{title}")
    print("-" * len(title))
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    if not rows:
        print("(no rows)")


def _fired_vs_nonfired(df: pd.DataFrame, fired_col: str, result_col: str, label_prefix: str) -> None:
    fired = _bool_series(df, fired_col)
    if fired is None or result_col not in df.columns:
        print(f"\n{label_prefix} fired vs non-fired\n------------------------\n(required columns missing)")
        return
    rows = []
    for label, mask in [("fired", fired), ("not_fired", ~fired)]:
        sub = df.loc[mask]
        n, w, l, p, hit = _result_summary(sub, result_col)
        rows.append([label, str(n), str(w), str(l), str(p), hit])
    _print_table(
        f"{label_prefix} fired vs non-fired",
        ["cohort", "rows", "wins", "losses", "pushes", "hit_rate"],
        rows,
    )


def _bucket_summary(df: pd.DataFrame, bucket_col: str, result_col: str, title: str) -> None:
    if bucket_col not in df.columns or result_col not in df.columns:
        print(f"\n{title}\n{'-' * len(title)}\n(required columns missing)")
        return
    work = df.copy()
    work[bucket_col] = work[bucket_col].fillna("").astype(str).str.strip()
    work = work.loc[work[bucket_col] != ""]
    rows = []
    for bucket in sorted(work[bucket_col].unique(), key=lambda s: (999 if "-" not in s else int(s.split("-", 1)[0]), s)):
        sub = work.loc[work[bucket_col] == bucket]
        n, w, l, p, hit = _result_summary(sub, result_col)
        rows.append([bucket, str(n), str(w), str(l), str(p), hit, _avg_clv_str(sub)])
    _print_table(
        title,
        ["bucket", "rows", "wins", "losses", "pushes", "hit_rate", "avg_clv"],
        rows,
    )


def _dq_split(df: pd.DataFrame, result_col: str, title: str) -> None:
    if "Data_Quality_Flag" not in df.columns or result_col not in df.columns:
        print(f"\n{title}\n{'-' * len(title)}\n(required columns missing)")
        return
    dq = df["Data_Quality_Flag"].fillna("").astype(str).str.strip()
    rows = []
    for label, mask in [("clean", dq == ""), ("degraded", dq != "")]:
        sub = df.loc[mask]
        n, w, l, p, hit = _result_summary(sub, result_col)
        rows.append([label, str(n), str(w), str(l), str(p), hit, _avg_clv_str(sub)])
    _print_table(
        title,
        ["cohort", "rows", "wins", "losses", "pushes", "hit_rate", "avg_clv"],
        rows,
    )


def _edge_bin_label(v: float) -> str:
    av = abs(v)
    if av < 0.5:
        return "0.00-0.49"
    if av < 1.0:
        return "0.50-0.99"
    if av < 1.5:
        return "1.00-1.49"
    if av < 2.0:
        return "1.50-1.99"
    return "2.00+"


def _ou_edge_calibration(df: pd.DataFrame) -> None:
    edge = _numeric_series(df, "OU_Edge")
    if edge is None and "Edge" in df.columns:
        edge = _numeric_series(df, "Edge")
    if edge is None or "OU_Result" not in df.columns:
        print("\nO/U edge calibration\n--------------------\n(required columns missing)")
        return
    work = df.copy()
    work["_edge_bin"] = edge.map(lambda v: _edge_bin_label(float(v)) if pd.notna(v) else "")
    work = work.loc[work["_edge_bin"] != ""]
    order = ["0.00-0.49", "0.50-0.99", "1.00-1.49", "1.50-1.99", "2.00+"]
    rows = []
    for bucket in order:
        sub = work.loc[work["_edge_bin"] == bucket]
        if sub.empty:
            continue
        n, w, l, p, hit = _result_summary(sub, "OU_Result")
        rows.append([bucket, str(n), str(w), str(l), str(p), hit, _avg_clv_str(sub)])
    _print_table(
        "O/U edge calibration (abs Edge)",
        ["edge_bin", "rows", "wins", "losses", "pushes", "hit_rate", "avg_clv"],
        rows,
    )


def _optional_bool_cohort(df: pd.DataFrame, bool_col: str, result_col: str, title: str) -> None:
    b = _bool_series(df, bool_col)
    if b is None or result_col not in df.columns:
        return
    rows = []
    for label, mask in [("false", ~b), ("true", b)]:
        sub = df.loc[mask]
        n, w, l, p, hit = _result_summary(sub, result_col)
        rows.append([label, str(n), str(w), str(l), str(p), hit, _avg_clv_str(sub)])
    _print_table(
        title,
        [bool_col, "rows", "wins", "losses", "pushes", "hit_rate", "avg_clv"],
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Performance validation report from archived graded predictions CSVs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional predictions CSV files or directories. Default: archive/predictions_*.csv",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT / "archive",
        help="Archive directory when no paths are provided (default: <repo>/archive)",
    )
    parser.add_argument(
        "--season",
        type=int,
        metavar="YEAR",
        help="Optional year filter using predictions_YYYYMMDD_HHMM.csv filename pattern",
    )
    args = parser.parse_args()

    files = _collect_files(args.paths, args.archive_dir, args.season)
    if not files:
        print("No predictions CSV files found for the requested scope.", file=sys.stderr)
        return 1

    df, read_errors = _load_frames(files)
    if df.empty:
        print("No readable predictions CSVs found for the requested scope.", file=sys.stderr)
        return 1

    print("==============================")
    print("OVER GANG VALIDATION REPORT")
    print("==============================")
    print(f"Files matched: {len(files)}")
    print(f"Files read: {len(df['__source_file'].astype(str).unique())}")
    print(f"Rows loaded: {len(df)}")
    if read_errors:
        print(f"Files failed to read: {read_errors}")

    _fired_vs_nonfired(df, "OU_Fired", "OU_Result", "O/U")
    _fired_vs_nonfired(df, "ML_Fired", "ML_Result", "ML")
    _bucket_summary(df, "OU_Confidence_Bucket", "OU_Result", "O/U by OU_Confidence_Bucket")
    _bucket_summary(df, "ML_Confidence_Bucket", "ML_Result", "ML by ML_Confidence_Bucket")
    _dq_split(df, "OU_Result", "O/U degraded vs clean (Data_Quality_Flag)")
    _dq_split(df, "ML_Result", "ML degraded vs clean (Data_Quality_Flag)")
    _ou_edge_calibration(df)
    _optional_bool_cohort(df, "Total_Is_Real", "OU_Result", "O/U by Total_Is_Real")
    _optional_bool_cohort(df, "Projection_Cap_Flag", "OU_Result", "O/U by Projection_Cap_Flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())

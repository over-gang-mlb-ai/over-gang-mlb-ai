#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline O/U empirical calibration: map signed model edge (runs) to outcome frequencies.

Reads graded archive predictions CSVs (OU_Result WIN/LOSS/PUSH), bins by signed edge,
applies simple Dirichlet smoothing, writes JSON artifact + console summary.

Stratifies diagnostics by Odds_Book and Total_Line_Source when present; flags thin
cohorts/bins (configurable thresholds). Exploratory / offline only.

Does not modify production prediction logic, fire gates, or odds loading.

Usage:
  python scripts/calibrate_ou.py
  python scripts/calibrate_ou.py archive/
  python scripts/calibrate_ou.py archive/predictions_20260423_1318.csv
  python scripts/calibrate_ou.py --glob 'archive/predictions_2026*.csv' \\
      --output calibration/ou_edge_calibration.json
  python scripts/calibrate_ou.py --min-bin-rows 5 --min-cohort-rows 20
  python scripts/calibrate_ou.py --book-filter Pinnacle --source-filter parlay_api
"""
from __future__ import annotations

import argparse
import glob as glob_std
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Signed-edge bins: [lo, hi) except first/last which are (-inf, hi) and [lo, inf)
EDGE_BIN_EDGES: List[float] = [
    -math.inf,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    math.inf,
]

DIRICHLET_ALPHA = 1.0  # Laplace / uniform prior per outcome (over / under / push)
PRIOR_STRENGTH = 10.0  # for sample_weight denominator (larger = need more games for full weight)

DEFAULT_MIN_BIN_ROWS = 5
DEFAULT_MIN_COHORT_ROWS = 15

# Production-readiness: bounded-edge "core" bins only (exclude (-inf,-2) and [2,inf)).
def _core_bin_indices() -> Tuple[int, ...]:
    n = len(EDGE_BIN_EDGES) - 1
    return tuple(range(1, max(1, n - 1)))


DEFAULT_MIN_PRODUCTION_TOTAL_ROWS = 120
DEFAULT_MIN_PRODUCTION_CORE_BIN_ROWS = 15


def _bin_label(lo: float, hi: float, idx: int, n_bins: int) -> str:
    if idx == 0:
        return f"(-inf,{hi})"
    if idx == n_bins - 1:
        return f"[{lo},inf)"
    return f"[{lo},{hi})"


def _signed_edge_bin(value: float) -> Tuple[int, str]:
    """Return (bin_index, label)."""
    n = len(EDGE_BIN_EDGES) - 1
    for i in range(n):
        lo, hi = EDGE_BIN_EDGES[i], EDGE_BIN_EDGES[i + 1]
        if i == 0:
            if value < hi:
                return i, _bin_label(lo, hi, i, n)
        elif i == n - 1:
            if value >= lo:
                return i, _bin_label(lo, hi, i, n)
        else:
            if lo <= value < hi:
                return i, _bin_label(lo, hi, i, n)
    return n - 1, _bin_label(EDGE_BIN_EDGES[-2], EDGE_BIN_EDGES[-1], n - 1, n)


def _parse_firm_ou_side(prediction: Any) -> Optional[str]:
    """
    Return 'over' or 'under' only for firm O/U picks (not LEAN, not NO BET).
    Archive Prediction examples: 'OVER 8.5', 'UNDER 7.0', 'LEAN OVER ...' (exclude).
    """
    if prediction is None or (isinstance(prediction, float) and pd.isna(prediction)):
        return None
    s = str(prediction).strip()
    if not s:
        return None
    u = s.upper()
    if u.startswith("LEAN") or u.startswith("NO BET"):
        return None
    if re.match(r"^OVER\b", u):
        return "over"
    if re.match(r"^UNDER\b", u):
        return "under"
    return None


def _norm_ou_result(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    t = str(val).strip().upper()
    if t in ("WIN", "LOSS", "PUSH"):
        return t
    return ""


def _infer_total_outcome(side: str, ou_result: str) -> Optional[str]:
    """
    Infer whether final total was over / under / push vs the bet line,
    from pick side and graded OU_Result (no final score column required).
    """
    if ou_result == "PUSH":
        return "push"
    if side == "over" and ou_result == "WIN":
        return "over"
    if side == "over" and ou_result == "LOSS":
        return "under"
    if side == "under" and ou_result == "WIN":
        return "under"
    if side == "under" and ou_result == "LOSS":
        return "over"
    return None


def _all_nonempty(series: pd.Series) -> bool:
    """True if every row has a non-missing, non-blank (for strings) value."""
    if not series.notna().all():
        return False
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return bool(series.astype(str).str.strip().ne("").all())
    return True


def _select_dedupe_subset(work: pd.DataFrame) -> Optional[List[str]]:
    """
    Prefer a key that distinguishes game instances (doubleheaders, reruns).
    Order: Game_ID → Game+Datetime → Game+Game_Date+Bet_Line → legacy Game(+Game_Date).
    """
    if "Game_ID" in work.columns and _all_nonempty(work["Game_ID"]):
        return ["Game_ID"]
    if (
        "Game" in work.columns
        and "Datetime" in work.columns
        and _all_nonempty(work["Game"])
        and _all_nonempty(work["Datetime"])
    ):
        return ["Game", "Datetime"]
    triple = ("Game", "Game_Date", "Bet_Line")
    if all(c in work.columns for c in triple) and all(
        _all_nonempty(work[c]) for c in triple
    ):
        return ["Game", "Game_Date", "Bet_Line"]
    if "Game" in work.columns:
        out: List[str] = ["Game"]
        if "Game_Date" in work.columns:
            out.append("Game_Date")
        return out
    return None


def _to_bool_real_total(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin(("true", "1", "yes", "y"))


def _market_line(row: pd.Series) -> Optional[float]:
    for col in ("Bet_Line", "Odds_Line", "Vegas_Line"):
        if col in row.index and pd.notna(row[col]):
            try:
                v = float(row[col])
                if math.isfinite(v):
                    return v
            except (TypeError, ValueError):
                continue
    return None


def _signed_edge_row(row: pd.Series) -> Optional[float]:
    if "OU_Edge" in row.index and pd.notna(row["OU_Edge"]):
        try:
            e = float(row["OU_Edge"])
            if math.isfinite(e):
                return e
        except (TypeError, ValueError):
            pass
    if "Edge" in row.index and pd.notna(row["Edge"]):
        try:
            e = float(row["Edge"])
            if math.isfinite(e):
                return e
        except (TypeError, ValueError):
            pass
    line = _market_line(row)
    if line is None:
        return None
    if "Projected_Total" not in row.index or pd.isna(row["Projected_Total"]):
        return None
    try:
        pt = float(row["Projected_Total"])
        if not math.isfinite(pt):
            return None
        return round(pt - line, 2)
    except (TypeError, ValueError):
        return None


def _expand_glob_pattern(pattern: str) -> List[Path]:
    """Expand a glob pattern; relative patterns resolve from repo ROOT."""
    p = Path(pattern)
    if p.is_absolute():
        return [Path(x) for x in sorted(glob_std.glob(str(p)))]
    return [Path(x) for x in sorted(glob_std.glob(str(ROOT / pattern)))]


def _collect_csv_paths(
    paths_arg: Optional[List[str]], glob_patterns: Optional[List[str]]
) -> List[Path]:
    paths: List[Path] = []
    if paths_arg:
        for p in paths_arg:
            rp = Path(p).resolve()
            if rp.is_dir():
                paths.extend(sorted(rp.glob("predictions_*.csv")))
            elif rp.is_file():
                paths.append(rp)
    if glob_patterns:
        for pattern in glob_patterns:
            paths.extend(_expand_glob_pattern(pattern))
    if not paths and not glob_patterns:
        arch = ROOT / "archive"
        if arch.is_dir():
            paths = sorted(arch.glob("predictions_*.csv"))
    seen = set()
    out: List[Path] = []
    for p in paths:
        r = p.resolve()
        if r not in seen and r.is_file():
            seen.add(r)
            out.append(r)
    return out


def load_frames(files: List[Path]) -> Tuple[pd.DataFrame, int, List[str]]:
    frames: List[pd.DataFrame] = []
    errors = 0
    names: List[str] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = f.name
            frames.append(df)
            names.append(str(f))
        except Exception:
            errors += 1
    if not frames:
        return pd.DataFrame(), errors, names
    return pd.concat(frames, ignore_index=True, sort=False), errors, names


def build_calibration_table(
    df: pd.DataFrame, dedupe: bool
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    required = {"Prediction", "OU_Result", "Projected_Total"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_ou_res"] = work["OU_Result"].map(_norm_ou_result)
    work = work.loc[work["_ou_res"].isin(["WIN", "LOSS", "PUSH"])]

    if "Total_Is_Real" in work.columns:
        work = work.loc[_to_bool_real_total(work["Total_Is_Real"])]

    work["_side"] = work["Prediction"].map(_parse_firm_ou_side)
    work = work.loc[work["_side"].isin(["over", "under"])]

    work["_edge"] = work.apply(_signed_edge_row, axis=1)
    work = work.loc[work["_edge"].notna()]

    work["_outcome"] = [
        _infer_total_outcome(s, r)
        for s, r in zip(work["_side"], work["_ou_res"])
    ]
    work = work.loc[pd.Series(work["_outcome"]).notna()]

    dedupe_subset: Optional[List[str]] = None
    if dedupe:
        dedupe_subset = _select_dedupe_subset(work)
        if dedupe_subset:
            work = work.drop_duplicates(subset=dedupe_subset, keep="last")

    work["_bin_idx"], work["_bin_label"] = zip(
        *[_signed_edge_bin(float(x)) for x in work["_edge"]]
    )
    return work, (dedupe_subset if dedupe else None)


def _stratify_series(work: pd.DataFrame, column: str) -> pd.Series:
    """Normalize a stratification column; missing column or blank → '(missing)'."""
    if column not in work.columns:
        return pd.Series(["(missing)"] * len(work), index=work.index, dtype=object)
    s = work[column].fillna("").astype(str).str.strip()
    s = s.replace("", "(missing)")
    return s


def _collect_filter_tokens(raw: Optional[List[str]]) -> List[str]:
    """Flatten repeatable CLI args; allow comma-separated tokens per flag."""
    if not raw:
        return []
    out: List[str] = []
    for chunk in raw:
        for part in str(chunk).split(","):
            t = part.strip()
            if t:
                out.append(t)
    return out


def _filter_match_set(tokens: List[str]) -> Set[str]:
    return {t.casefold() for t in tokens}


def apply_book_source_filters(
    work: pd.DataFrame,
    book_filters: List[str],
    source_filters: List[str],
) -> pd.DataFrame:
    """
    Restrict rows to Odds_Book / Total_Line_Source in filter lists (case-insensitive).
    Empty list for a dimension means no filter on that dimension.
    """
    out = work
    if book_filters:
        s = _stratify_series(out, "Odds_Book")
        allow = _filter_match_set(book_filters)
        out = out.loc[s.map(lambda x: str(x).casefold() in allow)]
    if source_filters:
        s = _stratify_series(out, "Total_Line_Source")
        allow = _filter_match_set(source_filters)
        out = out.loc[s.map(lambda x: str(x).casefold() in allow)]
    return out


def evaluate_production_readiness(
    bin_rows: List[Dict[str, Any]],
    rows_used: int,
    *,
    min_production_total_rows: int,
    min_production_core_bin_rows: int,
    diagnostic_min_bin_rows: int,
) -> Tuple[bool, List[str], bool]:
    """
    Declare production_ready when cohort size and core (bounded-edge) bin counts
    meet production floors. Also reports diagnostic-thin bins in core (games <
    diagnostic_min_bin_rows).

    Returns (production_ready, reasons, thin_bins_in_core_diagnostic).
    """
    reasons: List[str] = []
    core = _core_bin_indices()
    by_idx = {int(b["bin_index"]): b for b in bin_rows}

    if rows_used <= 0:
        reasons.append("fail: no rows in calibration sample after filters")
        return False, reasons, False

    ok_total = rows_used >= min_production_total_rows
    reasons.append(
        f"{'pass' if ok_total else 'fail'}: rows_used={rows_used} vs "
        f"min_production_total_rows={min_production_total_rows}"
    )

    thin_core_diag: List[str] = []
    short_core: List[str] = []
    for i in core:
        b = by_idx.get(i)
        if not b:
            short_core.append(f"bin_index={i} missing from aggregation")
            continue
        games = int(b.get("games", 0))
        label = str(b.get("label", i))
        if games < diagnostic_min_bin_rows:
            thin_core_diag.append(
                f"{label} (games={games} < diagnostic_min_bin_rows={diagnostic_min_bin_rows})"
            )
        if games < min_production_core_bin_rows:
            short_core.append(
                f"{label} (games={games} < min_production_core_bin_rows={min_production_core_bin_rows})"
            )

    thin_in_core = bool(thin_core_diag)
    if thin_in_core:
        reasons.append(
            "diagnostic: thin bin(s) in core bounded-edge range: "
            + "; ".join(thin_core_diag)
        )
    else:
        reasons.append(
            "pass: no diagnostic-thin bins in core range "
            f"(threshold={diagnostic_min_bin_rows})"
        )

    if short_core:
        reasons.append(
            "fail: production core-bin minimum not met: " + "; ".join(short_core)
        )
    else:
        reasons.append(
            f"pass: all core bins {core} meet min_production_core_bin_rows={min_production_core_bin_rows}"
        )

    production_ready = ok_total and not short_core
    return production_ready, reasons, thin_in_core


def aggregate_bins(
    work: pd.DataFrame, min_bin_rows: int = DEFAULT_MIN_BIN_ROWS
) -> List[Dict[str, Any]]:
    n_bins = len(EDGE_BIN_EDGES) - 1
    bin_defs: List[Dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = EDGE_BIN_EDGES[i], EDGE_BIN_EDGES[i + 1]
        bin_defs.append(
            {
                "index": i,
                "label": _bin_label(lo, hi, i, n_bins),
                "lo": None if math.isinf(lo) and lo < 0 else lo,
                "hi": None if math.isinf(hi) and hi > 0 else hi,
            }
        )

    rows: List[Dict[str, Any]] = []
    for i in range(n_bins):
        sub = work.loc[work["_bin_idx"] == i]
        games = int(len(sub))
        overs = int((sub["_outcome"] == "over").sum())
        unders = int((sub["_outcome"] == "under").sum())
        pushes = int((sub["_outcome"] == "push").sum())
        assert overs + unders + pushes == games

        over_hit = overs / games if games else 0.0
        under_hit = unders / games if games else 0.0
        push_r = pushes / games if games else 0.0

        a = DIRICHLET_ALPHA
        denom = games + 3 * a
        p_over = (overs + a) / denom if denom else 1 / 3
        p_under = (unders + a) / denom if denom else 1 / 3
        p_push = (pushes + a) / denom if denom else 1 / 3

        sample_weight = games / (games + PRIOR_STRENGTH) if (games + PRIOR_STRENGTH) else 0.0

        adequate = games >= min_bin_rows
        rows.append(
            {
                "bin_index": i,
                "label": bin_defs[i]["label"],
                "games": games,
                "overs": overs,
                "unders": unders,
                "pushes": pushes,
                "over_hit_rate": round(over_hit, 6),
                "under_hit_rate": round(under_hit, 6),
                "push_rate": round(push_r, 6),
                "model_prob_over": round(p_over, 6),
                "model_prob_under": round(p_under, 6),
                "model_prob_push": round(p_push, 6),
                "sample_weight": round(float(sample_weight), 6),
                "min_bin_rows_threshold": min_bin_rows,
                "adequate_sample": adequate,
                "sample_status": "adequate" if adequate else "thin",
            }
        )
    return rows


def build_stratified_reports(
    work: pd.DataFrame,
    *,
    strat_column: str,
    dimension_name: str,
    min_cohort_rows: int,
    min_bin_rows: int,
) -> List[Dict[str, Any]]:
    """One calibration table per distinct strat value (diagnostic only)."""
    s = _stratify_series(work, strat_column)
    out: List[Dict[str, Any]] = []
    for cohort in sorted(s.unique(), key=lambda x: (-(s == x).sum(), str(x))):
        sub = work.loc[s == cohort]
        n = int(len(sub))
        adequate_cohort = n >= min_cohort_rows
        out.append(
            {
                "dimension": dimension_name,
                "source_column": strat_column,
                "cohort": cohort,
                "rows_used": n,
                "min_cohort_rows_threshold": min_cohort_rows,
                "cohort_sample_status": "adequate" if adequate_cohort else "thin",
                "cohort_adequate_sample": adequate_cohort,
                "bins": aggregate_bins(sub, min_bin_rows=min_bin_rows),
            }
        )
    return out


def print_summary(
    bin_rows: List[Dict[str, Any]],
    *,
    title: str,
    cohort_rows: int,
    cohort_status: str,
    min_bin_rows: int,
    min_cohort_rows: int,
) -> None:
    cols = [
        "label",
        "games",
        "overs",
        "unders",
        "pushes",
        "over_hit_rate",
        "under_hit_rate",
        "model_prob_over",
        "model_prob_under",
        "sample_weight",
        "sample_status",
    ]
    if title.strip():
        print(f"\n{title}")
    print(
        f"Cohort rows={cohort_rows} | cohort_status={cohort_status} "
        f"(min_cohort={min_cohort_rows}, min_bin={min_bin_rows})\n"
    )
    header = " | ".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in bin_rows:
        line = " | ".join(f"{str(r.get(c, '')):>16}" for c in cols)
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ou_edge_calibration.json from graded archive predictions CSVs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        help="CSV files or directories containing predictions_*.csv (default: archive/)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="glob",
        default=[],
        help="Glob pattern(s) for predictions CSVs (relative to repo root if not absolute).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "calibration" / "ou_edge_calibration.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not drop duplicate Game (+ Game_Date when present).",
    )
    parser.add_argument(
        "--min-bin-rows",
        type=int,
        default=DEFAULT_MIN_BIN_ROWS,
        metavar="N",
        help=f"Bins with fewer than N graded rows are marked thin (default {DEFAULT_MIN_BIN_ROWS}).",
    )
    parser.add_argument(
        "--min-cohort-rows",
        type=int,
        default=DEFAULT_MIN_COHORT_ROWS,
        metavar="N",
        help=f"Cohorts (pooled or stratified) with fewer than N rows are marked thin (default {DEFAULT_MIN_COHORT_ROWS}).",
    )
    parser.add_argument(
        "--book-filter",
        action="append",
        default=None,
        metavar="BOOK",
        help="Repeatable. Restrict rows to Odds_Book matching any token (case-insensitive). Comma-separated allowed per flag.",
    )
    parser.add_argument(
        "--source-filter",
        action="append",
        default=None,
        metavar="SOURCE",
        help="Repeatable. Restrict rows to Total_Line_Source matching any token (case-insensitive). Comma-separated allowed per flag.",
    )
    parser.add_argument(
        "--min-production-rows",
        type=int,
        default=DEFAULT_MIN_PRODUCTION_TOTAL_ROWS,
        metavar="N",
        help=f"Minimum graded rows required for production_ready (default {DEFAULT_MIN_PRODUCTION_TOTAL_ROWS}).",
    )
    parser.add_argument(
        "--min-production-core-bin-rows",
        type=int,
        default=DEFAULT_MIN_PRODUCTION_CORE_BIN_ROWS,
        metavar="N",
        help=f"Minimum rows in each bounded-edge core bin for production_ready (default {DEFAULT_MIN_PRODUCTION_CORE_BIN_ROWS}).",
    )
    args = parser.parse_args()

    files = _collect_csv_paths(args.paths, args.glob)
    if not files:
        print("No predictions_*.csv files found.", file=sys.stderr)
        return 1

    df, read_errors, file_names = load_frames(files)
    rows_read = len(df)

    try:
        work, dedupe_subset = build_calibration_table(
            df, dedupe=not args.no_dedupe
        )
    except ValueError as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 1

    book_filters = _collect_filter_tokens(args.book_filter)
    source_filters = _collect_filter_tokens(args.source_filter)
    rows_before_slice = len(work)
    work = apply_book_source_filters(work, book_filters, source_filters)

    rows_used = len(work)
    min_bin = max(0, int(args.min_bin_rows))
    min_cohort = max(0, int(args.min_cohort_rows))
    min_prod_total = max(0, int(args.min_production_rows))
    min_prod_core = max(0, int(args.min_production_core_bin_rows))
    pooled_cohort_status = "adequate" if rows_used >= min_cohort else "thin"

    bin_rows = aggregate_bins(work, min_bin_rows=min_bin)

    production_ready, production_ready_reasons, thin_in_core_diag = (
        evaluate_production_readiness(
            bin_rows,
            rows_used,
            min_production_total_rows=min_prod_total,
            min_production_core_bin_rows=min_prod_core,
            diagnostic_min_bin_rows=min_bin,
        )
    )
    diagnostic_only = not production_ready

    stratified_odds_book = build_stratified_reports(
        work,
        strat_column="Odds_Book",
        dimension_name="Odds_Book",
        min_cohort_rows=min_cohort,
        min_bin_rows=min_bin,
    )
    stratified_line_source = build_stratified_reports(
        work,
        strat_column="Total_Line_Source",
        dimension_name="Total_Line_Source",
        min_cohort_rows=min_cohort,
        min_bin_rows=min_bin,
    )

    n_bins = len(EDGE_BIN_EDGES) - 1
    bin_definitions = []
    for i in range(n_bins):
        lo, hi = EDGE_BIN_EDGES[i], EDGE_BIN_EDGES[i + 1]
        bin_definitions.append(
            {
                "index": i,
                "label": _bin_label(lo, hi, i, n_bins),
                "lo": None if math.isinf(lo) and lo < 0 else lo,
                "hi": None if math.isinf(hi) and hi > 0 else hi,
            }
        )

    thin_bin_count = sum(1 for b in bin_rows if b.get("sample_status") == "thin")

    artifact: Dict[str, Any] = {
        "schema_version": 3,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_files": file_names,
            "files_matched": len(files),
            "read_errors": read_errors,
            "rows_read": rows_read,
            "rows_used": rows_used,
            "rows_used_before_book_source_filters": rows_before_slice,
            "dirichlet_alpha_per_outcome": DIRICHLET_ALPHA,
            "prior_strength_for_sample_weight": PRIOR_STRENGTH,
            "diagnostic_mode": True,
            "diagnostic_only": diagnostic_only,
            "production_ready": production_ready,
            "production_ready_reasons": production_ready_reasons,
            "thin_bins_in_core_diagnostic": thin_in_core_diag,
            "applied_book_filter": book_filters,
            "applied_source_filter": source_filters,
            "diagnostic_note": (
                "Offline calibration artifact. production_ready is False unless cohort and "
                "core-bin thresholds are met; diagnostic_only is True when not production_ready. "
                "Stratified views and thin-sample flags do not change production engine behavior."
            ),
            "sample_thresholds": {
                "min_bin_rows": min_bin,
                "min_cohort_rows": min_cohort,
            },
            "production_readiness_thresholds": {
                "min_production_total_rows": min_prod_total,
                "min_production_core_bin_rows": min_prod_core,
                "core_bin_indices": list(_core_bin_indices()),
                "diagnostic_min_bin_rows_reference": min_bin,
            },
            "pooled_cohort_sample_status": pooled_cohort_status,
            "pooled_cohort_adequate_sample": pooled_cohort_status == "adequate",
            "pooled_thin_bin_count": thin_bin_count,
            "filters": {
                "ou_result_in_win_loss_push": True,
                "total_is_real_true_when_column_present": True,
                "prediction_firm_over_or_under_only": True,
                "dedupe_game_game_date": not args.no_dedupe,
                "dedupe_subset_columns": dedupe_subset,
            },
            "edge_source": "OU_Edge or Edge else Projected_Total minus Bet_Line/Odds_Line/Vegas_Line",
        },
        "bin_definitions": bin_definitions,
        "bins": bin_rows,
        "stratified": {
            "by_odds_book": stratified_odds_book,
            "by_total_line_source": stratified_line_source,
        },
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote {out_path} ({rows_used} graded firm O/U rows from {rows_read} raw rows, {len(files)} files)")
    if book_filters or source_filters:
        print(
            f"Applied filters: book={book_filters or '(none)'} source={source_filters or '(none)'} "
            f"({rows_before_slice} → {rows_used} rows)\n"
        )
    print(
        "\n*** OFFLINE calibration — check metadata.production_ready before any prod use ***\n"
        f"production_ready={production_ready} | diagnostic_only={diagnostic_only}\n"
        f"Pooled cohort: {pooled_cohort_status} (rows={rows_used}, min_cohort={min_cohort}). "
        f"Thin bins (games < {min_bin}): {thin_bin_count} / {len(bin_rows)}. "
        f"Thin core bins (diagnostic): {thin_in_core_diag}\n"
        "See metadata.production_ready_reasons and stratified blocks in JSON.\n"
    )
    print_summary(
        bin_rows,
        title="--- Pooled (all books / sources) ---",
        cohort_rows=rows_used,
        cohort_status=pooled_cohort_status,
        min_bin_rows=min_bin,
        min_cohort_rows=min_cohort,
    )

    def _print_strat_block(label: str, items: List[Dict[str, Any]]) -> None:
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        for block in items:
            st = block["cohort_sample_status"]
            if st != "adequate":
                print(f"\n--- {block['dimension']}={block['cohort']!r} (rows={block['rows_used']}, "
                      f"cohort THIN — interpret cautiously) ---")
            else:
                print(f"\n--- {block['dimension']}={block['cohort']!r} (rows={block['rows_used']}, cohort adequate) ---")
            print_summary(
                block["bins"],
                title="",
                cohort_rows=block["rows_used"],
                cohort_status=st,
                min_bin_rows=min_bin,
                min_cohort_rows=min_cohort,
            )

    _print_strat_block("Stratified: Odds_Book", stratified_odds_book)
    _print_strat_block("Stratified: Total_Line_Source", stratified_line_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())

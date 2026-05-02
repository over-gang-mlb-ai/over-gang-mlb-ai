#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derived analysis: recompute Stage 1 O/U probability telemetry on historical archive rows
using the same formulas as model/overgang_model.py (inline ~3925–3995).

Does not import the live model, mutate archives, or call external APIs.

Usage:
  python scripts/backfill_ou_probability_telemetry.py
  python scripts/backfill_ou_probability_telemetry.py --input-glob 'archive/predictions_2026*.csv'
  python scripts/backfill_ou_probability_telemetry.py --output analysis/ou_probability_backfill_2026.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _american_odds_to_implied(american: Any) -> Optional[float]:
    """Mirror model/overgang_model.py — implied probability from American odds."""
    if american is None:
        return None
    try:
        v = float(american)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    if v < 0:
        return abs(v) / (100.0 + abs(v))
    return 100.0 / (100.0 + v)


def _ml_pair_devig_implied(over_am: Any, under_am: Any) -> Tuple[Optional[float], Optional[float], str]:
    """Mirror model — proportional de-vig for two American sides (over / under)."""
    ih = _american_odds_to_implied(over_am)
    ia = _american_odds_to_implied(under_am)
    if ih is None or ia is None:
        return None, None, "missing_market"
    s = ih + ia
    if s <= 0:
        return None, None, "invalid_odds"
    return ih / s, ia / s, "ok"


def _clamp_unit_interval(x: Any, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    """Mirror model/overgang_model.py."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return max(lo, min(hi, v))


def _is_ou_prediction_row(pred: Any) -> bool:
    p = (str(pred) if pred is not None else "").strip().upper()
    return (
        p.startswith("LEAN OVER")
        or p.startswith("LEAN UNDER")
        or p.startswith("OVER")
        or p.startswith("UNDER")
    )


def _side_from_row(row: Dict[str, str]) -> str:
    """Mirror live Side: over if OVER in prediction else under if UNDER in prediction."""
    pred = row.get("Prediction") or ""
    s = (
        "over"
        if "OVER" in pred.upper()
        else ("under" if "UNDER" in pred.upper() else "")
    )
    if s:
        return s
    ou = (row.get("OU_Side") or "").strip().lower()
    if ou in ("over", "under"):
        return ou
    return ""


def _parse_conf01_live_mirror(row: Dict[str, str]) -> Optional[float]:
    """
    Map archive columns to the live `confidence` scalar (0–1) used in telemetry.
    Prefer Confidence_Value, then Confidence, then OU_Confidence (percent strings).
    """
    cv = row.get("Confidence_Value")
    if cv is not None and str(cv).strip() != "":
        try:
            v = float(cv)
            if v == v and 0.0 <= v <= 1.0:
                return v
        except (TypeError, ValueError):
            pass
    for col in ("Confidence", "OU_Confidence"):
        s = str(row.get(col) or "").replace("%", "").strip()
        if not s:
            continue
        try:
            v = float(s)
            if v != v:
                continue
            if 0.0 <= v <= 1.0:
                return v
            if 0.0 < v <= 100.0:
                return v / 100.0
        except (TypeError, ValueError):
            continue
    return None


def _parse_edge_abs_live_mirror(row: Dict[str, str]) -> Optional[float]:
    """
    Mirror abs(edge): prefer OU_Edge / Edge; else abs(Projected_Total − line) using
    first parseable line among Bet_Line, Odds_Line, Vegas_Line.
    """
    for key in ("OU_Edge", "Edge"):
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return abs(float(raw))
        except (TypeError, ValueError):
            continue
    try:
        pt = float(row.get("Projected_Total"))
    except (TypeError, ValueError):
        return None
    for lk in ("Bet_Line", "Odds_Line", "Vegas_Line"):
        try:
            line = float(row.get(lk))
            return abs(pt - line)
        except (TypeError, ValueError):
            continue
    return None


def _is_graded_ou(row: Dict[str, str]) -> bool:
    s = (row.get("OU_Result") or "").strip().upper()
    return s in ("WIN", "LOSS", "PUSH")


def _apply_telemetry(row: Dict[str, str]) -> Dict[str, str]:
    """
    Apply the same sequence as live model (game_data updates + flags).
    Returns a new dict with telemetry keys filled (strings and numeric strings for CSV).
    """
    out = dict(row)
    _pr_flags = ["provisional_not_for_fire"]

    for _pk in (
        "OU_Implied_Prob_Over",
        "OU_Implied_Prob_Under",
        "OU_Implied_Prob_Pick",
        "OU_True_Prob_Over",
        "OU_True_Prob_Under",
        "OU_True_Prob_Pick",
        "OU_Prob_Edge",
        "OU_Prob_Edge_Side",
    ):
        out[_pk] = ""
    out["OU_Prob_Method"] = "v1_edge_conf_provisional"

    _im_over, _im_under, _im_st = _ml_pair_devig_implied(
        row.get("Over_Juice"), row.get("Under_Juice")
    )
    if _im_st == "ok" and _im_over is not None and _im_under is not None:
        out["OU_Implied_Prob_Over"] = str(round(float(_im_over), 4))
        out["OU_Implied_Prob_Under"] = str(round(float(_im_under), 4))
    else:
        _pr_flags.append("ou_implied_unavailable")

    _sd = _side_from_row(row)
    if _sd in ("over", "under"):
        out["OU_Prob_Edge_Side"] = _sd
        if _im_st == "ok":
            _im_pick = _im_over if _sd == "over" else _im_under
            out["OU_Implied_Prob_Pick"] = str(round(float(_im_pick), 4))

        _ou_conf01 = _parse_conf01_live_mirror(row)
        _edge_abs = _parse_edge_abs_live_mirror(row)

        if _ou_conf01 is not None and _edge_abs is not None:
            _edge_term = min(_edge_abs, 3.0) * 0.012
            _p_pick = _clamp_unit_interval(_ou_conf01 + _edge_term, 0.48, 0.85)
            if _p_pick is not None:
                if _sd == "over":
                    _t_o, _t_u = _p_pick, 1.0 - _p_pick
                else:
                    _t_u, _t_o = _p_pick, 1.0 - _p_pick
                out["OU_True_Prob_Pick"] = str(round(_p_pick, 4))
                out["OU_True_Prob_Over"] = str(round(_t_o, 4))
                out["OU_True_Prob_Under"] = str(round(_t_u, 4))
        else:
            _pr_flags.append("true_prob_inputs_missing")

        if (
            _im_st == "ok"
            and out["OU_Implied_Prob_Pick"] != ""
            and out["OU_True_Prob_Pick"] != ""
        ):
            try:
                out["OU_Prob_Edge"] = str(
                    round(
                        float(out["OU_True_Prob_Pick"])
                        - float(out["OU_Implied_Prob_Pick"]),
                        4,
                    )
                )
            except (TypeError, ValueError):
                pass
    else:
        _pr_flags.append("ou_side_missing")

    out["OU_Prob_Calibration_Flag"] = "|".join(_pr_flags)
    return out


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        return [dict(r) for r in reader]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill O/U probability telemetry on archive CSVs (derived analysis only)."
    )
    parser.add_argument(
        "--input-glob",
        default="archive/predictions_2026*.csv",
        help="Glob relative to repo root (default: archive/predictions_2026*.csv)",
    )
    parser.add_argument(
        "--output",
        default="analysis/ou_probability_backfill_2026.csv",
        help="Output CSV path relative to repo root",
    )
    args = parser.parse_args()

    pattern = str(ROOT / args.input_glob)
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}", file=sys.stderr)
        return 1

    total_read = 0
    ou_rows: List[Dict[str, str]] = []
    graded_ou = 0
    graded_clean = 0
    graded_degraded = 0
    n_implied = 0
    n_true = 0
    n_edge = 0

    for fp in paths:
        try:
            rows = _read_csv_rows(fp)
        except OSError as e:
            print(f"Skip {fp}: {e}", file=sys.stderr)
            continue
        for r in rows:
            r = {k: (v if v is not None else "") for k, v in r.items()}
            r["_file"] = fp.name
            total_read += 1
            if not _is_ou_prediction_row(r.get("Prediction")):
                continue
            ou = _apply_telemetry(r)
            ou_rows.append(ou)

            if ou.get("OU_Implied_Prob_Over") != "":
                n_implied += 1
            if ou.get("OU_True_Prob_Pick") != "":
                n_true += 1
            if ou.get("OU_Prob_Edge") != "":
                n_edge += 1

            if _is_graded_ou(ou):
                graded_ou += 1
                dq = (ou.get("Data_Quality_Flag") or "").strip()
                if not dq:
                    graded_clean += 1
                else:
                    graded_degraded += 1

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order: union of keys
    fieldnames: List[str] = []
    seen = set()
    for row in ou_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in ou_rows:
            w.writerow({c: row.get(c, "") for c in fieldnames})

    print("--- ou_probability_backfill (derived) ---")
    print(f"  Files read:           {len(paths)}")
    print(f"  Total rows read:      {total_read}")
    print(f"  O/U rows (filtered):  {len(ou_rows)}")
    print(f"  Graded O/U rows:      {graded_ou}")
    print(f"    clean (graded):     {graded_clean}")
    print(f"    degraded (graded): {graded_degraded}")
    print(f"  Implied prob filled:  {n_implied}")
    print(f"  True prob filled:     {n_true}")
    print(f"  Prob edge filled:     {n_edge}")
    print(f"  Output:               {out_path}")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())

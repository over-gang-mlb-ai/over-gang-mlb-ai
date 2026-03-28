#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-grade OU_Result and ML_Result in an exported predictions CSV using final MLB scores
for the slate date. Does not compute ROI or profit.

Uses the same Game-key normalization and slate-date rules as tools/fill_final_scores.py.

Usage:
  python scripts/auto_grade_predictions_csv.py path/to/predictions_YYYYMMDD_HHMM.csv
  python scripts/auto_grade_predictions_csv.py path/to/file.csv --date 2025-09-15
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _ensure_root_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _load_fill_final_scores():
    """Load tools/fill_final_scores.py without requiring tools to be a package."""
    path = ROOT / "tools" / "fill_final_scores.py"
    spec = importlib.util.spec_from_file_location("fill_final_scores", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# MLB finalized games: detailedState / statsapi typically "Final" or "Game Over".
_FINAL_STATUS_TOKENS = frozenset({"final", "game over"})


def _normalized_game_status_string(g: dict) -> str:
    """
    Comparable status for final detection. Aligns with tools/fill_final_scores._game_final_total:
    use status or top-level abstract_game_state; if status is a dict, prefer abstractGameState
    then detailedState; if those are empty, treat codedGameState/statusCode F as final (MLB API).
    """
    if not isinstance(g, dict):
        return ""
    raw = g.get("status") or g.get("abstract_game_state")
    if isinstance(raw, dict):
        inner = (
            raw.get("abstractGameState")
            or raw.get("detailedState")
            or ""
        )
        if not str(inner).strip():
            code = raw.get("codedGameState") or raw.get("statusCode")
            if str(code).strip().upper() in ("F", "FINAL"):
                return "final"
            return ""
        return str(inner).strip().lower()
    return str(raw or "").strip().lower()


def _is_final_status(g: dict) -> bool:
    return _normalized_game_status_string(g) in _FINAL_STATUS_TOKENS


def _parse_final_game_detail(g: dict) -> Optional[Dict[str, Any]]:
    """From statsapi schedule game dict or raw schedule JSON game dict."""
    if not isinstance(g, dict) or not _is_final_status(g):
        return None
    away_name = (
        (g.get("away_name") or "").strip()
        or ((g.get("teams") or {}).get("away", {}).get("team", {}) or {}).get("name", "").strip()
    )
    home_name = (
        (g.get("home_name") or "").strip()
        or ((g.get("teams") or {}).get("home", {}).get("team", {}) or {}).get("name", "").strip()
    )
    away_score = g.get("away_score")
    home_score = g.get("home_score")
    if away_score is None and isinstance(g.get("teams"), dict):
        away_score = g["teams"].get("away", {}).get("score")
    if home_score is None and isinstance(g.get("teams"), dict):
        home_score = g["teams"].get("home", {}).get("score")
    if not away_name or not home_name or away_score is None or home_score is None:
        return None
    try:
        ar = int(away_score)
        hr = int(home_score)
    except (TypeError, ValueError):
        return None
    return {
        "away_name": away_name,
        "home_name": home_name,
        "away_runs": ar,
        "home_runs": hr,
        "total": ar + hr,
    }


def fetch_final_games_by_key(
    date_yyyy_mm_dd: str, game_to_key_fn
) -> Tuple[Dict[str, Dict[str, Any]], dict]:
    """Normalized 'away @ home' -> detail dict with runs and total. Second value is fetch diagnostics."""
    out: Dict[str, Dict[str, Any]] = {}
    diag: dict = {
        "statsapi_games": 0,
        "statsapi_error": None,
        "statsapi_parseable": 0,
        "statsapi_key_drops": 0,
        "http_games": 0,
        "http_error": None,
        "http_parseable": 0,
        "http_key_drops": 0,
    }
    games = []
    try:
        from statsapi import schedule

        games = schedule(start_date=date_yyyy_mm_dd, end_date=date_yyyy_mm_dd) or []
    except Exception as e:
        diag["statsapi_error"] = str(e)
        games = []
    diag["statsapi_games"] = len(games)

    for g in games:
        det = _parse_final_game_detail(g)
        if not det:
            continue
        diag["statsapi_parseable"] += 1
        key = game_to_key_fn(f"{det['away_name']} @ {det['home_name']}")
        if not key:
            diag["statsapi_key_drops"] += 1
            continue
        out[key] = det

    if out:
        return out, diag

    try:
        import requests

        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_yyyy_mm_dd}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        diag["http_error"] = str(e)
        return out, diag

    http_games: list = []
    for d in data.get("dates") or []:
        for g in d.get("games") or []:
            http_games.append(g)
    diag["http_games"] = len(http_games)

    for g in http_games:
        det = _parse_final_game_detail(g)
        if not det:
            continue
        diag["http_parseable"] += 1
        key = game_to_key_fn(f"{det['away_name']} @ {det['home_name']}")
        if not key:
            diag["http_key_drops"] += 1
            continue
        out[key] = det
    return out, diag


def _print_fetch_diagnostics_if_empty(
    date_yyyy_mm_dd: str, games_map: Dict[str, Dict[str, Any]], diag: dict
) -> None:
    """If nothing loaded, print stderr hints (fetch errors vs parse vs key drops)."""
    if games_map:
        return
    print(
        f"Note: no finalized games loaded for {date_yyyy_mm_dd} "
        "(empty schedule, non-final status, missing scores, or game key mismatch).",
        file=sys.stderr,
    )
    if diag.get("statsapi_error"):
        print(f"  statsapi error: {diag['statsapi_error']}", file=sys.stderr)
    if diag.get("http_error"):
        print(f"  http schedule error: {diag['http_error']}", file=sys.stderr)
    sg, sp, sk = (
        diag.get("statsapi_games", 0),
        diag.get("statsapi_parseable", 0),
        diag.get("statsapi_key_drops", 0),
    )
    hg, hp, hk = (
        diag.get("http_games", 0),
        diag.get("http_parseable", 0),
        diag.get("http_key_drops", 0),
    )
    if sg or sp or sk:
        print(
            f"  statsapi: {sg} row(s), {sp} final w/scores, {sk} dropped (no game key).",
            file=sys.stderr,
        )
    if hg or hp or hk:
        print(
            f"  http: {hg} row(s), {hp} final w/scores, {hk} dropped (no game key).",
            file=sys.stderr,
        )


def _is_pending_result(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    s = str(val).strip().upper()
    return s == "" or s == "PENDING"


def _grade_ou(final_total: float, bet_line: float, side: str) -> Optional[str]:
    side = (side or "").strip().lower()
    if side not in ("over", "under"):
        return None
    try:
        ft = float(final_total)
        bl = float(bet_line)
    except (TypeError, ValueError):
        return None
    if pd.isna(ft) or pd.isna(bl):
        return None
    if ft > bl:
        return "WIN" if side == "over" else "LOSS"
    if ft < bl:
        return "LOSS" if side == "over" else "WIN"
    return "PUSH"


def _grade_ml(ml_pick: str, det: Dict[str, Any]) -> Optional[str]:
    raw = (ml_pick or "").strip()
    if not raw:
        return None
    if not raw.upper().endswith(" ML"):
        return None
    pick_team = raw[: -len(" ML")].strip()
    if not pick_team:
        return None
    ar = int(det["away_runs"])
    hr = int(det["home_runs"])
    if ar > hr:
        winner_name = det["away_name"]
    elif hr > ar:
        winner_name = det["home_name"]
    else:
        return "PUSH"

    _ensure_root_path()
    from core.public_betting_loader import normalize_team_name

    if normalize_team_name(pick_team) == normalize_team_name(winner_name):
        return "WIN"
    return "LOSS"


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
        description="Set OU_Result and ML_Result from final MLB scores (no ROI)."
    )
    parser.add_argument("csv_path", type=Path, help="Predictions CSV path")
    parser.add_argument(
        "--date",
        help="Slate date YYYY-MM-DD (default: from predictions_YYYYMMDD_HHMM filename)",
    )
    args = parser.parse_args()
    csv_path: Path = args.csv_path

    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    mod = _load_fill_final_scores()
    game_to_key_fn = mod.game_to_key
    parse_date_from_filename = mod.parse_date_from_filename

    if args.date:
        target_date = args.date.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", target_date):
            print("--date must be YYYY-MM-DD", file=sys.stderr)
            return 1
    else:
        target_date = parse_date_from_filename(str(csv_path))
        if not target_date:
            print(
                "Could not infer date from filename; use --date YYYY-MM-DD",
                file=sys.stderr,
            )
            return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        return 1

    for col in ("OU_Result", "ML_Result"):
        if col not in df.columns:
            df[col] = "PENDING"

    games, fetch_diag = fetch_final_games_by_key(target_date, game_to_key_fn)
    _print_fetch_diagnostics_if_empty(target_date, games, fetch_diag)

    ou_set = ml_set = 0
    ou_still = ml_still = 0

    for i in range(len(df)):
        game_val = df.at[i, "Game"]
        key = game_to_key_fn(str(game_val)) if game_val is not None else None
        det = games.get(key) if key else None

        if det is None:
            if _is_pending_result(df.at[i, "OU_Result"]):
                ou_still += 1
            if _is_pending_result(df.at[i, "ML_Result"]):
                ml_still += 1
            continue

        ft = float(det["total"])

        if _is_pending_result(df.at[i, "OU_Result"]):
            side = df.at[i, "Side"] if "Side" in df.columns else ""
            bet_line = df.at[i, "Bet_Line"] if "Bet_Line" in df.columns else None
            try:
                bl = float(bet_line)
            except (TypeError, ValueError):
                bl = float("nan")
            ou = _grade_ou(ft, bl, str(side) if side is not None else "")
            if ou is not None:
                df.at[i, "OU_Result"] = ou
                ou_set += 1
            else:
                ou_still += 1
        else:
            pass

        if _is_pending_result(df.at[i, "ML_Result"]):
            mp = df.at[i, "ML_Pick"] if "ML_Pick" in df.columns else ""
            ml = _grade_ml(str(mp) if mp is not None else "", det)
            if ml is not None:
                df.at[i, "ML_Result"] = ml
                ml_set += 1
            else:
                ml_still += 1
        else:
            pass

    for col in ("OU_Result", "ML_Result"):
        if col in df.columns:
            df[col] = df[col].astype(object)

    try:
        _write_csv_atomic(csv_path, df)
    except Exception as e:
        print(f"Write failed: {e}", file=sys.stderr)
        return 1

    print(f"Slate date: {target_date}")
    print(f"Final games loaded: {len(games)}")
    print(f"OU_Result set: {ou_set} | still pending: {ou_still}")
    print(f"ML_Result set: {ml_set} | still pending: {ml_still}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

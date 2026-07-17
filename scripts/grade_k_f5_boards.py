#!/usr/bin/env python3
"""Postgame grader for Over Gang pitcher-K and F5 board CSVs.

Writes separate graded outputs and never overwrites original archive files.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "archive"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINESCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
FINAL_STATES = {"final", "game over"}
GRADE_SOURCE = "mlb_statsapi"


def _ensure_root_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _normalize_team(name: Any) -> str:
    _ensure_root_path()
    try:
        from core.public_betting_loader import normalize_team_name

        return normalize_team_name(str(name or "").strip())
    except Exception:
        return str(name or "").strip().lower()


def _normalize_person_name(name: Any) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().replace(".", "").split())


def _game_key(game: Any) -> Optional[str]:
    s = str(game or "").strip()
    if " @ " not in s:
        return None
    away, home = s.split(" @ ", 1)
    away_norm = _normalize_team(away)
    home_norm = _normalize_team(home)
    if not away_norm or not home_norm:
        return None
    return f"{away_norm} @ {home_norm}"


def _date_arg_to_api(date_arg: str) -> str:
    s = str(date_arg or "").strip()
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    raise ValueError("--date must be YYYYMMDD or YYYY-MM-DD")


def _date_arg_to_compact(date_arg: str) -> str:
    return _date_arg_to_api(date_arg).replace("-", "")


def _extract_stamp(path: Path, prefix: str) -> Optional[str]:
    m = re.match(rf"{re.escape(prefix)}_(\d{{8}}_\d{{4}})\.csv$", path.name, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_date_from_path(path: Path, prefix: str) -> Optional[str]:
    stamp = _extract_stamp(path, prefix)
    if not stamp:
        return None
    compact = stamp.split("_", 1)[0]
    return _date_arg_to_api(compact)


def _latest_matching(pattern: str) -> Optional[Path]:
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _resolve_input_file(
    supplied: Optional[str],
    prefix: str,
    date_compact: Optional[str],
) -> Optional[Path]:
    if supplied:
        p = Path(supplied)
        if not p.is_absolute():
            p = ROOT / p
        return p
    if date_compact:
        return _latest_matching(str(ARCHIVE_DIR / f"{prefix}_{date_compact}_*.csv"))
    return _latest_matching(str(ARCHIVE_DIR / f"{prefix}_*.csv"))


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        f = float(value)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _status_text(game: dict) -> str:
    status = game.get("status") or {}
    if isinstance(status, dict):
        text = status.get("detailedState") or status.get("abstractGameState") or ""
        code = status.get("codedGameState") or status.get("statusCode") or ""
        if not text and str(code).strip().upper() in {"F", "FINAL"}:
            return "final"
        return str(text or "").strip().lower()
    return str(status or "").strip().lower()


def _is_final_game(game: dict) -> bool:
    return _status_text(game) in FINAL_STATES


def fetch_schedule_games(date_yyyy_mm_dd: str) -> Dict[str, Dict[str, Any]]:
    params = {"sportId": 1, "date": date_yyyy_mm_dd}
    r = requests.get(SCHEDULE_URL, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    out: Dict[str, Dict[str, Any]] = {}
    for date_row in payload.get("dates") or []:
        for game in date_row.get("games") or []:
            teams = game.get("teams") or {}
            away_name = (((teams.get("away") or {}).get("team") or {}).get("name") or "").strip()
            home_name = (((teams.get("home") or {}).get("team") or {}).get("name") or "").strip()
            key = _game_key(f"{away_name} @ {home_name}")
            if not key:
                continue
            out[key] = {
                "gamePk": game.get("gamePk"),
                "status": _status_text(game),
                "is_final": _is_final_game(game),
                "away_name": away_name,
                "home_name": home_name,
            }
    return out


def fetch_linescore(game_pk: Any) -> dict:
    r = requests.get(LINESCORE_URL.format(game_pk=game_pk), timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_boxscore(game_pk: Any) -> dict:
    r = requests.get(BOXSCORE_URL.format(game_pk=game_pk), timeout=20)
    r.raise_for_status()
    return r.json()


def _f5_actual_from_linescore(linescore: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    away_total = 0
    home_total = 0
    seen_any = False
    for inning in linescore.get("innings") or []:
        try:
            num = int(inning.get("num"))
        except (TypeError, ValueError):
            continue
        if not 1 <= num <= 5:
            continue
        away_runs = ((inning.get("away") or {}).get("runs"))
        home_runs = ((inning.get("home") or {}).get("runs"))
        if away_runs is None or home_runs is None:
            continue
        try:
            away_total += int(away_runs)
            home_total += int(home_runs)
            seen_any = True
        except (TypeError, ValueError):
            continue
    if not seen_any:
        return None, None, None
    return away_total, home_total, away_total + home_total


def _pitcher_ks_from_boxscore(boxscore: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    teams = boxscore.get("teams") or {}
    for side in ("away", "home"):
        players = ((teams.get(side) or {}).get("players") or {})
        for player in players.values():
            if not isinstance(player, dict):
                continue
            name = ((player.get("person") or {}).get("fullName") or "").strip()
            if not name:
                continue
            pitching = ((player.get("stats") or {}).get("pitching") or {})
            if "strikeOuts" not in pitching:
                continue
            try:
                out[_normalize_person_name(name)] = int(pitching.get("strikeOuts"))
            except (TypeError, ValueError):
                continue
    return out


def _grade_over_under(pick: Any, actual: Optional[float], line: Optional[float]) -> str:
    pick_s = str(pick or "").strip().upper()
    if pick_s not in {"OVER", "UNDER"}:
        return "NO_PLAY"
    if actual is None or line is None:
        return "NOT_GRADED"
    if actual == line:
        return "PUSH"
    if pick_s == "OVER":
        return "WIN" if actual > line else "LOSS"
    return "WIN" if actual < line else "LOSS"


def _record(df: pd.DataFrame, result_col: str, mask: pd.Series) -> str:
    sub = df[mask]
    wins = int((sub[result_col] == "WIN").sum()) if result_col in sub else 0
    losses = int((sub[result_col] == "LOSS").sum()) if result_col in sub else 0
    pushes = int((sub[result_col] == "PUSH").sum()) if result_col in sub else 0
    graded = wins + losses + pushes
    return f"{wins}-{losses}-{pushes} ({graded} graded)"


def grade_f5_board(
    f5_file: Path,
    schedule: Dict[str, Dict[str, Any]],
    linescore_cache: Dict[Any, dict],
) -> Path:
    df = pd.read_csv(f5_file)
    rows = []
    for _, row in df.iterrows():
        out = row.to_dict()
        game_key = _game_key(row.get("Game"))
        game_info = schedule.get(game_key or "", {})
        game_pk = game_info.get("gamePk")
        out["MLB_Game_PK"] = game_pk or ""
        out["F5_Away_Runs_Actual"] = ""
        out["F5_Home_Runs_Actual"] = ""
        out["F5_Total_Actual"] = ""
        out["F5_Result"] = "NOT_GRADED"
        out["F5_Graded"] = False
        out["F5_Grade_Source"] = GRADE_SOURCE
        out["F5_Grade_Note"] = ""

        legacy_pick = str(
            row.get("F5_Pick") or ""
        ).strip().upper()
        decision_side = str(
            row.get("F5_Decision_Side") or ""
        ).strip().upper()
        f5_fired = _parse_bool(
            row.get("F5_Fired")
        )

        # New independent F5 decisions are graded using the
        # actual price-aware selected side. Historical boards
        # without F5_Fired retain their legacy F5_Pick grading.
        if (
            f5_fired
            and decision_side in {"OVER", "UNDER"}
        ):
            pick = decision_side
        else:
            pick = legacy_pick

        out["F5_Graded_Side"] = pick
        line = _parse_float(row.get("F5_Market_Line"))
        if pick not in {"OVER", "UNDER"}:
            out["F5_Result"] = "NO_PLAY"
            out["F5_Grade_Note"] = "blank_f5_pick"
        elif line is None:
            out["F5_Result"] = "NO_PLAY"
            out["F5_Grade_Note"] = "missing_market_line"
        elif not game_pk:
            out["F5_Result"] = "NOT_GRADED"
            out["F5_Grade_Note"] = "game_not_found"
        elif not game_info.get("is_final"):
            out["F5_Result"] = "PENDING"
            out["F5_Grade_Note"] = "game_not_final"
        else:
            if game_pk not in linescore_cache:
                linescore_cache[game_pk] = fetch_linescore(game_pk)
            away_f5, home_f5, total_f5 = _f5_actual_from_linescore(linescore_cache[game_pk])
            out["F5_Away_Runs_Actual"] = away_f5 if away_f5 is not None else ""
            out["F5_Home_Runs_Actual"] = home_f5 if home_f5 is not None else ""
            out["F5_Total_Actual"] = total_f5 if total_f5 is not None else ""
            result = _grade_over_under(pick, total_f5, line)
            out["F5_Result"] = result
            out["F5_Graded"] = result in {"WIN", "LOSS", "PUSH"}
            if not out["F5_Graded"]:
                out["F5_Grade_Note"] = "missing_linescore"
        rows.append(out)

    out_df = pd.DataFrame(rows)
    stamp = _extract_stamp(f5_file, "f5_board") or "unknown"
    out_path = ARCHIVE_DIR / f"graded_f5_board_{stamp}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"F5 eligible record: {_record(out_df, 'F5_Result', out_df.get('F5_Eligible', pd.Series(False, index=out_df.index)).map(_parse_bool))}")
    has_pick = out_df.get("F5_Pick", pd.Series("", index=out_df.index)).astype(str).str.upper().isin(["OVER", "UNDER"])
    print(f"F5 all-play record: {_record(out_df, 'F5_Result', has_pick)}")
    return out_path


def grade_k_board(
    k_file: Path,
    schedule: Dict[str, Dict[str, Any]],
    boxscore_cache: Dict[Any, dict],
) -> Path:
    df = pd.read_csv(k_file)
    rows = []
    pitcher_cache: Dict[Any, Dict[str, int]] = {}
    for _, row in df.iterrows():
        out = row.to_dict()
        game_key = _game_key(row.get("Game"))
        game_info = schedule.get(game_key or "", {})
        game_pk = game_info.get("gamePk")
        out["MLB_Game_PK"] = game_pk or ""
        out["Actual_Ks"] = ""
        out["K_Result"] = "NOT_GRADED"
        out["K_Graded"] = False
        out["K_Grade_Source"] = GRADE_SOURCE
        out["K_Grade_Note"] = ""

        pick = str(row.get("K_Pick") or "").strip().upper()
        line = _parse_float(row.get("K_Line"))
        if pick not in {"OVER", "UNDER"}:
            out["K_Result"] = "NO_PLAY"
            out["K_Grade_Note"] = "blank_k_pick"
        elif line is None:
            out["K_Result"] = "NO_PLAY"
            out["K_Grade_Note"] = "missing_market_line"
        elif not game_pk:
            out["K_Result"] = "NOT_GRADED"
            out["K_Grade_Note"] = "game_not_found"
        elif not game_info.get("is_final"):
            out["K_Result"] = "PENDING"
            out["K_Grade_Note"] = "game_not_final"
        else:
            if game_pk not in boxscore_cache:
                boxscore_cache[game_pk] = fetch_boxscore(game_pk)
            if game_pk not in pitcher_cache:
                pitcher_cache[game_pk] = _pitcher_ks_from_boxscore(boxscore_cache[game_pk])
            actual_ks = pitcher_cache[game_pk].get(_normalize_person_name(row.get("Pitcher")))
            if actual_ks is None:
                out["K_Result"] = "NOT_GRADED"
                out["K_Grade_Note"] = "pitcher_not_found"
            else:
                out["Actual_Ks"] = actual_ks
                result = _grade_over_under(pick, actual_ks, line)
                out["K_Result"] = result
                out["K_Graded"] = result in {"WIN", "LOSS", "PUSH"}
        rows.append(out)

    out_df = pd.DataFrame(rows)
    stamp = _extract_stamp(k_file, "pitcher_k_board") or "unknown"
    out_path = ARCHIVE_DIR / f"graded_pitcher_k_board_{stamp}.csv"
    out_df.to_csv(out_path, index=False)
    k_fired = out_df.get("K_Fired", pd.Series(False, index=out_df.index)).map(_parse_bool)
    has_pick = out_df.get("K_Pick", pd.Series("", index=out_df.index)).astype(str).str.upper().isin(["OVER", "UNDER"])
    print(f"K fired record: {_record(out_df, 'K_Result', k_fired)}")
    print(f"K all-play record: {_record(out_df, 'K_Result', has_pick)}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade Over Gang pitcher-K and F5 board CSVs.")
    p.add_argument("--date", help="Slate date as YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--k-file", help="Path to archive/pitcher_k_board_YYYYMMDD_HHMM.csv")
    p.add_argument("--f5-file", help="Path to archive/f5_board_YYYYMMDD_HHMM.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    date_compact = (
        _date_arg_to_compact(args.date)
        if args.date
        else None
    )

    k_file = _resolve_input_file(
        args.k_file,
        "pitcher_k_board",
        date_compact,
    )
    f5_file = _resolve_input_file(
        args.f5_file,
        "f5_board",
        date_compact,
    )

    k_available = bool(
        k_file and k_file.exists()
    )
    f5_available = bool(
        f5_file and f5_file.exists()
    )

    if not k_available and not f5_available:
        print(
            "Missing both pitcher K and F5 board inputs.",
            file=sys.stderr,
        )
        return 2

    date_yyyy_mm_dd = (
        _date_arg_to_api(args.date)
        if args.date
        else (
            _extract_date_from_path(
                k_file,
                "pitcher_k_board",
            )
            if k_available
            else None
        )
        or (
            _extract_date_from_path(
                f5_file,
                "f5_board",
            )
            if f5_available
            else None
        )
    )

    if not date_yyyy_mm_dd:
        print(
            "Could not infer slate date; "
            "pass --date YYYYMMDD.",
            file=sys.stderr,
        )
        return 2

    if k_available:
        print(f"Using pitcher K board: {k_file}")
    else:
        print(
            "Pitcher K board unavailable; "
            "continuing with F5 grading."
        )

    if f5_available:
        print(f"Using F5 board: {f5_file}")
    else:
        print(
            "F5 board unavailable; "
            "continuing with pitcher K grading."
        )

    print(
        f"Fetching MLB schedule for: "
        f"{date_yyyy_mm_dd}"
    )

    schedule = fetch_schedule_games(
        date_yyyy_mm_dd
    )

    print(
        f"Schedule games mapped: {len(schedule)}"
    )

    boxscore_cache: Dict[Any, dict] = {}
    linescore_cache: Dict[Any, dict] = {}

    if k_available:
        k_out = grade_k_board(
            k_file,
            schedule,
            boxscore_cache,
        )
        print(
            f"Graded pitcher K board: {k_out}"
        )

    if f5_available:
        f5_out = grade_f5_board(
            f5_file,
            schedule,
            linescore_cache,
        )
        print(f"Graded F5 board: {f5_out}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())

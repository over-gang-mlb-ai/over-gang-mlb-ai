#!/usr/bin/env python3
"""
Fills Final_Total in an Over Gang predictions CSV using MLB game results for the slate date.
Uses MLB Stats API (statsapi or requests). Does not grade bets.
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = ["Game", "Final_Total"]
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"


def _ensure_core_on_path() -> None:
    """Ensure project root is on sys.path for normalize_team_name."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_date_from_filename(csv_path: str) -> Optional[str]:
    """
    Extract slate date from filename like predictions_YYYYMMDD_HHMM.csv.
    Returns YYYY-MM-DD or None if pattern does not match.
    """
    name = Path(csv_path).name
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv", name, re.IGNORECASE)
    if not m:
        return None
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def game_to_key(game_str: str) -> Optional[str]:
    """Convert Game column value to normalized 'away @ home' key."""
    if not game_str or not isinstance(game_str, str):
        return None
    parts = game_str.strip().split(" @ ", 1)
    if len(parts) != 2:
        return None
    try:
        from core.public_betting_loader import normalize_team_name
        return f"{normalize_team_name(parts[0].strip())} @ {normalize_team_name(parts[1].strip())}"
    except Exception:
        return None


def _game_final_total(g: dict) -> Optional[tuple]:
    """Extract (away_name, home_name, total_runs) from a game dict (API or statsapi). None if not final or missing scores."""
    if not isinstance(g, dict):
        return None
    status = g.get("status") or g.get("abstract_game_state") or ""
    if isinstance(status, dict):
        status = status.get("abstractGameState") or status.get("detailedState") or ""
    if str(status).lower() != "final":
        return None
    away_name = (g.get("away_name") or (g.get("teams") or {}).get("away", {}).get("team", {}).get("name") or "").strip()
    home_name = (g.get("home_name") or (g.get("teams") or {}).get("home", {}).get("team", {}).get("name") or "").strip()
    away_score = g.get("away_score")
    home_score = g.get("home_score")
    if away_score is None and isinstance(g.get("teams"), dict):
        away_score = g["teams"].get("away", {}).get("score")
    if home_score is None and isinstance(g.get("teams"), dict):
        home_score = g["teams"].get("home", {}).get("score")
    if not away_name or not home_name or away_score is None or home_score is None:
        return None
    try:
        return (away_name, home_name, int(away_score) + int(home_score))
    except (TypeError, ValueError):
        return None


def fetch_scores_via_statsapi(date_yyyy_mm_dd: str) -> dict:
    """
    Fetch games for date using statsapi.schedule if available.
    Returns dict: normalized "away @ home" -> final_total (int).
    """
    try:
        from statsapi import schedule
        games = schedule(start_date=date_yyyy_mm_dd, end_date=date_yyyy_mm_dd)
    except Exception:
        return {}
    result = {}
    for g in games or []:
        parsed = _game_final_total(g)
        if not parsed:
            continue
        away_name, home_name, total = parsed
        key = game_to_key(f"{away_name} @ {home_name}")
        if key:
            result[key] = total
    return result


def fetch_scores_via_requests(date_yyyy_mm_dd: str) -> dict:
    """
    Fetch games for date using requests to MLB schedule endpoint.
    Returns dict: normalized "away @ home" -> final_total (int).
    """
    try:
        import requests
        url = MLB_SCHEDULE_URL.format(date=date_yyyy_mm_dd)
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}
    result = {}
    for d in data.get("dates") or []:
        for g in d.get("games") or []:
            parsed = _game_final_total(g)
            if not parsed:
                continue
            away_name, home_name, total = parsed
            key = game_to_key(f"{away_name} @ {home_name}")
            if key:
                result[key] = total
    return result


def fetch_final_totals_for_date(date_yyyy_mm_dd: str) -> dict:
    """Build mapping normalized 'away @ home' -> final_total. Tries statsapi, then requests."""
    _ensure_core_on_path()
    scores = fetch_scores_via_statsapi(date_yyyy_mm_dd)
    if not scores:
        scores = fetch_scores_via_requests(date_yyyy_mm_dd)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill Final_Total in a predictions CSV using MLB game results."
    )
    parser.add_argument("csv_path", help="Path to the predictions CSV file")
    parser.add_argument("--date", help="Override slate date (YYYY-MM-DD). If not set, date is parsed from filename.")
    args = parser.parse_args()
    csv_path = args.csv_path

    _ensure_core_on_path()

    path = Path(csv_path)
    if not path.is_file():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.date:
        target_date = args.date.strip()
        if len(target_date) != 10 or target_date[4] != "-" or target_date[7] != "-":
            print("Error: --date must be YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
    else:
        target_date = parse_date_from_filename(csv_path)
        if not target_date:
            print("Error: Could not parse date from filename (expected predictions_YYYYMMDD_HHMM.csv or use --date YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)

    try:
        scores_map = fetch_final_totals_for_date(target_date)
    except Exception as e:
        print(f"Error fetching game scores: {e}", file=sys.stderr)
        sys.exit(1)

    filled = 0
    for i in range(len(df)):
        game_val = df.at[i, "Game"]
        key = game_to_key(game_val)
        if not key or key not in scores_map:
            continue
        total = scores_map[key]
        df.at[i, "Final_Total"] = total
        filled += 1

    if "Final_Total" in df.columns:
        df["Final_Total"] = df["Final_Total"].astype(object)

    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error writing CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Filled Final_Total for {filled} rows in {csv_path}")


if __name__ == "__main__":
    main()

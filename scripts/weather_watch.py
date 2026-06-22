#!/usr/bin/env python3
"""
Pregame directional wind watcher for Over Gang MLB.

No CSV output.
No model math changes.
Reads latest predictions archive, checks games inside pregame window,
recomputes directional weather, and sends Telegram only on material changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH_DEFAULT = ROOT / "runtime" / "weather_watch_state.json"
MT = ZoneInfo("America/Denver")

# Alert thresholds
DEFAULT_WINDOW_MINUTES = 90
DEFAULT_MULT_DELTA = 0.006
GUST_THRESHOLDS_MPH = (10.0, 15.0)
COMPONENT_MODERATE_MPH = 5.0
COMPONENT_STRONG_MPH = 8.0

# Import weather logic from production weather module.
sys.path.insert(0, str(ROOT))
from core.weather_adjustment import (  # noqa: E402
    VENUE_LAT_LON,
    VENUE_CF_BEARING_DEG,
    compute_weather_runs_mult,
    _fetch_hourly_temp_wind_ms,
    _directional_wind_component,
    _fetch_mlb_weather_condition,
    _roof_state_blocks_outdoor_weather,
)

MPS_TO_MPH = 2.2369362921


def _parse_dt_utc(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _latest_predictions_archive(target_date: Optional[str] = None) -> Optional[Path]:
    archive = ROOT / "archive"
    files = sorted(archive.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    if target_date:
        needle = f"predictions_{target_date.replace('-', '')}_"
        dated = [p for p in files if p.name.startswith(needle)]
        if dated:
            return dated[0]

    today_mt = datetime.now(MT).strftime("%Y%m%d")
    needle = f"predictions_{today_mt}_"
    dated = [p for p in files if p.name.startswith(needle)]
    if dated:
        return dated[0]

    return files[0] if files else None


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"games": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"games": {}}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_telegram_credentials() -> Tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return token, chat

    # Fallback: parse assignments from model/overgang_model.py without importing it.
    path = ROOT / "model" / "overgang_model.py"
    if not path.exists():
        return token, chat

    text = path.read_text(encoding="utf-8", errors="replace")
    if not token:
        m = re.search(r"^\s*TELEGRAM_BOT_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if m:
            token = m.group(1).strip()
    if not chat:
        m = re.search(r"^\s*TELEGRAM_CHAT_ID\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if m:
            chat = m.group(1).strip()
    return token, chat


def _telegram_send(text: str) -> bool:
    token, chat = _load_telegram_credentials()
    if not token or not chat:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"Telegram send failed: {e}", file=sys.stderr)
        return False


def _gust_bucket(gust_mph: float) -> str:
    if gust_mph >= 15.0:
        return "15+"
    if gust_mph >= 10.0:
        return "10-15"
    return "<10"


def _wind_verdict(component_mph: Optional[float], gust_mph: float, has_bearing: bool) -> str:
    if not has_bearing:
        return "scalar_only"

    if component_mph is None:
        return "unknown"

    if gust_mph < 10.0:
        return "weak_wind"

    if component_mph >= COMPONENT_STRONG_MPH:
        return "wind_out_strong"
    if component_mph >= COMPONENT_MODERATE_MPH:
        return "wind_out_moderate"
    if component_mph <= -COMPONENT_STRONG_MPH:
        return "wind_in_strong"
    if component_mph <= -COMPONENT_MODERATE_MPH:
        return "wind_in_moderate"

    return "crosswind_or_neutral"


def _side_of_verdict(v: str) -> str:
    if v.startswith("wind_out"):
        return "OUT"
    if v.startswith("wind_in"):
        return "IN"
    if v == "roof_blocked":
        return "ROOF"
    return "NEUTRAL"


def _weather_snapshot(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    game = row.get("Game", "")
    venue = row.get("Venue", "")
    game_id = row.get("Game_ID", "") or row.get("game_id", "") or row.get("gamePk", "")
    dt_raw = row.get("Datetime", "")

    dt_utc = _parse_dt_utc(dt_raw)
    if dt_utc is None:
        return None

    coords = VENUE_LAT_LON.get(venue)
    if not coords:
        return {
            "game_id": str(game_id),
            "game": game,
            "venue": venue,
            "datetime": dt_raw,
            "verdict": "no_weather_coords",
            "weather_mult": 1.0,
        }

    roof_condition = None
    roof_blocked = False
    if game_id:
        roof_condition = _fetch_mlb_weather_condition(game_id)
        roof_blocked = _roof_state_blocks_outdoor_weather(roof_condition)

    if roof_blocked:
        return {
            "game_id": str(game_id),
            "game": game,
            "venue": venue,
            "datetime": dt_raw,
            "verdict": "roof_blocked",
            "weather_mult": 1.0,
            "roof_condition": roof_condition or "",
            "gust_mph": 0.0,
            "component_mph": 0.0,
            "wind_dir_from_deg": None,
        }

    lat, lon = coords
    tw = _fetch_hourly_temp_wind_ms(lat, lon, dt_utc)
    if tw is None:
        return None

    temp_c, wind_ms, gust_ms, wind_dir_from_deg = tw
    has_bearing = venue in VENUE_CF_BEARING_DEG
    comp_ms = _directional_wind_component(venue, wind_ms, gust_ms, wind_dir_from_deg)
    comp_mph = None if comp_ms is None else comp_ms * MPS_TO_MPH
    gust_mph = max(float(wind_ms), float(gust_ms)) * MPS_TO_MPH

    weather_mult = compute_weather_runs_mult(venue, dt_raw, game_pk=game_id)

    return {
        "game_id": str(game_id),
        "game": game,
        "venue": venue,
        "datetime": dt_raw,
        "verdict": _wind_verdict(comp_mph, gust_mph, has_bearing),
        "weather_mult": round(float(weather_mult), 6),
        "gust_mph": round(gust_mph, 1),
        "component_mph": None if comp_mph is None else round(float(comp_mph), 1),
        "wind_dir_from_deg": None if wind_dir_from_deg is None else round(float(wind_dir_from_deg), 1),
        "has_bearing": has_bearing,
        "roof_condition": roof_condition or "",
        "prediction": row.get("Prediction", ""),
        "ou_confidence": row.get("OU_Confidence", ""),
        "ou_fired": row.get("OU_Fired", ""),
        "baseline_weather_mult": row.get("Weather_Runs_Mult", ""),
    }


def _material_reasons(prev: Dict[str, Any], cur: Dict[str, Any], mult_delta: float) -> List[str]:
    reasons: List[str] = []

    prev_side = _side_of_verdict(str(prev.get("verdict", "")))
    cur_side = _side_of_verdict(str(cur.get("verdict", "")))

    if prev_side in ("OUT", "IN") and cur_side in ("OUT", "IN") and prev_side != cur_side:
        reasons.append(f"wind flipped {prev_side} → {cur_side}")

    if prev_side != cur_side and cur_side == "ROOF":
        reasons.append("roof/dome condition now blocks outdoor weather")
    if prev_side == "ROOF" and cur_side != "ROOF":
        reasons.append("roof/dome block no longer active")

    try:
        d = abs(float(cur.get("weather_mult", 1.0)) - float(prev.get("weather_mult", 1.0)))
        if d >= mult_delta:
            reasons.append(f"Weather_Runs_Mult changed {d:.4f}")
    except Exception:
        pass

    try:
        prev_bucket = _gust_bucket(float(prev.get("gust_mph", 0.0)))
        cur_bucket = _gust_bucket(float(cur.get("gust_mph", 0.0)))
        if prev_bucket != cur_bucket:
            reasons.append(f"gust bucket changed {prev_bucket} → {cur_bucket}")
    except Exception:
        pass

    prev_verdict = str(prev.get("verdict", ""))
    cur_verdict = str(cur.get("verdict", ""))
    if prev_verdict != cur_verdict and (cur_side in ("OUT", "IN") or prev_side in ("OUT", "IN")):
        reasons.append(f"wind verdict changed {prev_verdict} → {cur_verdict}")

    return reasons


def _format_alert(cur: Dict[str, Any], prev: Dict[str, Any], reasons: List[str], minutes_to_first_pitch: int) -> str:
    component = cur.get("component_mph")
    component_txt = "n/a" if component is None else f"{component:+.1f} mph"

    prev_component = prev.get("component_mph")
    prev_component_txt = "n/a" if prev_component is None else f"{prev_component:+.1f} mph"

    return (
        f"⚠️ WEATHER WATCH — {cur.get('game', '')}\n\n"
        f"First pitch in ~{minutes_to_first_pitch} min\n"
        f"Venue: {cur.get('venue', '')}\n\n"
        f"Reason: {'; '.join(reasons)}\n\n"
        f"Previous: {prev.get('verdict', '')} | gust {prev.get('gust_mph', 'n/a')} mph | component {prev_component_txt} | mult {prev.get('weather_mult', '')}\n"
        f"Now: {cur.get('verdict', '')} | gust {cur.get('gust_mph', 'n/a')} mph | component {component_txt} | mult {cur.get('weather_mult', '')}\n\n"
        f"Model side: {cur.get('prediction', '')} | Conf: {cur.get('ou_confidence', '')} | Fired: {cur.get('ou_fired', '')}\n\n"
        f"Recheck O/U exposure before lock.\n\n"
        f"**⚙️ Powered by Over Gang AI**"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="", help="Specific predictions archive path")
    ap.add_argument("--target-date", default="", help="YYYY-MM-DD slate date; defaults to MT today")
    ap.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
    ap.add_argument("--mult-delta", type=float, default=DEFAULT_MULT_DELTA)
    ap.add_argument("--state", default=str(STATE_PATH_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--alert-first", action="store_true", help="Alert on first observed snapshot; default only stores baseline")
    args = ap.parse_args()

    archive = Path(args.archive) if args.archive else _latest_predictions_archive(args.target_date or None)
    if archive is None or not archive.exists():
        print("No predictions archive found.")
        return 1

    rows = _read_rows(archive)
    state_path = Path(args.state)
    state = _load_state(state_path)
    games_state = state.setdefault("games", {})

    now = datetime.now(timezone.utc)
    checked = 0
    alerts = 0
    stored = 0

    print(f"Archive: {archive}")
    print(f"Now UTC: {now.isoformat()}")
    print(f"Window minutes: {args.window_minutes}")

    for row in rows:
        status = str(row.get("Game_Status", "")).strip().lower()
        if status and status not in ("scheduled", "pre-game", "pregame", "preview", "warmup"):
            continue

        dt_utc = _parse_dt_utc(row.get("Datetime", ""))
        if dt_utc is None:
            continue

        minutes_to_first_pitch = int((dt_utc - now).total_seconds() // 60)
        if minutes_to_first_pitch < 0:
            continue
        if minutes_to_first_pitch > args.window_minutes:
            continue

        cur = _weather_snapshot(row)
        if cur is None:
            continue

        checked += 1
        gid = cur.get("game_id") or f"{cur.get('game')}|{cur.get('datetime')}"
        prev = games_state.get(gid)

        if not prev:
            games_state[gid] = cur
            stored += 1
            if args.alert_first:
                reasons = ["first weather-watch snapshot"]
                msg = _format_alert(cur, {}, reasons, minutes_to_first_pitch)
                alerts += 1
                if args.dry_run:
                    print("\n--- DRY RUN ALERT ---")
                    print(msg)
                else:
                    _telegram_send(msg)
            continue

        reasons = _material_reasons(prev, cur, args.mult_delta)
        games_state[gid] = cur

        if reasons:
            msg = _format_alert(cur, prev, reasons, minutes_to_first_pitch)
            alerts += 1
            if args.dry_run:
                print("\n--- DRY RUN ALERT ---")
                print(msg)
            else:
                _telegram_send(msg)

    state["updated_at_utc"] = now.isoformat()
    state["archive"] = str(archive)
    _save_state(state_path, state)

    print(f"Checked games: {checked}")
    print(f"Stored baselines: {stored}")
    print(f"Alerts: {alerts}")
    print(f"State: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

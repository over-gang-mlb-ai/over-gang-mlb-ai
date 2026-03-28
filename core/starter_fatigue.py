# core/starter_fatigue.py
"""
Bounded starter fatigue overlay (Phase 2C v1): days of rest only.

Uses MLB Stats API pitching gameLog for the current season to find the pitcher's last
start before the scheduled game, then maps short rest to a small xERA bump on that starter
(so expected runs scored against them rise modestly). Does not replace season xERA or
interact with LowIP / velocity — those stay separate.

If rest cannot be resolved (unknown pitcher, API failure, first MLB start in season log),
returns 0.0 adjustment.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

import requests
import statsapi

logger = logging.getLogger(__name__)

# Hard cap on xERA bump from rest alone (keeps effect modest vs other signals).
MAX_XERA_BUMP_FROM_REST = 0.12

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OverGangPredictor/1.0)"}


def _parse_schedule_game_date(schedule_game_date: Optional[str]) -> Optional[date]:
    """
    MLB schedule `date` from statsapi (YYYY-MM-DD) — same convention as pitching gameLog `date`.
    Prefer this over UTC .date() from game_datetime so rest days align with official game dates.
    """
    if not schedule_game_date or not str(schedule_game_date).strip():
        return None
    try:
        return datetime.strptime(str(schedule_game_date).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_game_datetime_utc(game_datetime: Optional[str]) -> Optional[datetime]:
    if not game_datetime or not str(game_datetime).strip():
        return None
    s = str(game_datetime).strip()
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _lookup_pitcher_id(pitcher_name: str) -> Optional[int]:
    if not pitcher_name or not str(pitcher_name).strip():
        return None
    if "league avg" in str(pitcher_name).lower():
        return None
    try:
        hits = statsapi.lookup_player(str(pitcher_name).strip())
    except Exception as e:
        logger.debug("statsapi.lookup_player failed: %s", e)
        return None
    if not hits:
        return None
    pitchers = [p for p in hits if p.get("primaryPosition", {}).get("code") == "1"]
    if len(pitchers) >= 1:
        try:
            return int(pitchers[0]["id"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _last_start_date_in_season_before(
    player_id: int, season: int, game_date: date
) -> Optional[date]:
    """Latest game `date` in season gameLog strictly before game_date (date objects)."""
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}/stats",
            params={
                "stats": "gameLog",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("gameLog fetch failed for player %s: %s", player_id, e)
        return None

    stats = data.get("stats") or []
    if not stats:
        return None
    splits = stats[0].get("splits") or []
    prior = []
    for sp in splits:
        d = sp.get("date")
        if not d:
            continue
        try:
            gd = datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if gd < game_date:
            prior.append(gd)
    if not prior:
        return None
    return max(prior)


def _days_rest(game_date: date, last_start_date: date) -> int:
    """Calendar off-days between last start and this game (MLB-style spacing)."""
    diff = (game_date - last_start_date).days
    return max(0, diff - 1)


def _rest_days_to_xera_delta(rest_days: Optional[int]) -> float:
    """Conservative tiers; 5+ days normal → no bump. Capped at MAX_XERA_BUMP_FROM_REST."""
    if rest_days is None:
        return 0.0
    if rest_days >= 5:
        return 0.0
    if rest_days == 4:
        return min(0.025, MAX_XERA_BUMP_FROM_REST)
    if rest_days == 3:
        return min(0.06, MAX_XERA_BUMP_FROM_REST)
    # 0–2 days (short rest / quick turnaround)
    return min(0.10, MAX_XERA_BUMP_FROM_REST)


def xera_delta_for_pitcher_days_rest(
    pitcher_name: Optional[str],
    game_datetime: Optional[str],
    schedule_game_date: Optional[str] = None,
) -> float:
    """
    Return a small non-negative xERA increment for this starter based on days rest before
    the scheduled game. 0.0 when unavailable or on normal rest (5+ days).

    v1 uses current-season gameLog only (no cross-season prior start).

    schedule_game_date: optional YYYY-MM-DD from statsapi schedule `game_date` (same bucket as
    gameLog `date`); when set, used as the game calendar date instead of UTC midnight from
    game_datetime.
    """
    game_date = _parse_schedule_game_date(schedule_game_date)
    if game_date is None:
        dt = _parse_game_datetime_utc(game_datetime)
        if dt is None:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        game_date = dt.date()
    season = int(game_date.year)

    pid = _lookup_pitcher_id(pitcher_name or "")
    if pid is None:
        return 0.0

    last_start = _last_start_date_in_season_before(pid, season, game_date)
    if last_start is None:
        return 0.0

    rest = _days_rest(game_date, last_start)
    delta = _rest_days_to_xera_delta(rest)
    # Temporary validation: set STARTER_FATIGUE_DEBUG=1 to log nonzero rest bumps only.
    if delta > 0 and os.environ.get("STARTER_FATIGUE_DEBUG", "").strip() == "1":
        print(f"[STARTER REST] {pitcher_name!r} rest={rest}d xERA+{delta:.3f}")
    return delta

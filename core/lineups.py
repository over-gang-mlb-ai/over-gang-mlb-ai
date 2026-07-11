"""
Confirmed MLB starting-lineup loader.

Canonical source:
    MLB StatsAPI live feed keyed by gamePk.

This module is read-only. It does not modify projections, archives, alerts,
or model state. A lineup is confirmed only when all nine original batting
slots are present with unique MLB player IDs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

import requests


MLB_LIVE_FEED_URL = (
    "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None


def _empty_result(
    game_pk: Optional[int],
    *,
    error: str = "",
    status: str = "",
) -> Dict[str, Any]:
    return {
        "game_pk": game_pk,
        "source": "mlb_live_feed",
        "fetched_at_utc": _utc_now_iso(),
        "status": status,
        "away_lineup": [],
        "home_lineup": [],
        "away_confirmed": False,
        "home_confirmed": False,
        "both_confirmed": False,
        "away_signature": "",
        "home_signature": "",
        "error": error,
    }


def lineup_signature(lineup: List[Dict[str, Any]]) -> str:
    """
    Stable identity for a posted lineup.

    A late scratch or batting-order change produces a different signature.
    """
    ordered = sorted(
        lineup,
        key=lambda player: int(player.get("slot", 0)),
    )

    return "|".join(
        f"{int(player['slot'])}:{int(player['player_id'])}"
        for player in ordered
        if _safe_int(player.get("slot"))
        and _safe_int(player.get("player_id"))
    )


def _extract_original_starting_lineup(
    team_node: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract the original hitter assigned to each batting slot.

    MLB battingOrder examples:
        100 = original first hitter
        101 = later replacement in first slot
        200 = original second hitter

    Choosing the lowest battingOrder code per slot preserves the original
    posted starting lineup if the feed is inspected after substitutions.
    """
    players = team_node.get("players") or {}
    slots: Dict[int, Dict[str, Any]] = {}

    for player in players.values():
        order_code = _safe_int(player.get("battingOrder"))

        if order_code is None:
            continue

        slot = order_code // 100

        if slot < 1 or slot > 9:
            continue

        person = player.get("person") or {}
        player_id = _safe_int(person.get("id"))
        name = str(
            person.get("fullName")
            or person.get("boxscoreName")
            or ""
        ).strip()

        if player_id is None or not name:
            continue

        candidate = {
            "slot": slot,
            "order_code": order_code,
            "player_id": player_id,
            "name": name,
        }

        current = slots.get(slot)

        if (
            current is None
            or order_code < int(current["order_code"])
        ):
            slots[slot] = candidate

    return [slots[slot] for slot in sorted(slots)]


def _is_complete_confirmed_lineup(
    lineup: List[Dict[str, Any]],
) -> bool:
    if len(lineup) != 9:
        return False

    slots = [int(player.get("slot", 0)) for player in lineup]
    player_ids = [
        _safe_int(player.get("player_id"))
        for player in lineup
    ]

    return (
        slots == list(range(1, 10))
        and None not in player_ids
        and len(set(player_ids)) == 9
    )


def parse_confirmed_lineups(
    feed: Dict[str, Any],
    game_pk: int,
) -> Dict[str, Any]:
    """
    Parse one MLB live-feed payload without making a network request.
    """
    parsed_game_pk = _safe_int(game_pk)

    if parsed_game_pk is None:
        return _empty_result(
            None,
            error="invalid_game_pk",
        )

    status = str(
        (
            feed.get("gameData", {})
            .get("status", {})
            .get("detailedState")
        )
        or ""
    ).strip()

    box_teams = (
        feed.get("liveData", {})
        .get("boxscore", {})
        .get("teams", {})
        or {}
    )

    away_lineup = _extract_original_starting_lineup(
        box_teams.get("away") or {}
    )
    home_lineup = _extract_original_starting_lineup(
        box_teams.get("home") or {}
    )

    away_confirmed = _is_complete_confirmed_lineup(
        away_lineup
    )
    home_confirmed = _is_complete_confirmed_lineup(
        home_lineup
    )

    # Fail closed per side. Partial lineups are useful for telemetry but
    # must never be treated as confirmed projection inputs.
    return {
        "game_pk": parsed_game_pk,
        "source": "mlb_live_feed",
        "fetched_at_utc": _utc_now_iso(),
        "status": status,
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "away_confirmed": away_confirmed,
        "home_confirmed": home_confirmed,
        "both_confirmed": (
            away_confirmed and home_confirmed
        ),
        "away_signature": (
            lineup_signature(away_lineup)
            if away_confirmed
            else ""
        ),
        "home_signature": (
            lineup_signature(home_lineup)
            if home_confirmed
            else ""
        ),
        "error": "",
    }


def fetch_confirmed_lineups(
    game_pk: Any,
    *,
    timeout: float = 20.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Fetch and parse confirmed lineups for one MLB gamePk.

    Network, HTTP, JSON, and incomplete-lineup failures all fail closed.
    """
    parsed_game_pk = _safe_int(game_pk)

    if parsed_game_pk is None:
        return _empty_result(
            None,
            error="invalid_game_pk",
        )

    client = session or requests

    try:
        response = client.get(
            MLB_LIVE_FEED_URL.format(
                game_pk=parsed_game_pk
            ),
            timeout=timeout,
        )
        response.raise_for_status()
        feed = response.json()
    except Exception as exc:
        logging.warning(
            "⚠️ Confirmed lineup fetch failed for gamePk=%s: %s",
            parsed_game_pk,
            exc,
        )
        return _empty_result(
            parsed_game_pk,
            error=f"fetch_failed:{type(exc).__name__}",
        )

    try:
        return parse_confirmed_lineups(
            feed,
            parsed_game_pk,
        )
    except Exception as exc:
        logging.warning(
            "⚠️ Confirmed lineup parse failed for gamePk=%s: %s",
            parsed_game_pk,
            exc,
        )
        return _empty_result(
            parsed_game_pk,
            error=f"parse_failed:{type(exc).__name__}",
        )

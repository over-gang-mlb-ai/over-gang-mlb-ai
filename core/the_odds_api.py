"""
The Odds API ingestion (event-level F5 totals).

Bulk endpoint /v4/sports/baseball_mlb/odds does NOT expose 1st-5-innings markets
(totals_1st_5_innings / h2h_3_way_1st_5_innings / spreads_1st_5_innings); they
are only available at the per-event endpoint:

    GET /v4/sports/baseball_mlb/events/{event_id}/odds

This module is the canonical owner of THE_ODDS_API_KEY usage. It is intentionally
separate from core/odds_api.py (which is Parlay-owned, ODDS_API_KEY). It exposes
only F5 totals today; no fire / Kelly / Telegram logic lives here.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from core.public_betting_loader import normalize_team_name

logger = logging.getLogger(__name__)

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
F5_TOTALS_MARKET = "totals_1st_5_innings"
OU_TOTALS_MARKET = "totals"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"

# Order matters: first qualifying book wins. Pinnacle first for sharp anchor; then
# major US books most likely to post F5 totals.
PREFERRED_BOOKS: List[str] = [
    "pinnacle",
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "bet365",
    "mybookieag",
]

SOURCE_LABEL = "the_odds_api"
_REQUEST_TIMEOUT_S = 15


def _get_api_key() -> Optional[str]:
    """Return THE_ODDS_API_KEY from env, whitespace-stripped, or None when unset/empty.

    Intentionally does NOT read ODDS_API_KEY (that is Parlay).
    """
    raw = os.getenv("THE_ODDS_API_KEY")
    if raw is None:
        return None
    key = str(raw).strip()
    return key or None


def fetch_mlb_events() -> List[Dict[str, Any]]:
    """GET /v4/sports/baseball_mlb/events — list of MLB events with event ids.

    Returns [] on any error (missing key, network failure, malformed body).
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("THE_ODDS_API_KEY not set; fetch_mlb_events returning [].")
        return []
    url = f"{THE_ODDS_API_BASE}/events"
    params = {"apiKey": api_key}
    try:
        r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("the_odds_api fetch_mlb_events failed: %s", e)
        return []
    if not isinstance(body, list):
        logger.warning("the_odds_api fetch_mlb_events unexpected payload type: %s", type(body).__name__)
        return []
    return body


def fetch_event_f5_odds(event_id: str) -> Optional[Dict[str, Any]]:
    """GET /v4/sports/baseball_mlb/events/{event_id}/odds for totals_1st_5_innings.

    Returns the parsed event dict (with bookmakers/markets/outcomes) or None on failure.
    """
    if not event_id or not str(event_id).strip():
        return None
    api_key = _get_api_key()
    if not api_key:
        logger.warning("THE_ODDS_API_KEY not set; fetch_event_f5_odds(%s) returning None.", event_id)
        return None
    url = f"{THE_ODDS_API_BASE}/events/{str(event_id).strip()}/odds"
    params = {
        "apiKey": api_key,
        "markets": F5_TOTALS_MARKET,
        "regions": DEFAULT_REGIONS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
    }
    try:
        r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("the_odds_api fetch_event_f5_odds(%s) failed: %s", event_id, e)
        return None
    if not isinstance(body, dict):
        return None
    return body


def _parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_point(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_preferred_f5_book(event_odds: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the first PREFERRED_BOOKS entry exposing a valid F5 totals market.

    Required for a book to qualify:
      - market.key == totals_1st_5_innings
      - both Over and Under outcomes present
      - same point on both sides
      - both prices parseable as numbers
    """
    if not isinstance(event_odds, dict):
        return None
    bookmakers = event_odds.get("bookmakers") or []
    if not isinstance(bookmakers, list):
        return None

    by_key: Dict[str, Dict[str, Any]] = {}
    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        key = (bm.get("key") or "").strip().lower()
        if key:
            by_key.setdefault(key, bm)

    for pref_key in PREFERRED_BOOKS:
        bm = by_key.get(pref_key)
        if bm is None:
            continue
        markets = bm.get("markets") or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            if (market.get("key") or "").strip().lower() != F5_TOTALS_MARKET:
                continue
            outcomes = market.get("outcomes") or []
            if not isinstance(outcomes, list):
                continue
            over_outcome = None
            under_outcome = None
            for oc in outcomes:
                if not isinstance(oc, dict):
                    continue
                name = (oc.get("name") or "").strip().lower()
                if name == "over":
                    over_outcome = oc
                elif name == "under":
                    under_outcome = oc
            if over_outcome is None or under_outcome is None:
                continue
            over_point = _parse_point(over_outcome.get("point"))
            under_point = _parse_point(under_outcome.get("point"))
            if over_point is None or under_point is None or over_point != under_point:
                continue
            over_price = _parse_price(over_outcome.get("price"))
            under_price = _parse_price(under_outcome.get("price"))
            if over_price is None or under_price is None:
                continue
            return {
                "f5_total_line": over_point,
                "f5_over_juice": over_price,
                "f5_under_juice": under_price,
                "f5_book": pref_key,
                "f5_source": SOURCE_LABEL,
                "f5_last_update": (
                    market.get("last_update") or bm.get("last_update") or ""
                ),
                "f5_market_ok": True,
            }
    return None


def _event_game_key(event: Dict[str, Any]) -> Optional[str]:
    if not isinstance(event, dict):
        return None
    away_raw = event.get("away_team") or ""
    home_raw = event.get("home_team") or ""
    try:
        away_norm = normalize_team_name(str(away_raw))
        home_norm = normalize_team_name(str(home_raw))
    except Exception:
        return None
    if not away_norm or not home_norm:
        return None
    return f"{away_norm} @ {home_norm}"


def fetch_ou_totals_by_game() -> Dict[str, List[Dict[str, Any]]]:
    """Build mapping `'<away_norm> @ <home_norm>' -> list[full-game totals quote]`.

    Uses The Odds API bulk full-game totals endpoint:
        GET /v4/sports/baseball_mlb/odds?markets=totals&regions=us&oddsFormat=american

    Fail-soft: returns {} on missing key, request failure, malformed JSON, or unusable body.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("THE_ODDS_API_KEY not set; fetch_ou_totals_by_game returning {}.")
        return {}

    url = f"{THE_ODDS_API_BASE}/odds"
    params = {
        "apiKey": api_key,
        "markets": OU_TOTALS_MARKET,
        "regions": DEFAULT_REGIONS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
    }
    try:
        r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("the_odds_api fetch_ou_totals_by_game failed: %s", e)
        return {}
    if not isinstance(body, list):
        logger.warning(
            "the_odds_api fetch_ou_totals_by_game unexpected payload type: %s",
            type(body).__name__,
        )
        return {}

    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for event in body:
        if not isinstance(event, dict):
            continue
        game_key = _event_game_key(event)
        if not game_key:
            continue
        bookmakers = event.get("bookmakers") or []
        if not isinstance(bookmakers, list):
            continue

        for bm in bookmakers:
            if not isinstance(bm, dict):
                continue
            book = (bm.get("key") or "").strip().lower()
            if not book:
                continue
            markets = bm.get("markets") or []
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue
                if (market.get("key") or "").strip().lower() != OU_TOTALS_MARKET:
                    continue
                outcomes = market.get("outcomes") or []
                if not isinstance(outcomes, list):
                    continue

                over_outcome = None
                under_outcome = None
                for oc in outcomes:
                    if not isinstance(oc, dict):
                        continue
                    name = (oc.get("name") or "").strip().lower()
                    if name == "over":
                        over_outcome = oc
                    elif name == "under":
                        under_outcome = oc
                if over_outcome is None or under_outcome is None:
                    continue

                over_point = _parse_point(over_outcome.get("point"))
                under_point = _parse_point(under_outcome.get("point"))
                if over_point is None or under_point is None or over_point != under_point:
                    continue
                over_price = _parse_price(over_outcome.get("price"))
                under_price = _parse_price(under_outcome.get("price"))
                if over_price is None or under_price is None:
                    continue

                mapping.setdefault(game_key, []).append({
                    "book": book,
                    "total_line": over_point,
                    "over_juice": over_price,
                    "under_juice": under_price,
                    "last_update": market.get("last_update") or bm.get("last_update") or "",
                    "source": SOURCE_LABEL,
                })

    return mapping


def fetch_f5_totals_by_game() -> Dict[str, Dict[str, Any]]:
    """Build mapping `'<away_norm> @ <home_norm>' -> f5 market dict` from The Odds API.

    Iterates fetch_mlb_events(), calls fetch_event_f5_odds() per event id, selects the
    first preferred book with a valid F5 totals market, and emits one entry per event.

    Each value dict carries:
        f5_total_line, f5_over_juice, f5_under_juice, f5_book, f5_source,
        f5_last_update, f5_market_ok, f5_event_id, f5_commence_time

    Events with no qualifying F5 market are omitted (caller treats absence as
    no_f5_market_source).

    Doubleheader / rescheduled-game safety: if a normalized game key ("<away> @ <home>")
    is produced by more than one event, ALL entries for that key are dropped and a
    warning is logged. The caller then sees no F5 market for that game and keeps
    F5_No_Line_Reason = "no_f5_market_source". This avoids silently attaching the wrong
    F5 line to the wrong game when /events returns multiple events with the same teams.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    ambiguous_keys: set = set()
    key_to_first_event: Dict[str, Dict[str, Any]] = {}

    events = fetch_mlb_events()
    if not events:
        return mapping

    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("id") or ev.get("event_id")
        if not event_id:
            continue
        game_key = _event_game_key(ev)
        if not game_key:
            continue
        event_odds = fetch_event_f5_odds(str(event_id))
        if event_odds is None:
            continue
        selected = _select_preferred_f5_book(event_odds)
        if selected is None:
            continue

        commence_time = ev.get("commence_time") or ""
        enriched = dict(selected)
        enriched["f5_event_id"] = str(event_id)
        enriched["f5_commence_time"] = commence_time

        if game_key in ambiguous_keys:
            logger.warning(
                "the_odds_api duplicate F5 event key skipped: %s "
                "(additional event_id=%s commence_time=%s)",
                game_key, event_id, commence_time,
            )
            continue

        if game_key in mapping:
            prev = key_to_first_event.get(game_key, {})
            logger.warning(
                "the_odds_api duplicate F5 event key skipped: %s "
                "(first event_id=%s commence_time=%s; second event_id=%s commence_time=%s)",
                game_key,
                prev.get("event_id", ""), prev.get("commence_time", ""),
                event_id, commence_time,
            )
            ambiguous_keys.add(game_key)
            mapping.pop(game_key, None)
            key_to_first_event.pop(game_key, None)
            continue

        mapping[game_key] = enriched
        key_to_first_event[game_key] = {
            "event_id": str(event_id),
            "commence_time": commence_time,
        }

    return mapping

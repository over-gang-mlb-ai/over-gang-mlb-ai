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
PITCHER_STRIKEOUTS_MARKET = "pitcher_strikeouts"
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

FULL_GAME_BOOK_PRIORITY: List[str] = [
    "fanduel",
    "draftkings",
    "betmgm",
    "betrivers",
    "lowvig",
    "betonlineag",
    "mybookieag",
    "bovada",
]


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


def fetch_event_pitcher_strikeout_odds(event_id: str) -> Optional[Dict[str, Any]]:
    """GET /v4/sports/baseball_mlb/events/{event_id}/odds for pitcher_strikeouts.

    Returns the parsed event dict with bookmakers/markets/outcomes, or None on failure.
    """
    if not event_id or not str(event_id).strip():
        return None
    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            "THE_ODDS_API_KEY not set; fetch_event_pitcher_strikeout_odds(%s) returning None.",
            event_id,
        )
        return None

    url = f"{THE_ODDS_API_BASE}/events/{str(event_id).strip()}/odds"
    params = {
        "apiKey": api_key,
        "markets": PITCHER_STRIKEOUTS_MARKET,
        "regions": DEFAULT_REGIONS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
    }
    try:
        r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("the_odds_api fetch_event_pitcher_strikeout_odds(%s) failed: %s", event_id, e)
        return None

    if not isinstance(body, dict):
        return None
    return body


def _extract_pitcher_strikeout_props_from_event_odds(event_odds: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize The Odds API pitcher_strikeouts event odds into existing K prop rows.

    The Odds API player prop outcomes use:
      name = Over/Under
      description = pitcher name
      point = strikeout line
      price = American odds

    Returned rows intentionally match the existing Parlay-shaped fields consumed by
    model/overgang_model.py. Final attachment remains gated by slate starter name.
    """
    if not isinstance(event_odds, dict):
        return []

    event_id = str(event_odds.get("id") or event_odds.get("event_id") or "").strip()
    rows: List[Dict[str, Any]] = []

    bookmakers = event_odds.get("bookmakers") or []
    if not isinstance(bookmakers, list):
        return rows

    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        book_key = str(bm.get("key") or "").strip().lower()
        if not book_key:
            continue

        markets = bm.get("markets") or []
        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict):
                continue
            if str(market.get("key") or "").strip().lower() != PITCHER_STRIKEOUTS_MARKET:
                continue

            grouped: Dict[tuple, Dict[str, Any]] = {}
            last_update = (
                market.get("last_update")
                or bm.get("last_update")
                or event_odds.get("last_update")
                or ""
            )

            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue

                side = str(outcome.get("name") or "").strip().lower()
                if side not in {"over", "under"}:
                    continue

                player = (
                    str(outcome.get("description") or "").strip()
                    or str(outcome.get("player") or "").strip()
                    or str(outcome.get("participant") or "").strip()
                )
                line = _parse_point(outcome.get("point"))
                price = _parse_american_price(outcome.get("price"))

                if not player or line is None or price is None:
                    continue

                key = (player.lower(), float(line))
                grouped.setdefault(
                    key,
                    {
                        "player": player,
                        "line": float(line),
                        "over_price": None,
                        "under_price": None,
                    },
                )
                grouped[key][f"{side}_price"] = price

            for item in grouped.values():
                if item.get("over_price") is None or item.get("under_price") is None:
                    continue
                rows.append(
                    {
                        "source": SOURCE_LABEL,
                        "player": item["player"],
                        "market_key": "player_strikeouts",
                        "market": "pitcher strikeouts",
                        "line": item["line"],
                        "over_price": item["over_price"],
                        "under_price": item["under_price"],
                        "bookmaker": book_key,
                        "last_update": last_update,
                        "canonical_event_id": event_id,
                    }
                )

    return rows


def fetch_pitcher_strikeout_props() -> List[Dict[str, Any]]:
    """Fallback fetch for MLB pitcher strikeout props from The Odds API.

    Uses /events to get event IDs, then /events/{event_id}/odds with market
    pitcher_strikeouts. Call only as fallback because event-level prop odds can
    consume quota.
    """
    events = fetch_mlb_events()
    if not events:
        return []

    rows: List[Dict[str, Any]] = []
    checked = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue

        checked += 1
        event_odds = fetch_event_pitcher_strikeout_odds(event_id)
        rows.extend(_extract_pitcher_strikeout_props_from_event_odds(event_odds))

    logger.info(
        "the_odds_api pitcher_strikeouts fallback checked %s event(s), parsed %s prop row(s).",
        checked,
        len(rows),
    )
    return rows


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


def _parse_american_price(value: Any) -> Optional[int]:
    price = _parse_price(value)
    if price is None:
        return None
    return int(price)


def _canonical_commence_time(commence_time: Any) -> str:
    if not commence_time:
        return ""
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(commence_time).strip()


def _parse_totals_market(book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(book, dict):
        return None
    for market in book.get("markets") or []:
        if not isinstance(market, dict):
            continue
        if (market.get("key") or "").strip().lower() != OU_TOTALS_MARKET:
            continue
        over_outcome = None
        under_outcome = None
        for outcome in market.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            name = (outcome.get("name") or "").strip().lower()
            if name == "over":
                over_outcome = outcome
            elif name == "under":
                under_outcome = outcome
        if over_outcome is None or under_outcome is None:
            continue
        over_point = _parse_point(over_outcome.get("point"))
        under_point = _parse_point(under_outcome.get("point"))
        over_price = _parse_american_price(over_outcome.get("price"))
        under_price = _parse_american_price(under_outcome.get("price"))
        if (
            over_point is None
            or under_point is None
            or over_point != under_point
            or over_price is None
            or under_price is None
        ):
            continue
        return {
            "total_line": over_point,
            "over_juice": over_price,
            "under_juice": under_price,
        }
    return None


def _parse_h2h_market(book: Dict[str, Any], home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
    if not isinstance(book, dict):
        return None
    home_norm = normalize_team_name(home_team or "")
    away_norm = normalize_team_name(away_team or "")
    ml_home = None
    ml_away = None
    for market in book.get("markets") or []:
        if not isinstance(market, dict):
            continue
        if (market.get("key") or "").strip().lower() != "h2h":
            continue
        for outcome in market.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            outcome_norm = normalize_team_name(str(outcome.get("name") or ""))
            price = _parse_american_price(outcome.get("price"))
            if price is None:
                continue
            if outcome_norm == home_norm:
                ml_home = price
            elif outcome_norm == away_norm:
                ml_away = price
    if ml_home is None or ml_away is None:
        return None
    return {"ml_home": ml_home, "ml_away": ml_away}


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


def fetch_full_game_odds_map() -> Dict[str, Dict[str, Any]]:
    """Fetch The Odds API full-game h2h + totals in core.odds_api-compatible shape.

    Intended as a fail-soft fallback when the Parlay main odds feed has zero usable
    scheduled-game real totals. Does not invent default totals/prices.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("THE_ODDS_API_KEY not set; fetch_full_game_odds_map returning {}.")
        return {}

    url = f"{THE_ODDS_API_BASE}/odds"
    params = {
        "apiKey": api_key,
        "markets": "h2h,totals",
        "regions": DEFAULT_REGIONS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
    }
    try:
        r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning("the_odds_api fetch_full_game_odds_map failed: %s", e)
        return {}
    if not isinstance(body, list):
        logger.warning(
            "the_odds_api fetch_full_game_odds_map unexpected payload type: %s",
            type(body).__name__,
        )
        return {}

    odds_map: Dict[str, Dict[str, Any]] = {}
    book_rank = {book: i for i, book in enumerate(FULL_GAME_BOOK_PRIORITY)}
    for event in body:
        if not isinstance(event, dict):
            continue
        away_team = str(event.get("away_team") or "").strip()
        home_team = str(event.get("home_team") or "").strip()
        if not away_team or not home_team:
            continue
        coarse_key = _event_game_key(event)
        if not coarse_key:
            continue
        commence_time = _canonical_commence_time(event.get("commence_time"))
        event_key = f"{coarse_key} @@ {commence_time}" if commence_time else coarse_key
        bookmakers = event.get("bookmakers") or []
        if not isinstance(bookmakers, list):
            continue

        totals_choice = None
        totals_book = ""
        ml_choice = None
        ml_book = ""
        for book in sorted(
            [b for b in bookmakers if isinstance(b, dict)],
            key=lambda b: (
                book_rank.get((b.get("key") or "").strip().lower(), len(FULL_GAME_BOOK_PRIORITY)),
                (b.get("key") or "").strip().lower(),
            ),
        ):
            book_key = (book.get("key") or "").strip().lower()
            if book_key not in book_rank:
                continue
            if totals_choice is None:
                parsed_totals = _parse_totals_market(book)
                if parsed_totals is not None:
                    totals_choice = parsed_totals
                    totals_book = book_key
            if ml_choice is None:
                parsed_ml = _parse_h2h_market(book, home_team, away_team)
                if parsed_ml is not None:
                    ml_choice = parsed_ml
                    ml_book = book_key
            if totals_choice is not None and ml_choice is not None:
                break

        if totals_choice is None and ml_choice is None:
            continue

        row = {
            "total_line": None,
            "over_juice": None,
            "under_juice": None,
            "ml_home": None,
            "ml_away": None,
            "book": totals_book,
            "ml_book": ml_book,
            "_coarse_game_key": coarse_key,
            "_commence_time": commence_time,
            "_source": SOURCE_LABEL,
        }
        if totals_choice is not None:
            row.update(totals_choice)
        if ml_choice is not None:
            row.update(ml_choice)
        odds_map[event_key] = row

    return odds_map


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

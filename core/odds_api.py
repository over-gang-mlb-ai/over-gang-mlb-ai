"""
Parlay API integration for MLB.
Loads ODDS_API_KEY from .env; fetches totals + moneylines (American odds).
Prefer sharp book first; return safe fallbacks when unavailable.
Target-date filtering: keep events whose commence_time maps to target_date (YYYY-MM-DD) using the same
04:00 Mountain Time slate rollover as the predictor (_event_slate_date_mt), not strict calendar date in MT.
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
MLB_SPORT_KEY = "baseball_mlb"
ODDS_BASE_URL = "https://parlay-api.com/v1"
# Single HTTP GET timeout (seconds). One automatic retry on transient Parlay API failures.
PARLAY_FETCH_TIMEOUT_SEC = 25
# Prefer sharp / liquid books first; fall back to first available
BOOK_PRIORITY = (
    "pinnacle",
    "lowvig",  # if present in Odds API
    "betonlineag",
    "draftkings",
    "fanduel",
    "betrivers",
    "betmgm",
    "pointsbetus",
    "williamhill_us",
    "bovada",
)

DEFAULT_TOTAL = 8.5
DEFAULT_JUICE = -110
MT_ZONE = "America/Denver"


def _event_date_mt(commence_time_str):
    """
    Parse API commence_time (ISO UTC) and return date string YYYY-MM-DD in Mountain Time.
    Returns "" if parse fails.
    """
    if not commence_time_str:
        return ""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        mt = dt.astimezone(ZoneInfo(MT_ZONE))
        return mt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _event_slate_date_mt(commence_time_str):
    """
    Parse API commence_time (ISO UTC), convert to Mountain Time, then map to active slate calendar
    date (04:00 MT rollover): local times before 04:00 belong to the previous calendar day.
    Returns YYYY-MM-DD, or "" if parse fails.
    """
    if not commence_time_str:
        return ""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(str(commence_time_str).replace("Z", "+00:00"))
        mt = dt.astimezone(ZoneInfo(MT_ZONE))
        d = mt.date()
        if mt.hour < 4:
            d = d - timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _commence_window_utc(target_date_yyyy_mm_dd):
    """
    For a given date in Mountain Time (YYYY-MM-DD), return (from_utc, to_utc) in ISO format
    for the API commenceTimeFrom / commenceTimeTo params. Covers full calendar day in MT.
    Returns (None, None) if target_date is invalid.
    """
    if not target_date_yyyy_mm_dd or len(target_date_yyyy_mm_dd) != 10:
        return None, None
    try:
        from zoneinfo import ZoneInfo
        y, m, d = int(target_date_yyyy_mm_dd[:4]), int(target_date_yyyy_mm_dd[5:7]), int(target_date_yyyy_mm_dd[8:10])
        day_start_mt = datetime(y, m, d, 0, 0, 0, tzinfo=ZoneInfo(MT_ZONE))
        day_end_mt = datetime(y, m, d, 23, 59, 59, tzinfo=ZoneInfo(MT_ZONE))
        utc_start = day_start_mt.astimezone(ZoneInfo("UTC"))
        utc_end = day_end_mt.astimezone(ZoneInfo("UTC"))
        from_utc = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_utc = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        return from_utc, to_utc
    except Exception:
        return None, None


def _game_key(away_team, home_team):
    """Build key for lookup: normalized 'away @ home' (match public_betting_loader style)."""
    try:
        from core.public_betting_loader import normalize_team_name
        a = normalize_team_name(away_team or "")
        h = normalize_team_name(home_team or "")
        return f"{a} @ {h}"
    except Exception:
        return f"{(away_team or '').lower().strip()} @ {(home_team or '').lower().strip()}"


def _canonical_commence_time(commence_time_str):
    """Normalize an ISO-like UTC time to YYYY-MM-DDTHH:MM:SSZ, else ''."""
    if not commence_time_str:
        return ""
    try:
        dt = datetime.fromisoformat(str(commence_time_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(commence_time_str).strip()


# Substrings for Japanese / non-MLB clubs occasionally mixed into Parlay MLB feeds.
_PARLAY_NON_MLB_NAME_MARKERS = (
    "yomiuri",
    "chunichi",
    "orix",
    "nippon-ham",
    "nippon ham",
    "hiroshima toyo",
    "yakult",
    "hokkaido nippon",
    "tokyo yakult",
    "fukuoka softbank",
    "saitama seibu",
    "tohoku rakuten",
    "chiba lotte",
)


def _american_odds_int(val) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _parlay_row_excludes_non_mlb_teams(away_team: str, home_team: str) -> bool:
    blob = f"{away_team or ''} {home_team or ''}".lower()
    for frag in _PARLAY_NON_MLB_NAME_MARKERS:
        if frag in blob:
            return True
    if "nippon" in blob and "fighters" in blob:
        return True
    return False


def _is_parlay_flat_event_row(ev: dict) -> bool:
    """Parlay flat schema: team/total/commence on the row; no nested bookmakers."""
    if not isinstance(ev, dict):
        return False
    if ev.get("bookmakers"):
        return False
    if not (ev.get("home_team") or "").strip() or not (ev.get("away_team") or "").strip():
        return False
    if ev.get("total") is None or ev.get("total") == "":
        return False
    if not (ev.get("commence_time") or "").strip():
        return False
    return True


def _event_key(away_team, home_team, commence_time_str=None):
    coarse = _game_key(away_team, home_team)
    ct = _canonical_commence_time(commence_time_str)
    return f"{coarse} @@ {ct}" if coarse and ct else coarse


def _h2h_name_matches(outcome_name, event_team):
    """True when an h2h outcome `name` refers to `event_team`, tolerant of provider
    abbreviations / aliases (e.g. "ARI Diamondbacks" vs "Arizona Diamondbacks",
    "CHI White Sox" vs "Chicago White Sox", "Braves" vs "Atlanta Braves").

    Matching order (each rejects fast on empty inputs; first match wins):
      1) byte-equal (preserves original behavior for exact matches).
      2) case- and whitespace-collapsed equality.
      3) alias normalization on both sides via core.public_betting_loader
         .normalize_team_name (direct lookup + partial-substring fallback).
      4) bounded token-sweep: every consecutive-token substring of the outcome
         name is tested as a direct TEAM_ALIASES key; if any resolves to the
         same canonical full name as the event team, it matches. This handles
         leading-code abbreviations like "ARI Diamondbacks" where neither full
         string is an alias key but "diamondbacks" alone is.

    Does not invent prices. Does not modify caller state. Returns bool.
    """
    if not outcome_name or not event_team:
        return False
    a = str(outcome_name).strip()
    b = str(event_team).strip()
    if a == b:
        return True
    a_lc = " ".join(a.lower().split())
    b_lc = " ".join(b.lower().split())
    if a_lc and a_lc == b_lc:
        return True
    try:
        from core.public_betting_loader import normalize_team_name, TEAM_ALIASES
    except Exception:
        return False
    b_norm = normalize_team_name(b_lc) if b_lc else ""
    if not b_norm:
        return False
    a_norm = normalize_team_name(a_lc) if a_lc else ""
    if a_norm and a_norm == b_norm:
        return True
    tokens = a_lc.split()
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            frag = " ".join(tokens[start:end])
            if not frag:
                continue
            resolved = TEAM_ALIASES.get(frag)
            if resolved and resolved == b_norm:
                return True
    return False


def _parse_totals_and_h2h(book):
    """From one bookmaker, extract totals (line, over_juice, under_juice) and h2h (ml_home, ml_away).

    None-safe for `price` and `point`: Parlay (exchange-sourced feeds in particular) can return
    outcomes with explicit ``"price": null`` / ``"point": null`` for illiquid sides. ``dict.get``
    returns the stored ``None`` rather than the default in that case, so each cast is guarded
    before ``int()`` / ``float()`` is invoked. Return shape and existing semantics are unchanged
    when values are valid; missing/null totals fall through to DEFAULT_TOTAL / DEFAULT_JUICE, and
    a missing/null h2h price for a side leaves that side's ML value as ``None``.

    Team-label matching on the h2h branch goes through ``_h2h_name_matches`` which accepts
    provider abbreviations / aliases (e.g. ``"ARI Diamondbacks"`` vs ``"Arizona Diamondbacks"``)
    in addition to exact-string equality. Does not invent prices.
    """
    total_line = DEFAULT_TOTAL
    over_juice = under_juice = DEFAULT_JUICE
    ml_home = ml_away = None
    for market in book.get("markets") or []:
        if market.get("key") == "totals":
            outcomes = market.get("outcomes") or []
            for o in outcomes:
                name = (o.get("name") or "").lower()
                if name not in ("over", "under"):
                    continue
                raw_point = o.get("point")
                raw_price = o.get("price")
                try:
                    pt = float(raw_point) if raw_point is not None else DEFAULT_TOTAL
                except (TypeError, ValueError):
                    pt = DEFAULT_TOTAL
                try:
                    pr = int(raw_price) if raw_price is not None else DEFAULT_JUICE
                except (TypeError, ValueError):
                    pr = DEFAULT_JUICE
                if name == "over":
                    total_line = pt
                    over_juice = pr
                else:
                    total_line = pt
                    under_juice = pr
        elif market.get("key") == "h2h":
            home = book.get("_home")  # we'll set this from event
            away = book.get("_away")
            for o in market.get("outcomes") or []:
                n = (o.get("name") or "").strip()
                raw_price = o.get("price")
                if raw_price is None:
                    continue
                try:
                    p = int(raw_price)
                except (TypeError, ValueError):
                    continue
                if _h2h_name_matches(n, home):
                    ml_home = p
                elif _h2h_name_matches(n, away):
                    ml_away = p
    return total_line, over_juice, under_juice, ml_home, ml_away


def _extract_per_book_ml_flags(event, home, away):
    """Additive overlay for per-book h2h prices plus sharp/exchange presence flags.

    Does not alter the existing totals/h2h book selection. Purely additive fields.

    The Novig / ProphetX / Pinnacle entries remain the source for the missing-side
    backfill loop and the sharpness / market-truth fire gate (no change there).
    The additional DraftKings / Espn_Draftkings / Bovada / Caesars / Fliff entries
    are **observational only** — they are written into ``odds_info`` so that when
    a row later exports ``ML_Market_Status='missing_market'`` we can prove from
    logs / snapshots whether an untracked book actually carried the missing side
    at capture time. They do not feed backfill, gating, or selection.

    Uses case-insensitive substring match on bookmaker.key so vendor variants
    (e.g. 'novig' / 'novig_exchange', 'prophetx' / 'prophet_x', 'pinnacle',
    'draftkings' / 'espn_draftkings') resolve to their logical book. For each
    target, picks the first bookmaker whose key contains the substring, is not
    excluded, and which carries an h2h market. The ``draftkings`` target excludes
    ``espn_`` so it never resolves to the ESPN-branded variant (which has its own
    entry).

    Returns a dict with:
      novig_ml_home, novig_ml_away,
      prophetx_ml_home, prophetx_ml_away,
      pinnacle_ml_home, pinnacle_ml_away,
      draftkings_ml_home, draftkings_ml_away,
      espn_draftkings_ml_home, espn_draftkings_ml_away,
      bovada_ml_home, bovada_ml_away,
      caesars_ml_home, caesars_ml_away,
      fliff_ml_home, fliff_ml_away,
      exchange_present (bool; True when novig or prophetx has a usable h2h pair),
      prophetx_present (bool; ProphetX h2h pair usable — observational only),
      pinnacle_present (bool; Pinnacle h2h pair usable).
    Missing values default to None / False.
    """
    result = {
        "novig_ml_home": None,
        "novig_ml_away": None,
        "prophetx_ml_home": None,
        "prophetx_ml_away": None,
        "pinnacle_ml_home": None,
        "pinnacle_ml_away": None,
        "draftkings_ml_home": None,
        "draftkings_ml_away": None,
        "espn_draftkings_ml_home": None,
        "espn_draftkings_ml_away": None,
        "bovada_ml_home": None,
        "bovada_ml_away": None,
        "caesars_ml_home": None,
        "caesars_ml_away": None,
        "fliff_ml_home": None,
        "fliff_ml_away": None,
        "exchange_present": False,
        "prophetx_present": False,
        "pinnacle_present": False,
    }
    targets = (
        ("novig", "novig_ml_home", "novig_ml_away", ()),
        ("prophetx", "prophetx_ml_home", "prophetx_ml_away", ()),
        ("pinnacle", "pinnacle_ml_home", "pinnacle_ml_away", ()),
        ("espn_draftkings", "espn_draftkings_ml_home", "espn_draftkings_ml_away", ()),
        ("draftkings", "draftkings_ml_home", "draftkings_ml_away", ("espn_",)),
        ("bovada", "bovada_ml_home", "bovada_ml_away", ()),
        ("caesars", "caesars_ml_home", "caesars_ml_away", ()),
        ("fliff", "fliff_ml_home", "fliff_ml_away", ()),
    )
    for sub, home_key, away_key, excludes in targets:
        chosen = None
        for book in event.get("bookmakers") or []:
            bk = (book.get("key") or "").lower()
            if sub not in bk:
                continue
            if excludes and any(ex in bk for ex in excludes):
                continue
            if not any((m.get("key") == "h2h") for m in (book.get("markets") or [])):
                continue
            chosen = book
            break
        if chosen is None:
            continue
        chosen["_home"] = home
        chosen["_away"] = away
        try:
            _, _, _, mh, ma = _parse_totals_and_h2h(chosen)
        except Exception:
            mh = None
            ma = None
        result[home_key] = mh
        result[away_key] = ma
        pair_present = (mh is not None) and (ma is not None)
        if sub == "novig":
            if pair_present:
                result["exchange_present"] = True
        elif sub == "prophetx":
            result["prophetx_present"] = pair_present
            if pair_present:
                result["exchange_present"] = True
        elif sub == "pinnacle":
            result["pinnacle_present"] = pair_present
    return result


def _extract_per_book_totals_flags(event):
    """Additive overlay for per-book totals prices.

    Observational only. Does not alter the existing totals-book selection
    (``best`` in ``fetch_mlb_odds``), ``BOOK_PRIORITY``, the trusted-total
    filter in ``model/overgang_model.py``, or the final ``total_line``
    passed downstream. Written into ``result[key]`` so that later inspection
    can prove whether a Pinnacle-based O/U sharpness pivot is supported by
    the data already arriving in the Parlay payload. Does not feed totals
    selection, confidence, O/U fire logic, or any existing export.

    Matching uses case-insensitive substring on ``bookmaker.key`` with the
    same target set and exclusion rules as ``_extract_per_book_ml_flags``
    so Pinnacle / Novig / ProphetX / DraftKings / Espn_Draftkings / Bovada /
    Caesars / Fliff resolve consistently across both overlays. For each
    target, picks the first bookmaker whose key contains the substring, is
    not excluded, and which carries a ``totals`` market.

    For the selected book, reads the ``totals`` market only; ``total_line``,
    ``over_juice``, ``under_juice`` are recorded as raw parsed values (no
    ``DEFAULT_TOTAL`` / ``DEFAULT_JUICE`` fallback) so downstream inspection
    can tell "book absent" and "value null" apart from a real price. Does
    not invent prices.

    Returns a dict with (per target):
      <book>_total_line, <book>_over_juice, <book>_under_juice
    and the presence flags:
      pinnacle_totals_present (bool),
      prophetx_totals_present (bool),
      exchange_totals_present (bool; Novig or ProphetX totals pair present).
    Missing values default to None / False.
    """
    result = {
        "pinnacle_total_line": None,
        "pinnacle_over_juice": None,
        "pinnacle_under_juice": None,
        "novig_total_line": None,
        "novig_over_juice": None,
        "novig_under_juice": None,
        "prophetx_total_line": None,
        "prophetx_over_juice": None,
        "prophetx_under_juice": None,
        "draftkings_total_line": None,
        "draftkings_over_juice": None,
        "draftkings_under_juice": None,
        "espn_draftkings_total_line": None,
        "espn_draftkings_over_juice": None,
        "espn_draftkings_under_juice": None,
        "bovada_total_line": None,
        "bovada_over_juice": None,
        "bovada_under_juice": None,
        "caesars_total_line": None,
        "caesars_over_juice": None,
        "caesars_under_juice": None,
        "fliff_total_line": None,
        "fliff_over_juice": None,
        "fliff_under_juice": None,
        "pinnacle_totals_present": False,
        "prophetx_totals_present": False,
        "exchange_totals_present": False,
    }
    targets = (
        ("novig", "novig", ()),
        ("prophetx", "prophetx", ()),
        ("pinnacle", "pinnacle", ()),
        ("espn_draftkings", "espn_draftkings", ()),
        ("draftkings", "draftkings", ("espn_",)),
        ("bovada", "bovada", ()),
        ("caesars", "caesars", ()),
        ("fliff", "fliff", ()),
    )
    for sub, prefix, excludes in targets:
        chosen = None
        for book in event.get("bookmakers") or []:
            bk = (book.get("key") or "").lower()
            if sub not in bk:
                continue
            if excludes and any(ex in bk for ex in excludes):
                continue
            if not any((m.get("key") == "totals") for m in (book.get("markets") or [])):
                continue
            chosen = book
            break
        if chosen is None:
            continue
        tl = None
        oj = None
        uj = None
        for market in chosen.get("markets") or []:
            if market.get("key") != "totals":
                continue
            for o in market.get("outcomes") or []:
                name = (o.get("name") or "").lower()
                if name not in ("over", "under"):
                    continue
                raw_point = o.get("point")
                raw_price = o.get("price")
                try:
                    pt = float(raw_point) if raw_point is not None else None
                except (TypeError, ValueError):
                    pt = None
                try:
                    pr = int(raw_price) if raw_price is not None else None
                except (TypeError, ValueError):
                    pr = None
                if pt is not None:
                    tl = pt
                if name == "over":
                    oj = pr
                else:
                    uj = pr
        result[f"{prefix}_total_line"] = tl
        result[f"{prefix}_over_juice"] = oj
        result[f"{prefix}_under_juice"] = uj
        totals_pair = (tl is not None) and (oj is not None) and (uj is not None)
        if sub == "pinnacle":
            result["pinnacle_totals_present"] = totals_pair
        elif sub == "prophetx":
            result["prophetx_totals_present"] = totals_pair
            if totals_pair:
                result["exchange_totals_present"] = True
        elif sub == "novig":
            if totals_pair:
                result["exchange_totals_present"] = True
    return result


def fetch_mlb_odds(target_date=None):
    """
    Fetch MLB odds from The Odds API (US, American odds, h2h + totals).
    target_date: optional YYYY-MM-DD string (e.g. predictor slate date in MT). If set, we pass
    commenceTimeFrom/commenceTimeTo so the API returns only events on that date (MT), and we keep
    only events whose commence_time falls on that date.
    Returns dict: event_key -> { total_line, over_juice, under_juice, ml_home, ml_away, book }.
    Note: The API returns only events that bookmakers have posted; there is no param to "request
    more events" for a date—if no odds exist for that date, the response will be empty.
    """
    print(f"[ODDS API] ODDS_API_KEY exists: {bool(ODDS_API_KEY)}")
    if target_date:
        print(f"[ODDS API] Target date filter: {target_date} (MT)")
    if not ODDS_API_KEY:
        logging.warning("⚠️ ODDS_API_KEY not set; odds API disabled.")
        return {}

    url = f"{ODDS_BASE_URL}/sports/{MLB_SPORT_KEY}/odds"
    params = {
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    # Python-side target_date filtering only (no commenceTimeFrom/To on request) for backfill testing.
    if target_date:
        print(f"[ODDS API] Using Python-side target_date filtering only for: {target_date}")
    try:
        import requests
        from requests.exceptions import RequestException
    except ImportError:
        logging.warning("⚠️ requests not available; odds API disabled.")
        return {}

    r = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, params=params, timeout=PARLAY_FETCH_TIMEOUT_SEC)
            break
        except RequestException as e:
            if attempt == 1:
                print(
                    f"[ODDS API] Parlay API HTTPS request failed (attempt {attempt}/2): {e}; "
                    f"retrying once with timeout={PARLAY_FETCH_TIMEOUT_SEC}s..."
                )
                logging.warning(f"⚠️ Odds API request failed (will retry once): {e}")
                continue
            print(f"[ODDS API] Request failed after retry: {e}")
            logging.warning(f"⚠️ Odds API request failed after retry: {e}")
            return {}

    try:
        print(f"[ODDS API] Request succeeded: {r.status_code == 200}, HTTP status: {r.status_code}")
        r.raise_for_status()
        raw_payload = r.json()
        if isinstance(raw_payload, list):
            data = raw_payload
        elif (
            isinstance(raw_payload, dict)
            and isinstance(raw_payload.get("data"), list)
        ):
            data = raw_payload["data"]
            print(
                f"[ODDS API] Parlay API response wrapper detected: rows={len(data)}"
            )
        else:
            if isinstance(raw_payload, dict):
                keys_preview = list(raw_payload.keys())[:12]
                print(
                    "[ODDS API] Unrecognized odds JSON "
                    f"(dict keys={keys_preview}); returning empty odds_map."
                )
            else:
                print(
                    "[ODDS API] Unrecognized odds JSON "
                    f"(type={type(raw_payload).__name__}); returning empty odds_map."
                )
            return {}
        if len(data) == 0:
            print("[ODDS API] Empty events list from API; returning empty odds_map (no error).")
            return {}
        print(f"[ODDS API] Events returned: {len(data)}")
    except Exception as e:
        print(f"[ODDS API] Request failed: {e}")
        logging.warning(f"⚠️ Odds API request failed: {e}")
        return {}

    # Raw event inspection (first 5 events)
    events_to_inspect = data[:5] if isinstance(data, list) else []
    for i, ev in enumerate(events_to_inspect):
        away = (ev.get("away_team") or "").strip()
        home = (ev.get("home_team") or "").strip()
        commence = ev.get("commence_time", "")
        books = ev.get("bookmakers") or []
        book_names = [b.get("title") or b.get("key") or "?" for b in books]
        totals_per_book = []
        for b in books:
            has_totals = any((m.get("key") == "totals") for m in (b.get("markets") or []))
            totals_per_book.append(has_totals)
        any_totals = any(totals_per_book)
        print(f"[ODDS API] Event[{i}] away={repr(away)} home={repr(home)} commence_time={commence}")
        print(f"[ODDS API]   bookmakers ({len(books)}): {book_names}")
        print(f"[ODDS API]   totals exists per book: {totals_per_book}  any_totals={any_totals}")

    result = {}
    skip_log_cap = 3
    skip_log_count = 0
    no_totals_log_cap = 3
    no_totals_log_count = 0
    book_choice_log_cap = 3
    book_choice_log_count = 0
    date_skip_log_cap = 5
    date_skip_count = 0
    ml_incomplete_log_cap = 8
    ml_incomplete_log_count = 0
    ou_diag_log_cap = 8
    ou_diag_log_count = 0
    parlay_flat_count = 0
    for event in data:
        commence_str = event.get("commence_time") or ""
        event_date_mt = _event_date_mt(commence_str)
        event_slate_date_mt = _event_slate_date_mt(commence_str)
        if target_date and event_slate_date_mt != target_date:
            if date_skip_count < date_skip_log_cap:
                print(
                    f"[ODDS API] Skip (wrong slate date): event_date_mt={event_date_mt} "
                    f"event_slate_date_mt={event_slate_date_mt} target={target_date} "
                    f"away={repr((event.get('away_team') or '').strip())} home={repr((event.get('home_team') or '').strip())}"
                )
                date_skip_count += 1
            continue

        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        if _parlay_row_excludes_non_mlb_teams(away, home):
            continue
        key = _event_key(away, home, commence_str)
        coarse_key = _game_key(away, home)
        if not key or key in result:
            if skip_log_count < skip_log_cap:
                reason = "empty key" if not key else "duplicate key (already in result)"
                print(f"[ODDS API] Skip reason (event {repr(away)} @ {repr(home)}): {reason}")
                skip_log_count += 1
            continue

        if _is_parlay_flat_event_row(event):
            try:
                selected_total_line = float(event["total"])
            except (TypeError, ValueError):
                if skip_log_count < skip_log_cap:
                    print(
                        f"[ODDS API] Skip (flat row bad total): away={repr(away)} "
                        f"home={repr(home)} total={event.get('total')!r}"
                    )
                    skip_log_count += 1
                continue
            oj = _american_odds_int(event.get("over_juice"))
            uj = _american_odds_int(event.get("under_juice"))
            if oj is None:
                oj = DEFAULT_JUICE
            if uj is None:
                uj = DEFAULT_JUICE
            flat_ml_home = _american_odds_int(event.get("home_ml"))
            flat_ml_away = _american_odds_int(event.get("away_ml"))
            selected_book = (event.get("source") or "").strip() or "parlay_api"
            per_book_ml_flat = _extract_per_book_ml_flags(event, home, away)
            per_book_totals_flat = _extract_per_book_totals_flags(event)
            ml_h, ml_a = flat_ml_home, flat_ml_away
            if ml_h is None:
                for _src_home in ("pinnacle_ml_home", "novig_ml_home", "prophetx_ml_home"):
                    _v = per_book_ml_flat.get(_src_home)
                    if _v is not None:
                        ml_h = _v
                        break
            if ml_a is None:
                for _src_away in ("pinnacle_ml_away", "novig_ml_away", "prophetx_ml_away"):
                    _v = per_book_ml_flat.get(_src_away)
                    if _v is not None:
                        ml_a = _v
                        break
            result[key] = {
                "total_line": selected_total_line,
                "over_juice": oj,
                "under_juice": uj,
                "ml_home": ml_h,
                "ml_away": ml_a,
                "book": selected_book,
                "ml_book": selected_book,
                "_coarse_game_key": coarse_key,
                "_commence_time": _canonical_commence_time(commence_str),
                **per_book_ml_flat,
                **per_book_totals_flat,
            }
            parlay_flat_count += 1
            continue

        best = None
        best_rank = len(BOOK_PRIORITY) + 1
        for book in event.get("bookmakers") or []:
            book_key = (book.get("key") or "").lower()
            try:
                rank = BOOK_PRIORITY.index(book_key)
            except ValueError:
                rank = len(BOOK_PRIORITY)
            if rank >= best_rank:
                continue
            has_totals = any((m.get("key") == "totals") for m in (book.get("markets") or []))
            if not has_totals:
                continue
            best_rank = rank
            best = book

        used_fallback_loop = False
        if best is None:
            for book in event.get("bookmakers") or []:
                if any((m.get("key") == "totals") for m in (book.get("markets") or [])):
                    best = book
                    used_fallback_loop = True
                    break

        if best is None:
            # No totals book on this event: keep the diagnostic log but do NOT early-exit.
            # Totals fields stay at defaults; the h2h / per-book ML passes below still run so
            # moneyline markets and sharpness inputs aren't discarded along with missing totals.
            if no_totals_log_count < no_totals_log_cap:
                book_keys = [b.get("key") or "?" for b in (event.get("bookmakers") or [])]
                market_keys_per_book = []
                for b in (event.get("bookmakers") or []):
                    mkt = [m.get("key") for m in (b.get("markets") or [])]
                    market_keys_per_book.append((b.get("key"), mkt))
                print(f"[ODDS API] Event has no book with totals market: away={repr(away)} home={repr(home)}")
                print(f"[ODDS API]   books: {book_keys}; markets per book: {market_keys_per_book}")
                no_totals_log_count += 1

        if best is not None and book_choice_log_count < book_choice_log_cap:
            book_name = best.get("title") or best.get("key") or "?"
            print(f"[ODDS API] Event parsed: away={repr(away)} home={repr(home)} key={repr(key)} book={book_name} (from_fallback_loop={used_fallback_loop})")
            book_choice_log_count += 1

        # Select h2h-capable book independently using the same BOOK_PRIORITY ranking.
        # Sharp totals books (e.g. pinnacle) may lack h2h; moneylines then come from the
        # highest-ranked h2h-capable book on the same event. Totals still come from `best`,
        # so `book` continues to reflect the totals-source book for O/U logic.
        #
        # Two-sided preference: for each ranked h2h bookmaker we call _parse_totals_and_h2h
        # to see whether it yields BOTH ml_home and ml_away. The first such book wins.
        # If no ranked book is two-sided, we fall back to the best ranked one-sided book,
        # and finally to the first bookmaker with any h2h market (matching the prior
        # fallback order). Team-label injection is required on the probe copy so the h2h
        # branch in _parse_totals_and_h2h can match outcome.name == _home/_away.
        best_ml = None
        best_ml_rank = len(BOOK_PRIORITY) + 1
        best_ml_is_two_sided = False
        best_ml_one_sided_candidate = None
        best_ml_one_sided_rank = len(BOOK_PRIORITY) + 1
        for book in event.get("bookmakers") or []:
            book_key = (book.get("key") or "").lower()
            try:
                rank = BOOK_PRIORITY.index(book_key)
            except ValueError:
                rank = len(BOOK_PRIORITY)
            has_h2h = any((m.get("key") == "h2h") for m in (book.get("markets") or []))
            if not has_h2h:
                continue
            book["_home"] = home
            book["_away"] = away
            try:
                _, _, _, probe_home, probe_away = _parse_totals_and_h2h(book)
            except Exception:
                probe_home = None
                probe_away = None
            two_sided = (probe_home is not None) and (probe_away is not None)
            one_sided = (probe_home is not None) or (probe_away is not None)
            if two_sided and rank < best_ml_rank:
                best_ml = book
                best_ml_rank = rank
                best_ml_is_two_sided = True
            if (not best_ml_is_two_sided) and one_sided and rank < best_ml_one_sided_rank:
                best_ml_one_sided_candidate = book
                best_ml_one_sided_rank = rank
        if best_ml is None:
            best_ml = best_ml_one_sided_candidate
        if best_ml is None:
            for book in event.get("bookmakers") or []:
                if any((m.get("key") == "h2h") for m in (book.get("markets") or [])):
                    best_ml = book
                    break

        # Totals parsing (only when a totals book exists). When best is None, totals fields
        # remain at the defaults declared above — matching the prior no-totals row shape.
        if best is not None:
            best["_home"] = home
            best["_away"] = away
            total_line, over_juice, under_juice, _, _ = _parse_totals_and_h2h(best)
        else:
            total_line = DEFAULT_TOTAL
            over_juice = DEFAULT_JUICE
            under_juice = DEFAULT_JUICE
        if best_ml is not None:
            best_ml["_home"] = home
            best_ml["_away"] = away
            _, _, _, ml_home, ml_away = _parse_totals_and_h2h(best_ml)
        else:
            ml_home = None
            ml_away = None
        per_book_ml = _extract_per_book_ml_flags(event, home, away)

        # Missing-side backfill: if selected-book parsing left ml_home or ml_away as None,
        # fill the missing side from already-computed per-book ML pairs in priority order
        # (Pinnacle → Novig → ProphetX). `book` stays tied to the totals source for O/U
        # logic; this only populates the ML scalars. No downstream contract changes.
        if ml_home is None:
            for _src_home in ("pinnacle_ml_home", "novig_ml_home", "prophetx_ml_home"):
                _v = per_book_ml.get(_src_home)
                if _v is not None:
                    ml_home = _v
                    break
        if ml_away is None:
            for _src_away in ("pinnacle_ml_away", "novig_ml_away", "prophetx_ml_away"):
                _v = per_book_ml.get(_src_away)
                if _v is not None:
                    ml_away = _v
                    break

        # Observational-only: when the tracked-donor backfill cannot complete the ML pair,
        # emit a single capped diagnostic dumping every per-book ML value we already parsed
        # (tracked + untracked). This is evidence-gathering to prove whether untracked books
        # actually carried the missing side. No behavior change: ml_home/ml_away are NOT
        # altered here, backfill donors are NOT widened.
        if (ml_home is None or ml_away is None) and ml_incomplete_log_count < ml_incomplete_log_cap:
            per_book_ml_snapshot = {
                k: v for k, v in per_book_ml.items() if k.endswith("_ml_home") or k.endswith("_ml_away")
            }
            print(
                f"[ODDS API] ML incomplete after backfill: away={repr(away)} home={repr(home)} "
                f"key={repr(key)} ml_home={ml_home} ml_away={ml_away} "
                f"primary_ml_book={repr((best_ml.get('title') or best_ml.get('key') or '') if best_ml is not None else '')}"
            )
            print(f"[ODDS API]   per_book_ml_snapshot={per_book_ml_snapshot}")
            ml_incomplete_log_count += 1

        # Observational-only per-book totals overlay. Mirrors the ML overlay
        # pattern and is written into result[key] so later inspection can
        # prove whether a Pinnacle-based O/U sharpness pivot is supported by
        # the data already arriving in the Parlay payload. Does NOT change
        # totals-book selection, BOOK_PRIORITY, the trusted-total filter,
        # the selected `total_line`, O/U confidence, O/U fire logic, or any
        # existing export contract.
        per_book_totals = _extract_per_book_totals_flags(event)

        # Capped O/U diagnostic: surface events where per-book totals truth
        # diverges from the selected-book scalar. Fires when at least two
        # tracked books carry totals, or Pinnacle totals exist but were not
        # chosen as `best`, or the event has no totals book at all. Does
        # not alter any selection or gating; purely evidence-gathering.
        _selected_book_key = (
            ((best.get("key") or "") if best is not None else "").lower()
        )
        _pinnacle_total = per_book_totals.get("pinnacle_total_line")
        _books_with_totals = sum(
            1
            for _k, _v in per_book_totals.items()
            if _k.endswith("_total_line") and _v is not None
        )
        _should_log_ou_diag = (
            (_books_with_totals >= 2)
            or (_pinnacle_total is not None and "pinnacle" not in _selected_book_key)
            or (best is None)
        )
        if _should_log_ou_diag and ou_diag_log_count < ou_diag_log_cap:
            per_book_totals_snapshot = {
                _k: _v
                for _k, _v in per_book_totals.items()
                if _k.endswith("_total_line")
                or _k.endswith("_over_juice")
                or _k.endswith("_under_juice")
            }
            print(
                f"[ODDS API] O/U per-book totals snapshot: away={repr(away)} home={repr(home)} "
                f"key={repr(key)} selected_total_line={total_line if best is not None else None} "
                f"selected_book={repr((best.get('title') or best.get('key') or '') if best is not None else '')} "
                f"books_with_totals={_books_with_totals} pinnacle_total={_pinnacle_total} "
                f"pinnacle_totals_present={per_book_totals.get('pinnacle_totals_present')} "
                f"exchange_totals_present={per_book_totals.get('exchange_totals_present')}"
            )
            print(f"[ODDS API]   per_book_totals_snapshot={per_book_totals_snapshot}")
            ou_diag_log_count += 1

        result[key] = {
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "book": (best.get("title") or best.get("key") or "") if best is not None else "",
            "ml_book": (best_ml.get("title") or best_ml.get("key") or "") if best_ml is not None else "",
            "_coarse_game_key": coarse_key,
            "_commence_time": _canonical_commence_time(commence_str),
            **per_book_ml,
            **per_book_totals,
        }

    if parlay_flat_count:
        print(
            f"[ODDS API] Parlay flat rows parsed into odds_map: {parlay_flat_count}"
        )
    print(f"[ODDS API] Games parsed into odds_map: {len(result)} (from {len(data)} API events)")
    if target_date and len(result) == 0:
        print(f"[ODDS API] No odds found for target date {target_date}; returning empty odds_map.")
    sample_keys = list(result.keys())[:3]
    print(f"[ODDS API] Sample odds_map keys (3): {sample_keys}")
    for k in sample_keys:
        v = result.get(k, {})
        print(f"[ODDS API]   Sample value: total_line={v.get('total_line')}, over_juice={v.get('over_juice')}, under_juice={v.get('under_juice')}, book={repr(v.get('book'))}")
    return result


def get_game_odds(away_team, home_team, odds_map=None, commence_time=None):
    """
    Get odds for one game. odds_map from fetch_mlb_odds() or None to fetch now.
    Returns dict: total_line, over_juice, under_juice, ml_home, ml_away, book.
    Uses 8.5 / -110 / None / "" when missing.
    """
    if odds_map is None:
        odds_map = fetch_mlb_odds()
    coarse_key = _game_key(away_team, home_team)
    strong_key = _event_key(away_team, home_team, commence_time) if commence_time else ""
    row = None
    matched_key = ""
    if strong_key:
        row = odds_map.get(strong_key)
        if row:
            matched_key = strong_key
    if row is None:
        row = odds_map.get(coarse_key)
        if row:
            matched_key = coarse_key
    if row is None:
        candidates = []
        wanted_commence = _canonical_commence_time(commence_time)
        for k, v in (odds_map or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("_coarse_game_key") == coarse_key:
                candidates.append((k, v))
        if candidates:
            if wanted_commence:
                for k, v in candidates:
                    if (v.get("_commence_time") or "") == wanted_commence:
                        row = v
                        matched_key = k
                        break
            if row is not None:
                candidates = []
        if row is None and candidates:
            candidates.sort(key=lambda kv: str(kv[1].get("_commence_time") or ""))
            row = candidates[0][1]
            matched_key = candidates[0][0]
    if row:
        raw_line = row.get("total_line")
        if raw_line is None or raw_line == "":
            total_line = DEFAULT_TOTAL
        else:
            try:
                total_line = float(raw_line)
            except (TypeError, ValueError):
                total_line = DEFAULT_TOTAL
        raw_over = row.get("over_juice")
        if raw_over is None or raw_over == "":
            over_juice = DEFAULT_JUICE
        else:
            try:
                over_juice = int(raw_over)
            except (TypeError, ValueError):
                over_juice = DEFAULT_JUICE
        raw_under = row.get("under_juice")
        if raw_under is None or raw_under == "":
            under_juice = DEFAULT_JUICE
        else:
            try:
                under_juice = int(raw_under)
            except (TypeError, ValueError):
                under_juice = DEFAULT_JUICE
        out = dict(row)
        out.update({
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
            "ml_home": row.get("ml_home"),
            "ml_away": row.get("ml_away"),
            "book": row.get("book") or "",
            "ml_book": row.get("ml_book") or "",
            "_match_found": True,
            "_lookup_key": matched_key or coarse_key,
            "_coarse_game_key": row.get("_coarse_game_key") or coarse_key,
            "_commence_time": row.get("_commence_time") or _canonical_commence_time(commence_time),
            "_raw_total_line": raw_line,
            "novig_ml_home": row.get("novig_ml_home"),
            "novig_ml_away": row.get("novig_ml_away"),
            "prophetx_ml_home": row.get("prophetx_ml_home"),
            "prophetx_ml_away": row.get("prophetx_ml_away"),
            "pinnacle_ml_home": row.get("pinnacle_ml_home"),
            "pinnacle_ml_away": row.get("pinnacle_ml_away"),
            "exchange_present": bool(row.get("exchange_present", False)),
            "prophetx_present": bool(row.get("prophetx_present", False)),
            "pinnacle_present": bool(row.get("pinnacle_present", False)),
        })
        return out
    return {
        "total_line": DEFAULT_TOTAL,
        "over_juice": DEFAULT_JUICE,
        "under_juice": DEFAULT_JUICE,
        "ml_home": None,
        "ml_away": None,
        "book": "",
        "ml_book": "",
        "_match_found": False,
        "_lookup_key": strong_key or coarse_key,
        "_coarse_game_key": coarse_key,
        "_commence_time": _canonical_commence_time(commence_time),
        "_raw_total_line": None,
        "novig_ml_home": None,
        "novig_ml_away": None,
        "prophetx_ml_home": None,
        "prophetx_ml_away": None,
        "pinnacle_ml_home": None,
        "pinnacle_ml_away": None,
        "exchange_present": False,
        "prophetx_present": False,
        "pinnacle_present": False,
    }

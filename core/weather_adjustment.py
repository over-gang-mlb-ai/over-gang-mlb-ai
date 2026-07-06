# core/weather_adjustment.py
"""
Bounded game-level weather overlay for totals (Phase 2B).

Does not replace static park factors: returns a small multiplier applied as:
  effective_park_runs_factor = park_runs_factor * weather_runs_mult

Uses Open-Meteo (no API key) for hourly temperature and wind at the venue when
coordinates are known. On any failure or unknown venue, returns 1.0.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Total weather effect clamp (combined temp + wind deltas).
# Normal weather remains modest, but extreme outdoor heat needs enough room to
# move run environment when stacked with park/offense/pitching pressure.
WEATHER_RUNS_MULT_MIN = 0.97
WEATHER_RUNS_MULT_MAX = 1.055

# Neutral anchors (conservative linear tweaks, then capped per-component).
_NEUTRAL_TEMP_C = 21.0  # ~70°F
_NEUTRAL_WIND_MS = 4.0
_TEMP_COEFF = 0.0006  # delta in mult per °C from neutral (then capped)
_WIND_COEFF = 0.002  # delta in mult per m/s from neutral (then capped)
_COMPONENT_CAP = 0.012  # max |contribution| from temp or wind before final clamp

# Nonlinear heat add-on. The base temp coefficient was intentionally mild and
# underweighted 95–105°F games. This adds only positive heat pressure above
# ~90°F; roof/dome suppression still happens before this helper is called.
_EXTREME_HEAT_START_C = 32.2  # ~90°F
_EXTREME_HEAT_FULL_C = 40.6   # ~105°F
_EXTREME_HEAT_EXTRA_MAX = 0.045
_EXTREME_HEAT_CURVE_POWER = 1.35

# Lat/lon for PARK_FACTORS venue names in model/overgang_model.py (approximate stadium centers).
VENUE_LAT_LON: dict[str, Tuple[float, float]] = {
    "Coors Field": (39.7559, -104.9942),
    "Fenway Park": (42.3467, -71.0972),
    "Globe Life Field": (32.7512, -97.0828),
    "Daikin Park": (29.7569, -95.3555),
    "Oriole Park at Camden Yards": (39.2839, -76.6217),
    "Great American Ball Park": (39.0974, -84.5066),
    "Wrigley Field": (41.9484, -87.6553),
    "Petco Park": (32.7073, -117.1570),
    "Oracle Park": (37.7786, -122.3893),
    "Citi Field": (40.7571, -73.8458),
    "UNIQLO Field at Dodger Stadium": (34.0736, -118.2400),
    "T-Mobile Park": (47.5914, -122.3325),
    "American Family Field": (43.02838, -87.97099),
    "Angel Stadium": (33.80019, -117.88240),
    "Busch Stadium": (38.62257, -90.19287),
    "Chase Field": (33.44530, -112.06669),
    "Citizens Bank Park": (39.90539, -75.16717),
    "Comerica Park": (42.33912, -83.04870),
    "Kauffman Stadium": (39.05157, -94.48048),
    "Nationals Park": (38.87286, -77.00750),
    "PNC Park": (40.44690, -80.00575),
    "Progressive Field": (41.49586, -81.68526),
    "Rate Field": (41.83000, -87.63417),
    "Rogers Centre": (43.64155, -79.38915),
    "Sutter Health Park": (38.57994, -121.51246),
    "Target Field": (44.98183, -93.27789),
    "Tropicana Field": (27.76778, -82.65250),
    "Truist Park": (33.89067, -84.46764),
    "Yankee Stadium": (40.82919, -73.92650),
    "loanDepot park": (25.77796, -80.21952),
}


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


def _fetch_hourly_temp_wind_ms(
    lat: float, lon: float, dt_utc: datetime
) -> Optional[Tuple[float, float, float, Optional[float]]]:
    """Return (temp_c, wind_m/s, gust_m/s, wind_dir_from_deg) for closest hourly slot, or None.

    gust_m/s falls back to wind_m/s when the gust array is missing or unparsable.
    wind_dir_from_deg is meteorological direction the wind comes FROM; None if unavailable.
    """
    now = datetime.now(timezone.utc)
    date_str = dt_utc.strftime("%Y-%m-%d")
    base = (
        "https://archive-api.open-meteo.com/v1/archive"
        if dt_utc < now - timedelta(minutes=30)
        else "https://api.open-meteo.com/v1/forecast"
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m",
        "timezone": "UTC",
        "start_date": date_str,
        "end_date": date_str,
        "windspeed_unit": "ms",
    }
    try:
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("Open-Meteo request failed: %s", e)
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    gusts = hourly.get("wind_gusts_10m") or []
    wind_dirs = hourly.get("wind_direction_10m") or []
    if not times or len(temps) != len(times) or len(winds) != len(times):
        return None

    best_i = 0
    best_d = None
    for i, t in enumerate(times):
        try:
            if str(t).endswith("Z"):
                ht = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            else:
                ht = datetime.fromisoformat(str(t))
            if ht.tzinfo is None:
                ht = ht.replace(tzinfo=timezone.utc)
            d = abs((ht - dt_utc).total_seconds())
        except Exception:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_i = i

    try:
        temp_c = float(temps[best_i])
        wind_ms = float(winds[best_i])
    except (TypeError, ValueError, IndexError):
        return None

    try:
        gust_ms = float(gusts[best_i])
        if not (gust_ms == gust_ms):  # NaN guard
            gust_ms = wind_ms
    except (TypeError, ValueError, IndexError):
        gust_ms = wind_ms

    wind_dir_from_deg = None
    try:
        wind_dir_from_deg = float(wind_dirs[best_i])
        if not (wind_dir_from_deg == wind_dir_from_deg):  # NaN guard
            wind_dir_from_deg = None
        elif not (0.0 <= wind_dir_from_deg <= 360.0):
            wind_dir_from_deg = None
    except (TypeError, ValueError, IndexError):
        wind_dir_from_deg = None

    return temp_c, wind_ms, gust_ms, wind_dir_from_deg


# Coordinate-derived center-field bearings in degrees from home plate toward center field.
# Source type: public Google Earth / coordinate-derived stadium orientation work, not live MLB API.
# Missing venues fall back to the prior scalar gust behavior.
VENUE_CF_BEARING_DEG: dict[str, float] = {
    "American Family Field": 128.8537047,
    "Angel Stadium": 44.17136547,
    "Busch Stadium": 62.69629144,
    "Chase Field": 359.7491679,
    "Citi Field": 13.99114182,
    "Citizens Bank Park": 10.07051968,
    "Coors Field": 4.962490772,
    "Comerica Park": 151.2270493,
    "Daikin Park": 343.1738414,
    "Fenway Park": 44.17424096,
    "Globe Life Field": 32.80283371,
    "Great American Ball Park": 122.3427354,
    "Kauffman Stadium": 46.71018194,
    "loanDepot park": 128.1321885,
    "Nationals Park": 28.94819353,
    "Oriole Park at Camden Yards": 31.33978546,
    "Oracle Park": 85.1519466,
    "Petco Park": 359.7374753,
    "PNC Park": 116.5764243,
    "Progressive Field": 359.2547923,
    "Rate Field": 127.0606723,
    "Rogers Centre": 344.0904107,
    "T-Mobile Park": 49.30884076,
    "Target Field": 90.50394579,
    "Truist Park": 157.64687,
    "UNIQLO Field at Dodger Stadium": 26.4845402,
    "Wrigley Field": 37.61572884,
    "Yankee Stadium": 75.61766684,
}


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _directional_wind_component(
    venue_name: str,
    wind_ms: float,
    gust_ms: float,
    wind_dir_from_deg: Optional[float],
) -> Optional[float]:
    """Return signed wind component in m/s above the neutral wind threshold.

    Positive = blowing out toward center field.
    Negative = blowing in toward home plate.
    Zero = weak wind or crosswind.
    None = no trusted bearing / no wind direction available.
    """
    cf_bearing = VENUE_CF_BEARING_DEG.get(str(venue_name).strip())
    if cf_bearing is None or wind_dir_from_deg is None:
        return None

    effective_wind_ms = max(float(wind_ms), float(gust_ms))
    if effective_wind_ms <= _NEUTRAL_WIND_MS:
        return 0.0

    # Open-Meteo wind direction is meteorological: direction wind comes FROM.
    # Convert to direction wind is blowing TO.
    wind_to_deg = (float(wind_dir_from_deg) + 180.0) % 360.0
    diff_to_cf = _angle_diff_deg(wind_to_deg, float(cf_bearing))

    # +1 = directly out to CF, -1 = directly in from CF, 0 = crosswind.
    alignment = math.cos(math.radians(diff_to_cf))

    # Ignore weak crosswind noise.
    if abs(alignment) < 0.35:
        return 0.0

    return (effective_wind_ms - _NEUTRAL_WIND_MS) * alignment


def _mult_from_temp_wind(
    temp_c: float,
    wind_ms: float,
    gust_ms: float,
    wind_dir_from_deg: Optional[float] = None,
    venue_name: Optional[str] = None,
) -> float:
    """Combine temp + directional wind deltas; clamp total multiplier to WEATHER_* bounds."""
    t_comp = _TEMP_COEFF * (temp_c - _NEUTRAL_TEMP_C)
    t_comp = max(-_COMPONENT_CAP, min(_COMPONENT_CAP, t_comp))

    heat_extra = 0.0
    if temp_c >= _EXTREME_HEAT_START_C:
        heat_span = max(0.001, _EXTREME_HEAT_FULL_C - _EXTREME_HEAT_START_C)
        heat_intensity = min(1.0, (temp_c - _EXTREME_HEAT_START_C) / heat_span)
        heat_extra = _EXTREME_HEAT_EXTRA_MAX * (
            heat_intensity ** _EXTREME_HEAT_CURVE_POWER
        )

    directional_component = None
    if venue_name:
        directional_component = _directional_wind_component(
            venue_name, wind_ms, gust_ms, wind_dir_from_deg
        )

    if directional_component is None:
        # Fallback for venues without a bearing: preserve prior scalar gust behavior.
        effective_wind_ms = max(wind_ms, gust_ms)
        w_comp = _WIND_COEFF * (effective_wind_ms - _NEUTRAL_WIND_MS)
    else:
        # Directional path: blowing out boosts, blowing in suppresses, crosswind is neutral.
        w_comp = _WIND_COEFF * directional_component

    w_comp = max(-_COMPONENT_CAP, min(_COMPONENT_CAP, w_comp))
    m = 1.0 + t_comp + heat_extra + w_comp
    return max(WEATHER_RUNS_MULT_MIN, min(WEATHER_RUNS_MULT_MAX, m))


_ALWAYS_ENCLOSED_VENUES: set[str] = {"Tropicana Field"}


def _fetch_mlb_weather_condition(game_pk) -> Optional[str]:
    """Return MLB StatsAPI gameData.weather.condition (lowercased) for a gamePk, or None.

    Fail-soft on any error; callers must treat None as 'no roof-state truth available'.
    """
    if game_pk is None:
        return None
    try:
        pk_int = int(game_pk)
    except (TypeError, ValueError):
        return None
    if pk_int <= 0:
        return None
    url = f"https://statsapi.mlb.com/api/v1.1/game/{pk_int}/feed/live"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "over-gang-mlb-ai/weather"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("MLB live-feed roof fetch failed for gamePk=%s: %s", pk_int, e)
        return None
    cond = (((data.get("gameData") or {}).get("weather") or {}).get("condition") or "")
    cond = str(cond).strip().lower()
    return cond or None


def _roof_state_blocks_outdoor_weather(condition_text: Optional[str]) -> bool:
    """True when MLB weather.condition indicates a dome/closed-roof game (no outdoor weather)."""
    if not condition_text:
        return False
    t = str(condition_text).strip().lower()
    if not t:
        return False
    if "closed" in t:
        return True
    if "dome" in t and "open" not in t:
        return True
    return False


def compute_weather_runs_mult(
    venue_name: Optional[str],
    game_datetime: Optional[str],
    game_pk=None,
) -> float:
    """
    Return a bounded multiplier in [WEATHER_RUNS_MULT_MIN, WEATHER_RUNS_MULT_MAX], or 1.0 when
    venue/time is unknown or weather cannot be fetched.

    When game_pk is provided, MLB StatsAPI weather.condition is consulted; if the condition
    indicates roof closed or a dome game, outdoor weather is suppressed (returns 1.0). When the
    feed is unavailable or the condition does not indicate closed/dome, behavior is unchanged.

    This is an environment overlay on top of static park factors, not a replacement for them.
    """
    if not venue_name or not str(venue_name).strip():
        return 1.0
    key = str(venue_name).strip()
    if key == "Unknown":
        return 1.0
    if key in _ALWAYS_ENCLOSED_VENUES:
        return 1.0
    coords = VENUE_LAT_LON.get(key)
    if not coords:
        return 1.0

    dt_utc = _parse_game_datetime_utc(game_datetime)
    if dt_utc is None:
        return 1.0
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    if game_pk is not None:
        roof_condition = _fetch_mlb_weather_condition(game_pk)
        if _roof_state_blocks_outdoor_weather(roof_condition):
            return 1.0

    lat, lon = coords
    tw = _fetch_hourly_temp_wind_ms(lat, lon, dt_utc)
    if tw is None:
        return 1.0
    temp_c, wind_ms, gust_ms, wind_dir_from_deg = tw
    return _mult_from_temp_wind(
        temp_c,
        wind_ms,
        gust_ms,
        wind_dir_from_deg=wind_dir_from_deg,
        venue_name=key,
    )

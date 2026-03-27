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
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Total weather effect clamp (combined temp + wind deltas); keep modest vs park table.
WEATHER_RUNS_MULT_MIN = 0.97
WEATHER_RUNS_MULT_MAX = 1.03

# Neutral anchors (conservative linear tweaks, then capped per-component).
_NEUTRAL_TEMP_C = 21.0  # ~70°F
_NEUTRAL_WIND_MS = 4.0
_TEMP_COEFF = 0.0006  # delta in mult per °C from neutral (then capped)
_WIND_COEFF = 0.002  # delta in mult per m/s from neutral (then capped)
_COMPONENT_CAP = 0.012  # max |contribution| from temp or wind before final clamp

# Lat/lon for PARK_FACTORS venue names in model/overgang_model.py (approximate stadium centers).
VENUE_LAT_LON: dict[str, Tuple[float, float]] = {
    "Coors Field": (39.7559, -104.9942),
    "Fenway Park": (42.3467, -71.0972),
    "Globe Life Field": (32.7512, -97.0828),
    "Camden Yards": (39.2839, -76.6217),
    "Great American Ball Park": (39.0974, -84.5066),
    "Wrigley Field": (41.9484, -87.6553),
    "Petco Park": (32.7073, -117.1570),
    "Oracle Park": (37.7786, -122.3893),
    "Citi Field": (40.7571, -73.8458),
    "Dodger Stadium": (34.0736, -118.2400),
    "T-Mobile Park": (47.5914, -122.3325),
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
) -> Optional[Tuple[float, float]]:
    """Return (temp_c, wind_m/s) for the hourly slot closest to dt_utc, or None."""
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
        "hourly": "temperature_2m,wind_speed_10m",
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
    return temp_c, wind_ms


def _mult_from_temp_wind(temp_c: float, wind_ms: float) -> float:
    """Combine small temp + wind deltas; clamp total multiplier to WEATHER_* bounds."""
    t_comp = _TEMP_COEFF * (temp_c - _NEUTRAL_TEMP_C)
    t_comp = max(-_COMPONENT_CAP, min(_COMPONENT_CAP, t_comp))
    w_comp = _WIND_COEFF * (wind_ms - _NEUTRAL_WIND_MS)
    w_comp = max(-_COMPONENT_CAP, min(_COMPONENT_CAP, w_comp))
    m = 1.0 + t_comp + w_comp
    return max(WEATHER_RUNS_MULT_MIN, min(WEATHER_RUNS_MULT_MAX, m))


def compute_weather_runs_mult(venue_name: Optional[str], game_datetime: Optional[str]) -> float:
    """
    Return a bounded multiplier in [WEATHER_RUNS_MULT_MIN, WEATHER_RUNS_MULT_MAX], or 1.0 when
    venue/time is unknown or weather cannot be fetched.

    This is an environment overlay on top of static park factors, not a replacement for them.
    """
    if not venue_name or not str(venue_name).strip():
        return 1.0
    key = str(venue_name).strip()
    if key == "Unknown":
        return 1.0
    coords = VENUE_LAT_LON.get(key)
    if not coords:
        return 1.0

    dt_utc = _parse_game_datetime_utc(game_datetime)
    if dt_utc is None:
        return 1.0
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    lat, lon = coords
    tw = _fetch_hourly_temp_wind_ms(lat, lon, dt_utc)
    if tw is None:
        return 1.0
    temp_c, wind_ms = tw
    return _mult_from_temp_wind(temp_c, wind_ms)

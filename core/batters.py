# core/batters.py
import os
import json
import logging
from typing import List, Optional, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np

# ---------- Paths ----------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

BATTER_STATS_CSV = os.path.join(DATA_DIR, "batter_stats.csv")
TEAM_BEST9_JSON = os.path.join(DATA_DIR, "team_best9.json")          # optional
PITCHER_HAND_CSV = os.path.join(DATA_DIR, "pitcher_handedness.csv")  # optional
TEAM_OFFENSE_SPLITS_CSV = os.path.join(DATA_DIR, "team_offense_splits.csv")  # optional clean team-vs-hand source


# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Lightweight name/team normalizer."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace(".", "")
    s = s.replace("-", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _safe_float(x, default=np.nan):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


_TEAM_OFFENSE_SPLITS_CACHE = None

_TEAM_NAME_ALIASES = {
    "oakland athletics": "athletics",
    "as": "athletics",
    "a s": "athletics",
    "az": "arizona diamondbacks",
    "ari": "arizona diamondbacks",
}


def _team_alias_norm(s: str) -> str:
    key = _norm(s)
    return _TEAM_NAME_ALIASES.get(key, key)


def _bounded_ratio(value, anchor, lo=0.80, hi=1.20):
    v = _safe_float(value)
    if np.isnan(v) or not anchor:
        return None
    try:
        ratio = float(v) / float(anchor)
    except Exception:
        return None
    return max(lo, min(hi, ratio))


def _load_team_offense_splits() -> pd.DataFrame:
    """
    Load clean team offense vs pitcher-hand splits if available.
    Source file is created by scripts/update_team_offense_splits.py.
    """
    global _TEAM_OFFENSE_SPLITS_CACHE

    if _TEAM_OFFENSE_SPLITS_CACHE is not None:
        return _TEAM_OFFENSE_SPLITS_CACHE

    if not os.path.exists(TEAM_OFFENSE_SPLITS_CSV):
        _TEAM_OFFENSE_SPLITS_CACHE = pd.DataFrame()
        return _TEAM_OFFENSE_SPLITS_CACHE

    try:
        df = pd.read_csv(TEAM_OFFENSE_SPLITS_CSV)
    except Exception as e:
        logging.warning(f"⚠️ Could not read team_offense_splits.csv: {e}")
        df = pd.DataFrame()

    _TEAM_OFFENSE_SPLITS_CACHE = df
    return _TEAM_OFFENSE_SPLITS_CACHE


def _team_offense_split_dict(team_name: str, pitcher_hand: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Prefer clean MLB StatsAPI team-vs-hand splits when available.

    Returns the same caller-facing shape as offense_vs_hand_dict:
    {"mult": float, "pop": "team", ...extra diagnostic keys...}

    Extra keys are intentionally backward-compatible; existing callers use "mult"/"pop".
    """
    if not team_name or pitcher_hand not in ("R", "L"):
        return None

    df = _load_team_offense_splits()
    if df is None or df.empty:
        return None

    required = {"Pitcher_Hand", "Quality", "Estimated_wOBA", "OPS", "ISO", "BB_Pct", "K_Pct"}
    if not required.issubset(set(df.columns)):
        return None

    hand = str(pitcher_hand).upper()
    team_key = _team_alias_norm(team_name)

    sdf = df[df["Pitcher_Hand"].astype(str).str.upper() == hand].copy()
    if sdf.empty:
        return None

    match = pd.DataFrame()
    for col in ["Team_Name", "Team", "Team_Code", "File_Code"]:
        if col in sdf.columns:
            col_norm = sdf[col].astype(str).map(_team_alias_norm)
            match = sdf[col_norm == team_key]
            if not match.empty:
                break

    if match.empty:
        return None

    row = match.iloc[0]

    if str(row.get("Quality", "")).strip() != "mlb_raw_complete":
        return None

    pa = _safe_float(row.get("PA"))
    if np.isnan(pa) or pa <= 0:
        return None

    # Clean source multiplier:
    # - Estimated_wOBA/OPS: broad scoring quality
    # - ISO: damage/power pressure
    # - BB%: traffic pressure
    # - K%: contact suppression/pressure, inverted because lower K% is better for offense
    factors = []

    woba_ratio = _bounded_ratio(row.get("Estimated_wOBA"), _LEAGUE_WOBA)
    if woba_ratio is not None:
        factors.append((0.42, woba_ratio))

    ops_ratio = _bounded_ratio(row.get("OPS"), _LEAGUE_OPS)
    if ops_ratio is not None:
        factors.append((0.26, ops_ratio))

    iso_ratio = _bounded_ratio(row.get("ISO"), _LEAGUE_ISO)
    if iso_ratio is not None:
        factors.append((0.16, iso_ratio))

    bb_ratio = _bounded_ratio(row.get("BB_Pct"), 8.5, lo=0.85, hi=1.15)
    if bb_ratio is not None:
        factors.append((0.08, bb_ratio))

    k_pct = _safe_float(row.get("K_Pct"))
    if not np.isnan(k_pct) and k_pct > 0:
        k_ratio = max(0.85, min(1.15, 22.0 / k_pct))
        factors.append((0.08, k_ratio))

    if not factors:
        return None

    weight_sum = sum(w for w, _ in factors)
    raw_rel = sum(w * v for w, v in factors) / weight_sum

    # Damp small-sample split rows toward neutral. Most team rows are safely above this.
    reliability = max(0.35, min(1.0, pa / 500.0))
    mult = 1.0 + ((raw_rel - 1.0) * reliability)

    # Keep existing offense multiplier safety rails.
    mult = max(0.90, min(1.10, float(mult)))

    return {
        "mult": mult,
        "pop": "team",
        "source": "team_offense_splits",
        "pa": int(pa),
        "estimated_woba": _safe_float(row.get("Estimated_wOBA")),
        "ops": _safe_float(row.get("OPS")),
        "iso": _safe_float(row.get("ISO")),
        "bb_pct": _safe_float(row.get("BB_Pct")),
        "k_pct": _safe_float(row.get("K_Pct")),
    }


# Anchors for normalizing platoon stats (fallback when MLB omits wOBA/wRC+ on statSplits)
_LEAGUE_WOBA = 0.310
_LEAGUE_WRC_SCALE = 100.0
_LEAGUE_OPS = 0.715
_LEAGUE_OBP = 0.320
_LEAGUE_SLG = 0.410
_LEAGUE_ISO = 0.155

# Weighted fallback run-pressure index when advanced wRC+/wOBA splits are unavailable.
# OPS = broad production, OBP = traffic, SLG = damage, ISO = pure power.
_FALLBACK_RUN_PRESSURE_WEIGHTS = {
    "OPS": 0.35,
    "OBP": 0.25,
    "SLG": 0.25,
    "ISO": 0.15,
}

# Halve deviation from 1.0 so basic split fallback does not move lineup impact
# as aggressively as wRC+/wOBA.
_FALLBACK_DAMP = 0.5

# Empirical expected plate-appearance weights by batting slot.
# Derived from 596 completed MLB team lineups across 300 archived games.
# Mean weight is 1.0, so this changes within-lineup opportunity allocation
# without creating an additional overall lineup-strength boost.
_LINEUP_SLOT_PA_WEIGHTS = {
    1: 1.1013,
    2: 1.0764,
    3: 1.0547,
    4: 1.0311,
    5: 1.0066,
    6: 0.9745,
    7: 0.9428,
    8: 0.9207,
    9: 0.8919,
}


def _weighted_numeric_mean(
    values: pd.Series,
    weights: List[float],
) -> Optional[float]:
    """Weighted mean over finite numeric values only."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    weight_array = np.asarray(
        weights,
        dtype=float,
    )

    if len(numeric) != len(weight_array):
        return None

    valid = (
        np.isfinite(numeric)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )

    if not valid.any():
        return None

    weight_sum = float(weight_array[valid].sum())

    if weight_sum <= 0.0:
        return None

    return float(
        np.average(
            numeric[valid],
            weights=weight_array[valid],
        )
    )


def _ordered_platoon_split_relative(
    df: pd.DataFrame,
    pitcher_hand: Optional[str],
    slot_weights: List[float],
) -> Optional[Tuple[float, str]]:
    """
    Order-aware equivalent of _platoon_split_relative.

    Uses the same metric hierarchy and fallback damping, but each hitter is
    weighted by the expected plate-appearance opportunity for their posted
    batting slot.
    """
    if df is None or df.empty:
        return None

    if len(df) != len(slot_weights):
        return None

    prefix = "vsL" if pitcher_hand == "L" else "vsR"

    col_wrc = (
        f"{prefix}_wRC+"
        if f"{prefix}_wRC+" in df.columns
        else None
    )
    col_woba = (
        f"{prefix}_wOBA"
        if f"{prefix}_wOBA" in df.columns
        else None
    )
    col_ops = (
        f"{prefix}_OPS"
        if f"{prefix}_OPS" in df.columns
        else None
    )
    col_obp = (
        f"{prefix}_OBP"
        if f"{prefix}_OBP" in df.columns
        else None
    )
    col_slg = (
        f"{prefix}_SLG"
        if f"{prefix}_SLG" in df.columns
        else None
    )
    col_iso = (
        f"{prefix}_ISO"
        if f"{prefix}_ISO" in df.columns
        else None
    )

    advanced: List[float] = []

    if col_wrc:
        weighted = _weighted_numeric_mean(
            df[col_wrc],
            slot_weights,
        )
        if weighted is not None:
            advanced.append(
                weighted / _LEAGUE_WRC_SCALE
            )

    if col_woba:
        weighted = _weighted_numeric_mean(
            df[col_woba],
            slot_weights,
        )
        if weighted is not None:
            advanced.append(
                weighted / _LEAGUE_WOBA
            )

    if advanced:
        return float(np.nanmean(advanced)), "advanced"

    parts: List[Tuple[float, float]] = []

    for column, anchor, metric_weight in [
        (
            col_ops,
            _LEAGUE_OPS,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["OPS"],
        ),
        (
            col_obp,
            _LEAGUE_OBP,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["OBP"],
        ),
        (
            col_slg,
            _LEAGUE_SLG,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["SLG"],
        ),
        (
            col_iso,
            _LEAGUE_ISO,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["ISO"],
        ),
    ]:
        if not column:
            continue

        weighted = _weighted_numeric_mean(
            df[column],
            slot_weights,
        )

        if weighted is None:
            continue

        parts.append((
            weighted / anchor,
            metric_weight,
        ))

    if not parts:
        return None

    weight_sum = sum(
        metric_weight
        for _, metric_weight in parts
    )

    if weight_sum <= 0.0:
        return None

    relative = sum(
        value * metric_weight
        for value, metric_weight in parts
    ) / weight_sum

    relative = (
        1.0
        + (relative - 1.0) * _FALLBACK_DAMP
    )

    return relative, "fallback_run_pressure"


def _platoon_split_relative(
    df: pd.DataFrame,
    pitcher_hand: Optional[str],
) -> Optional[Tuple[float, str]]:
    """
    Relative team/lineup strength ~1.0 from same-hand platoon columns, or None.

    Priority (do not skip earlier tiers when data exists):
      1) Advanced splits: vsR/L_wRC+ and vsR/L_wOBA (primary; unchanged from pre-fallback behavior)
      2) Bounded fallback: vsR/L_OPS if present; else mean of available vsR/L_OBP and vsR/L_SLG terms
      3) None only when no usable split signal exists for this pitcher hand
    """
    if df is None or df.empty:
        return None

    # Match existing convention: vs LHP -> vsL_*, else vsR_* (including unknown hand)
    prefix = "vsL" if pitcher_hand == "L" else "vsR"

    col_wrc = f"{prefix}_wRC+" if f"{prefix}_wRC+" in df.columns else None
    col_woba = f"{prefix}_wOBA" if f"{prefix}_wOBA" in df.columns else None
    col_ops = f"{prefix}_OPS" if f"{prefix}_OPS" in df.columns else None
    col_obp = f"{prefix}_OBP" if f"{prefix}_OBP" in df.columns else None
    col_slg = f"{prefix}_SLG" if f"{prefix}_SLG" in df.columns else None
    col_iso = f"{prefix}_ISO" if f"{prefix}_ISO" in df.columns else None

    adv: List[float] = []
    if col_wrc and df[col_wrc].notna().any():
        adv.append(float(np.nanmean(df[col_wrc].values)) / _LEAGUE_WRC_SCALE)
    if col_woba and df[col_woba].notna().any():
        adv.append(float(np.nanmean(df[col_woba].values)) / _LEAGUE_WOBA)

    if adv:
        return float(np.nanmean(adv)), "advanced"

    # Basic split fallback (statSplits supplies OPS/OBP/SLG/ISO but often not wOBA/wRC+).
    # Use a transparent internal run-pressure index instead of OPS alone.
    parts: List[Tuple[float, float]] = []

    if col_ops and df[col_ops].notna().any():
        parts.append((
            float(np.nanmean(df[col_ops].values)) / _LEAGUE_OPS,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["OPS"],
        ))
    if col_obp and df[col_obp].notna().any():
        parts.append((
            float(np.nanmean(df[col_obp].values)) / _LEAGUE_OBP,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["OBP"],
        ))
    if col_slg and df[col_slg].notna().any():
        parts.append((
            float(np.nanmean(df[col_slg].values)) / _LEAGUE_SLG,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["SLG"],
        ))
    if col_iso and df[col_iso].notna().any():
        parts.append((
            float(np.nanmean(df[col_iso].values)) / _LEAGUE_ISO,
            _FALLBACK_RUN_PRESSURE_WEIGHTS["ISO"],
        ))

    if not parts:
        return None

    weight_sum = sum(weight for _, weight in parts)
    if weight_sum <= 0:
        return None

    rel = sum(value * weight for value, weight in parts) / weight_sum
    rel = 1.0 + (rel - 1.0) * _FALLBACK_DAMP
    return rel, "fallback_run_pressure"


class Batters:
    """
    Static helpers for batter table + pitcher hand lookup + team split multipliers.
    """
    _df: Optional[pd.DataFrame] = None
    _hand_map: Optional[Dict[str, str]] = None

    # --------- Table loading ----------
    @staticmethod
    def load_table() -> pd.DataFrame:
        """
        Load batter_stats.csv defensively and normalize columns.
        Expects at least: name_norm, Name, Team, Bats, PA, vsR_wOBA, vsL_wOBA, vsR_wRC+, vsL_wRC+, vsR_ISO, vsL_ISO
        Returns empty DataFrame if file missing or invalid.
        """
        if Batters._df is not None:
            return Batters._df

        if not os.path.exists(BATTER_STATS_CSV):
            logging.warning(f"⚠️ batter_stats.csv not found at {BATTER_STATS_CSV}")
            Batters._df = pd.DataFrame()
            return Batters._df

        try:
            df = pd.read_csv(BATTER_STATS_CSV)
        except Exception as e:
            logging.error(f"❌ Failed to read batter_stats.csv: {e}")
            Batters._df = pd.DataFrame()
            return Batters._df

        # Normalize CSV columns to expected schema (only rename keys that exist)
        _rename = {"name": "Name", "team_name": "Team", "pa": "PA"}
        _rename = {k: v for k, v in _rename.items() if k in df.columns}
        if _rename:
            df = df.rename(columns=_rename)

        # Normalize name column
        if "name_norm" not in df.columns:
            if "Name" in df.columns:
                df["name_norm"] = df["Name"].astype(str).map(_norm)
            else:
                df["name_norm"] = ""

        # Ensure team column exists
        if "Team" not in df.columns:
            df["Team"] = ""

        # Coerce numeric fields if present
        for col in [
            "PA", "vsR_wOBA", "vsL_wOBA", "vsR_wRC+", "vsL_wRC+",
            "vsR_ISO", "vsL_ISO", "wOBA", "wRC+",
            "vsR_OPS", "vsL_OPS", "vsR_OBP", "vsL_OBP", "vsR_SLG", "vsL_SLG",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill sensible defaults for missing common split columns
        defaults = {
            "vsR_wOBA": np.nan, "vsL_wOBA": np.nan,
            "vsR_wRC+": np.nan, "vsL_wRC+": np.nan,
            "vsR_ISO": np.nan,  "vsL_ISO": np.nan,
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v

        # Index by name_norm for quick lookups
        df["name_norm"] = df["name_norm"].astype(str)
        df = df.drop_duplicates(subset=["name_norm"]).set_index("name_norm", drop=True)

        Batters._df = df
        return Batters._df

    # Backward-compat alias expected by overgang_model.py
    @staticmethod
    def load_batter_table() -> pd.DataFrame:
        return Batters.load_table()

    # --------- Pitcher hand lookup ----------
    @staticmethod
    def _load_hand_map() -> Dict[str, str]:
        """
        Load a simple map of pitcher normalized name -> 'R' or 'L' from CSV if present.
        CSV columns: name or name_norm, hand (hand in {'R','L'})
        """
        if Batters._hand_map is not None:
            return Batters._hand_map

        hand_map: Dict[str, str] = {}
        if os.path.exists(PITCHER_HAND_CSV):
            try:
                hdf = pd.read_csv(PITCHER_HAND_CSV)
                name_col = "name_norm" if "name_norm" in hdf.columns else "name"
                hand_col = "hand"
                if name_col in hdf.columns and hand_col in hdf.columns:
                    for _, r in hdf.iterrows():
                        nm = _norm(str(r[name_col]))
                        hv = str(r[hand_col]).strip().upper()
                        if nm and hv in ("R", "L"):
                            hand_map[nm] = hv
            except Exception as e:
                logging.warning(f"⚠️ Could not read pitcher_handedness.csv: {e}")

        Batters._hand_map = hand_map
        return Batters._hand_map

    @staticmethod
    def get_pitcher_hand(pitcher_name: str) -> Optional[str]:
        """
        Return 'R' or 'L' if known; otherwise None.
        """
        if not pitcher_name:
            return None
        nm = _norm(pitcher_name)
        hand_map = Batters._load_hand_map()
        return hand_map.get(nm)

    # --------- Team vs hand multiplier ----------
    @staticmethod
    def offense_vs_hand_dict(
        batter_df: pd.DataFrame,
        team_name: str,
        pitcher_hand: Optional[str],
        lineup_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Dict API for callers that prefer a mapping.
        Returns: {"mult": float, "pop": "team|lineup|none"}
        """
        if not team_name or pitcher_hand not in ("R", "L"):
            return {"mult": 1.0, "pop": "none"}

        # Prefer current active-roster batter_stats.csv for IL/trade correctness.
        # Fall back to team_offense_splits.csv only when the active-roster table is
        # unavailable or cannot produce a usable vs-hand split.
        if batter_df is None or batter_df.empty:
            team_split = _team_offense_split_dict(team_name, pitcher_hand)
            if team_split is not None:
                return team_split
            return {"mult": 1.0, "pop": "none"}

        team_key = _norm(team_name)
        df = batter_df

        # Filter by team if Team column exists and is populated.
        # Use the same _norm() helper as the rest of this module so names like
        # "St. Louis Cardinals" and normalized game/team strings match safely.
        if "Team" in df.columns and df["Team"].notna().any():
            team_df = df[df["Team"].astype(str).map(_norm) == team_key]
            if team_df.empty:
                team_df = df.copy()
        else:
            team_df = df.copy()

        # If a specific lineup is provided, try to filter by those batters.
        used_scope = "team"
        if lineup_names:
            norms = [_norm(n) for n in lineup_names if isinstance(n, str)]
            li_df = team_df.loc[team_df.index.intersection(norms)]
            if not li_df.empty:
                team_df = li_df
                used_scope = "lineup"

        # Reuse the central split scorer so advanced splits are preferred, but
        # OPS/OBP/SLG fallback splits are used when wRC+/wOBA are unavailable.
        rel = _platoon_split_relative(team_df, pitcher_hand)
        if rel is None:
            team_split = _team_offense_split_dict(team_name, pitcher_hand)
            if team_split is not None:
                return team_split
            return {"mult": 1.0, "pop": "none"}

        mult, _source = rel
        mult = max(0.90, min(1.10, float(mult)))  # gentle clamp
        return {"mult": mult, "pop": used_scope}

    @staticmethod
    def offense_vs_hand(
        batter_df: pd.DataFrame,
        team_name: str,
        pitcher_hand: Optional[str],
        lineup_names: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """
        Tuple API for legacy callers that unpack two values:
            mult, scope = Batters.offense_vs_hand(...)
        """
        out = Batters.offense_vs_hand_dict(batter_df, team_name, pitcher_hand, lineup_names)
        return float(out.get("mult", 1.0)), str(out.get("pop", "none"))


class LineupImpact:
    """
    Provides best-9 lookup and lineup scoring against a pitcher hand.
    """
    def __init__(self):
        # Make sure batter table is primed
        self.batter_df = Batters.load_table()

        # try to load a best9 mapping once
        self._best9_map: Dict[str, List[str]] = {}
        if os.path.exists(TEAM_BEST9_JSON):
            try:
                with open(TEAM_BEST9_JSON, "r") as f:
                    raw = json.load(f)
                # normalize keys to lower
                self._best9_map = {_norm(k): v for k, v in raw.items() if isinstance(v, list)}
            except Exception as e:
                logging.warning(f"⚠️ Could not read team_best9.json: {e}")

    # --------- Best-9 ----------
    def get_team_best9(self, team_name: str) -> List[str]:
        """
        Return a list of 9 batter names (or fewer) for a team if known.
        If no curated list, fall back to top-PA batters for that team from the table.
        """
        if not team_name:
            return []

        tkey = _norm(team_name)

        # curated list first
        if tkey in self._best9_map and self._best9_map[tkey]:
            return self._best9_map[tkey][:9]

        # Fall back to top-PA hitters for the requested team only.
        # Normalize both the requested team and CSV team values through the
        # same canonical function so punctuation differences such as
        # "St. Louis" vs "St Louis" cannot break the match.
        df = self.batter_df
        if df is not None and not df.empty and "Team" in df.columns and df["Team"].notna().any():
            team_keys = df["Team"].astype(str).map(_norm)
            subset = df.loc[team_keys == tkey]

            # Fail closed. Never replace an unmatched team with the league-wide
            # batter table, because that can score unrelated hitters as one team.
            if subset.empty:
                logging.warning(
                    "⚠️ No batter rows matched team %r (normalized=%r); "
                    "returning no best9 proxy",
                    team_name,
                    tkey,
                )
                return []

            subset = subset.sort_values(by="PA", ascending=False)
            names_series = subset["Name"] if "Name" in subset.columns else subset.index.to_series()
            return names_series.astype(str).tolist()[:9]

        return []

    # --------- Scoring ----------
    def score_ordered_lineup_dict(
        self,
        lineup: List[Dict[str, Any]],
        pitcher_hand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score a confirmed MLB lineup while preserving batting order.

        Expected input records:
            {
                "slot": 1..9,
                "player_id": MLB player ID,
                "name": player name,
            }

        The scorer fails closed unless:
          - all nine unique batting slots are present,
          - all nine unique player IDs are present,
          - all nine hitters match batter_stats.csv.

        Existing unordered top-PA scoring is intentionally unchanged.
        """
        base = {
            "lineup_impact": 0.0,
            "scope": "none",
            "ordered": False,
            "player_count": 0,
            "matched_count": 0,
            "match_method": "none",
            "split_mode": "none",
        }

        df = self.batter_df

        if df is None or df.empty:
            return base

        if not isinstance(lineup, (list, tuple)):
            return base

        parsed: List[Dict[str, Any]] = []

        for record in lineup:
            if not isinstance(record, dict):
                return base

            try:
                slot = int(record.get("slot"))
                player_id = int(record.get("player_id"))
            except (TypeError, ValueError):
                return base

            name = str(record.get("name") or "").strip()

            if not 1 <= slot <= 9:
                return base

            if player_id <= 0:
                return base

            parsed.append({
                "slot": slot,
                "player_id": player_id,
                "name": name,
            })

        parsed.sort(
            key=lambda record: record["slot"],
        )

        base["player_count"] = len(parsed)

        slots = [
            record["slot"]
            for record in parsed
        ]
        player_ids = [
            record["player_id"]
            for record in parsed
        ]

        if (
            len(parsed) != 9
            or slots != list(range(1, 10))
            or len(set(player_ids)) != 9
        ):
            return base

        numeric_player_ids = (
            pd.to_numeric(
                df["player_id"],
                errors="coerce",
            )
            if "player_id" in df.columns
            else pd.Series(
                np.nan,
                index=df.index,
            )
        )

        matched_rows: List[Dict[str, Any]] = []
        slot_weights: List[float] = []
        methods: List[str] = []

        for record in parsed:
            player_id = record["player_id"]
            name = record["name"]

            matched = df.loc[
                numeric_player_ids == float(player_id)
            ]

            method = "player_id"

            if matched.empty and name:
                normalized_name = _norm(name)

                if normalized_name in df.index:
                    matched = df.loc[[normalized_name]]
                    method = "name"

            if matched.empty:
                continue

            row = matched.iloc[0].copy()
            matched_rows.append(row.to_dict())
            slot_weights.append(
                _LINEUP_SLOT_PA_WEIGHTS[
                    record["slot"]
                ]
            )
            methods.append(method)

        base["matched_count"] = len(matched_rows)

        if methods:
            base["match_method"] = (
                methods[0]
                if len(set(methods)) == 1
                else "mixed"
            )

        # Confirmed source remains fail-closed if even one posted hitter
        # cannot be linked to the current batter table.
        if len(matched_rows) != 9:
            return base

        lineup_df = pd.DataFrame(
            matched_rows
        ).reset_index(drop=True)

        relative_result = (
            _ordered_platoon_split_relative(
                lineup_df,
                pitcher_hand,
                slot_weights,
            )
        )

        if relative_result is None:
            return {
                **base,
                "ordered": True,
                "scope": "none",
            }

        relative, mode = relative_result
        impact = relative - 1.0

        # Basic split fallback is intentionally held to the narrower
        # ±0.20 rail; advanced wRC+/wOBA scoring retains the ±0.30 rail.
        if mode in {"fallback", "fallback_run_pressure"}:
            impact = max(
                -0.20,
                min(0.20, impact),
            )
        else:
            impact = max(
                -0.30,
                min(0.30, impact),
            )

        return {
            **base,
            "lineup_impact": float(impact),
            "scope": "mlb_confirmed_ordered",
            "ordered": True,
            "player_count": 9,
            "matched_count": 9,
            "match_method": (
                methods[0]
                if len(set(methods)) == 1
                else "mixed"
            ),
            "split_mode": mode,
        }

    def score_lineup_dict(
        self,
        lineup: Union[List[str], pd.DataFrame, pd.Series],
        pitcher_hand: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dict API. Returns {"lineup_impact": float, "scope": "lineup|team|none"}.
        """
        df = self.batter_df
        if df is None or df.empty:
            return {"lineup_impact": 0.0, "scope": "none"}

        # normalize lineup into a list of names
        names: List[str] = []
        if isinstance(lineup, (list, tuple)):
            names = [str(x) for x in lineup if isinstance(x, str)]
        elif isinstance(lineup, pd.Series):
            names = [str(x) for x in lineup.tolist()]
        elif isinstance(lineup, pd.DataFrame):
            if "Name" in lineup.columns:
                names = [str(x) for x in lineup["Name"].tolist()]
            else:
                names = [str(x) for x in lineup.index.tolist()]

        norms = [_norm(n) for n in names]
        li_df = df.loc[df.index.intersection(norms)]
        scope = "lineup" if not li_df.empty else "none"
        if li_df.empty:
            return {"lineup_impact": 0.0, "scope": scope}

        # See _platoon_split_relative: advanced wRC+/wOBA first; OPS/OBP/SLG fallback; else neutral
        rel_t = _platoon_split_relative(li_df, pitcher_hand)
        if rel_t is None:
            return {"lineup_impact": 0.0, "scope": "none"}

        rel, mode = rel_t
        impact = rel - 1.0
        if mode == "fallback":
            impact = max(-0.20, min(0.20, impact))
        else:
            impact = max(-0.30, min(0.30, impact))
        return {"lineup_impact": impact, "scope": scope}

    def score_lineup(
        self,
        lineup: Union[List[str], pd.DataFrame, pd.Series],
        pitcher_hand: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Tuple API for legacy callers that unpack two values:
            impact, scope = LineupImpact().score_lineup(...)
        """
        out = self.score_lineup_dict(lineup, pitcher_hand)
        return float(out.get("lineup_impact", 0.0)), str(out.get("scope", "none"))


# ---------- Module-level convenience for the predictor ----------
# Allows: from core.batters import Batters, BATTER_DF
BATTER_DF: pd.DataFrame = Batters.load_table()

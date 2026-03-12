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
            "vsR_ISO", "vsL_ISO", "wOBA", "wRC+"
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
        if batter_df is None or batter_df.empty or not team_name or pitcher_hand not in ("R", "L"):
            return {"mult": 1.0, "pop": "none"}

        team_key = _norm(team_name)
        df = batter_df

        # Filter by team if Team column exists and is populated
        if "Team" in df.columns and df["Team"].notna().any():
            team_df = df[df["Team"].astype(str).str.lower() == team_key]
            if team_df.empty:
                team_df = df.copy()
        else:
            team_df = df.copy()

        # If a specific lineup is provided, try to filter by those batters
        used_scope = "team"
        if lineup_names:
            norms = [_norm(n) for n in lineup_names if isinstance(n, str)]
            li_df = team_df.loc[team_df.index.intersection(norms)]
            if not li_df.empty:
                team_df = li_df
                used_scope = "lineup"

        # choose split columns for the hand
        if pitcher_hand == "R":
            col_woba = "vsR_wOBA" if "vsR_wOBA" in team_df.columns else None
            col_wrc = "vsR_wRC+" if "vsR_wRC+" in team_df.columns else None
        else:
            col_woba = "vsL_wOBA" if "vsL_wOBA" in team_df.columns else None
            col_wrc = "vsL_wRC+" if "vsL_wRC+" in team_df.columns else None

        vals = []
        if col_wrc and team_df[col_wrc].notna().any():
            vals.append(np.nanmean(team_df[col_wrc].values) / 100.0)  # 100 -> 1.0
        if col_woba and team_df[col_woba].notna().any():
            vals.append(np.nanmean(team_df[col_woba].values) / 0.310)  # ~league wOBA

        if not vals:
            return {"mult": 1.0, "pop": "none"}

        mult = float(np.nanmean(vals))
        mult = max(0.90, min(1.10, mult))  # gentle clamp
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

        # fall back to top-PA by team if the Team column exists
        df = self.batter_df
        if df is not None and not df.empty and "Team" in df.columns and df["Team"].notna().any():
            subset = df[df["Team"].astype(str).str.lower() == tkey]
            if subset.empty:
                subset = df.copy()
            subset = subset.sort_values(by="PA", ascending=False)
            names_series = subset["Name"] if "Name" in subset.columns else subset.index.to_series()
            return names_series.astype(str).tolist()[:9]

        return []

    # --------- Scoring ----------
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

        # choose columns based on pitcher hand
        if pitcher_hand == "L":
            col_wrc = "vsL_wRC+" if "vsL_wRC+" in li_df.columns else None
            col_woba = "vsL_wOBA" if "vsL_wOBA" in li_df.columns else None
        else:
            col_wrc = "vsR_wRC+" if "vsR_wRC+" in li_df.columns else None
            col_woba = "vsR_wOBA" if "vsR_wOBA" in li_df.columns else None

        metrics = []
        if col_wrc and li_df[col_wrc].notna().any():
            metrics.append(np.nanmean(li_df[col_wrc].values) / 100.0)  # 100 -> 1.0
        if col_woba and li_df[col_woba].notna().any():
            metrics.append(np.nanmean(li_df[col_woba].values) / 0.310)  # ~league wOBA

        if not metrics:
            return {"lineup_impact": 0.0, "scope": "none"}

        rel = float(np.nanmean(metrics))  # around 1.0 = league average
        impact = rel - 1.0
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

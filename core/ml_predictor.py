# ================================
# 💵 Over Gang ML Predictor
# ================================
import pandas as pd
from core.utils import normalize_name
from pybaseball import standings


LEAGUE_AVG_STARTER_XERA = 4.30
LEAGUE_AVG_BULLPEN_ERA = 4.10
LEAGUE_AVG_BULLPEN_XERA = 4.10
LEAGUE_AVG_BULLPEN_IP_WEEK = 17.8
LEAGUE_AVG_BULLPEN_RELIEVERS = 14.0
BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK = 3.5


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _league_avg_bullpen_defaults(bullpen_df: pd.DataFrame) -> dict:
    defaults = {
        "bullpen_era": LEAGUE_AVG_BULLPEN_ERA,
        "bullpen_xera": LEAGUE_AVG_BULLPEN_XERA,
        "bullpen_ip_week": LEAGUE_AVG_BULLPEN_IP_WEEK,
        "bullpen_relievers": LEAGUE_AVG_BULLPEN_RELIEVERS,
    }
    try:
        row = bullpen_df.loc["league average"]
    except Exception:
        return defaults
    defaults["bullpen_era"] = _safe_float(row.get("ERA"), defaults["bullpen_era"])
    defaults["bullpen_xera"] = _safe_float(row.get("xERA"), defaults["bullpen_xera"])
    defaults["bullpen_ip_week"] = _safe_float(row.get("IP_Week"), defaults["bullpen_ip_week"])
    defaults["bullpen_relievers"] = _safe_float(row.get("Relievers"), defaults["bullpen_relievers"])
    return defaults


def _bullpen_fatigue_ratio(team: dict) -> float:
    relievers = max(1.0, _safe_float(team.get("bullpen_relievers"), LEAGUE_AVG_BULLPEN_RELIEVERS))
    ip_week = max(0.0, _safe_float(team.get("bullpen_ip_week"), LEAGUE_AVG_BULLPEN_IP_WEEK))
    expected_weekly_ip = relievers * BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK
    if expected_weekly_ip <= 0:
        return 1.0
    return ip_week / expected_weekly_ip


def get_pyth_win_pct(team_name):
    """
    Pulls real team win-loss record and calculates Pythagorean Win%.
    """
    try:
        df = standings()
        df['Team'] = df['Team'].str.lower().str.strip()

        row = df[df['Team'].str.contains(team_name.lower())]
        if row.empty:
            return 0.500  # fallback

        wins = int(row['W'].values[0])
        losses = int(row['L'].values[0])
        return round(wins / (wins + losses), 3)
    except:
        return 0.500

def calculate_team_win_probability(home: dict, away: dict) -> tuple:
    """
    Estimate win probabilities using team quality, starter skill, and bullpen quality/freshness.

    Inputs:
    - home: dict with keys: 'pyth_win', 'starter_xera', 'bullpen_era',
      'bullpen_xera', 'bullpen_ip_week', 'bullpen_relievers'
    - away: same structure

    Returns:
    - (home_win_prob, away_win_prob)
    """
    # Normalize each factor (lower xERA / bullpen ERA is better)
    pyth_score = home['pyth_win'] - away['pyth_win']  # Higher = better
    xera_score = away['starter_xera'] - home['starter_xera']  # Lower = better
    pen_score = away['bullpen_era'] - home['bullpen_era']     # Lower = better
    pen_xera_score = away['bullpen_xera'] - home['bullpen_xera']  # Lower = better
    home_fatigue = _bullpen_fatigue_ratio(home)
    away_fatigue = _bullpen_fatigue_ratio(away)
    freshness_score = max(-1.0, min(1.0, away_fatigue - home_fatigue))

    # Combine weighted score
    score = (
        0.45 * pyth_score +
        0.3 * (xera_score / 5) +
        0.15 * (pen_score / 5) +
        0.05 * (pen_xera_score / 5) +
        0.05 * freshness_score
    )

    # Convert to probability using logistic-like function
    home_win_prob = 1 / (1 + 10 ** (-score * 5))
    away_win_prob = 1 - home_win_prob

    return round(home_win_prob, 3), round(away_win_prob, 3)

def get_team_ml_data(team_name: str, pitcher_name: str) -> dict:
    # Normalize names
    team_name = normalize_name(team_name)
    pitcher_name = normalize_name(pitcher_name)

    # Load stats
    pitcher_df = pd.read_csv("data/pitcher_stats.csv", index_col="Name")
    pitcher_df.index = pitcher_df.index.astype(str).map(normalize_name)
    bullpen_df = pd.read_csv("data/bullpen_stats.csv", index_col="Team")
    bullpen_df.index = bullpen_df.index.astype(str).map(normalize_name)
    bullpen_defaults = _league_avg_bullpen_defaults(bullpen_df)

    # Starter xERA
    try:
        xera = pitcher_df.loc[pitcher_name]["xERA"]
    except:
        xera = LEAGUE_AVG_STARTER_XERA  # league average fallback

    # Bullpen quality + availability
    try:
        pen_row = bullpen_df.loc[team_name]
    except:
        pen_row = {}
    if not hasattr(pen_row, "get"):
        pen_row = {}

    pen_era = _safe_float(pen_row.get("ERA"), bullpen_defaults["bullpen_era"])
    pen_xera = _safe_float(pen_row.get("xERA"), bullpen_defaults["bullpen_xera"])
    pen_ip_week = _safe_float(pen_row.get("IP_Week"), bullpen_defaults["bullpen_ip_week"])
    pen_relievers = _safe_float(pen_row.get("Relievers"), bullpen_defaults["bullpen_relievers"])

    pyth = get_pyth_win_pct(team_name)

    return {
        "starter_xera": float(xera),
        "bullpen_era": float(pen_era),
        "bullpen_xera": float(pen_xera),
        "bullpen_ip_week": float(pen_ip_week),
        "bullpen_relievers": float(pen_relievers),
        "pyth_win": float(pyth)
    }
# ==========================
# 💰 Kelly Criterion Sizing
# ==========================
def calculate_kelly_units(prob: float, implied: float, bankroll_fraction: float = 1.0) -> float:
    """
    Kelly Criterion: Optimal bet sizing based on edge and odds.
    """
    edge = prob - implied
    if edge <= 0:
        return 0.0
    return round((edge / (1 - implied)) * bankroll_fraction, 2)

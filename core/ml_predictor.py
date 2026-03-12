# ================================
# 💵 Over Gang ML Predictor
# ================================
import pandas as pd
from core.utils import normalize_name
from pybaseball import standings

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
    Estimate win probabilities using basic team metrics.

    Inputs:
    - home: dict with keys: 'pyth_win', 'starter_xera', 'bullpen_era'
    - away: same structure

    Returns:
    - (home_win_prob, away_win_prob)
    """
    # Normalize each factor (lower xERA / bullpen ERA is better)
    pyth_score = home['pyth_win'] - away['pyth_win']  # Higher = better
    xera_score = away['starter_xera'] - home['starter_xera']  # Lower = better
    pen_score = away['bullpen_era'] - home['bullpen_era']     # Lower = better

    # Combine weighted score
    score = (
        0.5 * pyth_score +
        0.3 * (xera_score / 5) +
        0.2 * (pen_score / 5)
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
    bullpen_df = pd.read_csv("data/bullpen_stats.csv", index_col="Team")

    # Starter xERA
    try:
        xera = pitcher_df.loc[pitcher_name]["xERA"]
    except:
        xera = 4.30  # league average fallback

    # Bullpen ERA
    try:
        pen_era = bullpen_df.loc[team_name]["ERA"]
    except:
        pen_era = 4.10  # league average fallback

    pyth = get_pyth_win_pct(team_name)

    return {
        "starter_xera": float(xera),
        "bullpen_era": float(pen_era),
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

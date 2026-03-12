def calculate_kelly_units(prob: float, implied: float, bankroll_fraction: float = 1.0) -> float:
    """
    Calculate the number of units to bet using the Kelly Criterion.
    :param prob: Estimated probability of winning (0 to 1)
    :param implied: Implied probability from the odds (0 to 1)
    :param bankroll_fraction: Portion of bankroll you're willing to risk
    :return: Units to bet (can be negative for fading)
    """
    edge = prob - implied
    if implied == 0:
        return 0.0
    kelly = edge / (1 - implied)
    return round(max(kelly * bankroll_fraction, 0), 2)

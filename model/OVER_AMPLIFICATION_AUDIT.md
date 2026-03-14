# Audit: Over-amplification after ERA formula fix (totals 12.8–15.1, per-team 6.3–8.0)

## 1. Exact suspicious lines/blocks in `model/overgang_model.py`

### A. **Base runs formula — no regression toward league**

**Lines 140–144:**

```python
    effective_era = (
        STARTER_IP_SHARE * opponent_starter_xera + BULLPEN_IP_SHARE * opponent_bullpen_era
    )
    effective_era = max(2.5, min(7.0, effective_era))
    runs = LEAGUE_RUNS_PER_TEAM * (effective_era / LEAGUE_ERA)
```

**Why it over-projects:**  
`runs = 4.25 * (effective_era / 4.25) = effective_era`, so the base is exactly the opponent’s blended ERA. That’s the right *direction*, but there’s no regression to the mean. So a staff with effective_era 5.5 gives a 5.5-run base before any multipliers. In reality, run environment and sample size usually justify pulling the base slightly toward league (4.25). Using `effective_era` directly with no scaling makes the base too sensitive to high xERA/bullpen ERA and amplifies the effect of the following multipliers.

---

### B. **Six multipliers applied in sequence (main over-amplification)**

**Lines 146, 148, 151, 155, 159–160, 163–165:**

```python
    runs *= park_runs_factor

    offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(offense_mult)))
    runs *= offense_mult

    lineup_mult = 1.0 + max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, lineup_impact))
    runs *= lineup_mult

    whip_mult = opponent_starter_whip / WHIP_LEAGUE
    whip_mult = max(0.90, min(1.15, whip_mult))
    runs *= whip_mult

    if opponent_velo_drop < 0:
        velo_mult = 1.0 - (VELO_DROP_RUNS_PER_MPH * opponent_velo_drop)
        runs *= min(1.10, velo_mult)

    if opponent_bullpen_ip_week > BULLPEN_FATIGUE_IP_CUTOFF:
        ...
        runs *= min(1.08, fatigue_mult)
```

**Why they over-project:**  
Each factor is a multiplier on top of the base. Even modest values compound:

- **Park:** up to 1.25 (Coors), often 1.08–1.15 for hitter parks.
- **Offense:** up to 1.10 (when batter data is present).
- **Lineup:** 1.0 when impacts are 0; up to 1.10 when present.
- **WHIP:** up to 1.15 (bad opponent WHIP).
- **Velo:** up to 1.10 (tired opponent starter).
- **Fatigue:** up to 1.08 (overworked bullpen).

Example: base 5.0 × 1.12 (park) × 1.05 (offense) × 1.0 (lineup) × 1.08 (whip) × 1.04 (velo) × 1.04 (fatigue) ≈ 5.0 × 1.35 ≈ **6.75** per team → game total **13.5**. So the same formula that gives ~4.25 for a league-average matchup with no boosts quickly reaches 6.3–8.0 per team when several factors are slightly above 1.0. The **stacking of six multipliers** is the main reason totals land in the 12.8–15.1 range.

---

### C. **Park factor applied to both teams**

**Lines 353, 362, 374:**  
`over_boost` (park_runs_factor) is passed for both away and home. So both teams get the same park multiplier (e.g. 1.25 at Coors). That’s correct for venue, but it doubles the park effect on the **game total** (both sides boosted), which is appropriate for a total; the issue is the combination with the other five multipliers above, not park alone.

---

### D. **No cap on final runs per team**

**Line 166:**  
`return round(runs, 2)` — there is no upper (or lower) bound on the returned runs. So after base + six multipliers, a team can exceed 7 or 8 runs. A simple cap (e.g. max 6.5 or 7.0 per team) would prevent extreme game totals even when inputs are noisy or multipliers stack.

---

### E. **Low-IP penalty only raises xERA (already high base)**

**Lines 137–138:**  
`if opponent_low_ip: opponent_starter_xera = min(opponent_starter_xera + LOW_IP_XERA_PENALTY, 6.0)`  
So we push xERA up (max 6.0), which raises effective_era and thus the base. That correctly flags uncertain/bad starters, but in a model that already over-projects when effective_era is high, the low-IP bump adds more upside and no dampening. It’s a contributor, not the main driver.

---

## 2. Why per-team runs are now 6.3–8.0

- **Base:** With effective_era often in the 4.5–5.5 range (starter + bullpen blend), base is already 4.5–5.5.
- **Compound multipliers:** Park (1.0–1.25), offense (up to 1.10), lineup (1.0), WHIP (up to 1.15), velo (up to 1.10), fatigue (up to 1.08). A typical “slightly favorable” product is ~1.2–1.35.
- So: 5.0 × 1.25 ≈ 6.25, or 5.5 × 1.35 ≈ 7.4. That matches the 6.3–8.0 band. The fix is not to revert the ERA formula, but to **scale or cap** the base and/or **limit how much the multipliers can compound**.

---

## 3. Whether the ERA formula should be scaled differently

**Current:**  
`runs = LEAGUE_RUNS_PER_TEAM * (effective_era / LEAGUE_ERA)` → base = effective_era.

**Options:**

- **Keep as-is but regress base toward league:**  
  e.g. `base = 0.85 * effective_era + 0.15 * LEAGUE_ERA`, then `runs = base` (or `runs = LEAGUE_RUNS_PER_TEAM * (base / LEAGUE_ERA)`). So a 5.5 effective_era becomes 5.36 instead of 5.5, and extremes are pulled toward 4.25.

- **Keep formula, cap base before multipliers:**  
  e.g. `runs = min(LEAGUE_RUNS_PER_TEAM * (effective_era / LEAGUE_ERA), 6.0)` so no team gets a base above 6.0 before park/offense/etc.

- **Scale down the ratio:**  
  e.g. `runs = LEAGUE_RUNS_PER_TEAM * (0.9 * (effective_era / LEAGUE_ERA) + 0.1)` so the base is slightly dampened and never far above league.

So the formula *can* be scaled (regression, cap, or dampening) to reduce over-projection without changing the correct “worse pitching → more runs” direction.

---

## 4. Contribution of each factor (audit)

| Factor | Lines | Typical range | Effect when stacked |
|--------|--------|----------------|---------------------|
| **Park** | 146, 353, 362, 374 | 0.85–1.25 | High in Coors; often 1.05–1.15. Applied to both teams. |
| **offense_mult** | 148 | 0.90–1.10 | Up to +10%. When 1.0 (no data), no change. |
| **lineup_impact** | 151 | 1.0 when 0.000 | Currently no boost (all 0). If it were 0.10, +10%. |
| **WHIP** | 154–156 | 0.90–1.15 | Bad opponent WHIP can add up to +15%. |
| **Velo** | 158–160 | 1.0–1.10 | Tired opponent adds up to +10%. |
| **Bullpen fatigue** | 162–165 | 1.0–1.08 | Overworked bullpen adds up to +8%. |
| **Low-IP penalty** | 137–138 | Raises xERA, max 6.0 | Pushes base up for uncertain starters. |

None of these alone is wrong; the issue is **all of them multiplying together** with no overall cap or regression.

---

## 5. Minimal patch plan (order)

1. **Regress base toward league in `project_team_runs()` (lines 143–144)**  
   After computing `effective_era`, compute a regressed base, e.g.:  
   `base_era = 0.88 * effective_era + 0.12 * LEAGUE_ERA`  
   then  
   `runs = LEAGUE_RUNS_PER_TEAM * (base_era / LEAGUE_ERA)`.  
   (Coefficients 0.88/0.12 are tunable; goal is to pull 5.5 toward ~5.0 and avoid 6+ bases.)

2. **Cap total runs per team before return (line 166)**  
   Before `return round(runs, 2)`, add e.g. `runs = min(runs, 6.5)` (or 6.0) so that no single team projection exceeds 6.5 (or 6.0), preventing 15+ game totals even when multipliers stack.

3. **(Optional) Cap the product of multipliers**  
   Instead of applying park, offense, lineup, WHIP, velo, fatigue in sequence, compute a combined multiplier and cap it (e.g. max 1.20 or 1.25) so that the six factors cannot compound beyond a set amount. Requires a slightly larger refactor of the multiplier block.

4. **(Optional) Slightly tighten individual caps**  
   Reduce velo cap from 1.10 to 1.06, fatigue cap from 1.08 to 1.05, and/or WHIP ceiling from 1.15 to 1.10 so that each factor contributes less when stacked.

**Recommended first step:** (1) regress base toward league, and (2) cap runs per team at 6.0 or 6.5. That keeps the correct ERA direction and scale while bringing totals back into a realistic 7.5–10.5 range without touching other files or other logic in the script.

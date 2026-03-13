# v4.0 Projection Constants Audit — Over-Amplification

## Multiplicative stack (per team)

```
runs = base × park × offense_mult × lineup_mult × whip_mult × velo_mult × fatigue_mult
```

| Factor | Current range | Max contribution (favorable) |
|--------|----------------|------------------------------|
| Park | 0.85–1.25 | 1.25 |
| Offense mult | 0.90–1.10 | 1.10 |
| Lineup | 1 ± LINEUP_IMPACT_CAP (0.15) | 1.15 |
| WHIP | 0.85–1.20 (hard clamp) | 1.20 |
| Velo | 1.0 to 1.10 cap | 1.10 |
| Fatigue | 1.0 to 1.08 cap | 1.08 |

**Theoretical max product (one team):** 1.25 × 1.10 × 1.15 × 1.20 × 1.10 × 1.08 ≈ **2.16×** base.  
**Typical “all favorable”:** e.g. 1.10 × 1.05 × 1.08 × 1.10 × 1.04 × 1.04 ≈ **1.45×** base.

---

## Findings

### 1. Offense + lineup can stack

- **offense_mult** (platoon / batter splits) and **lineup_impact** (lineup strength) both describe “how good is this offense in this matchup.”
- Current: offense up to 1.10, lineup up to 1.15 → combined up to **1.265** from offense/lineup alone.
- Risk: In strong platoon + strong lineup games, projections can be pushed up more than intended.

### 2. LINEUP_IMPACT_CAP (0.15)

- ±15% is large now that offense_mult exists.
- Recommendation: **0.10** (±10%) so offense + lineup together stay in a more conservative band (e.g. 1.10 × 1.10 = 1.21 max).

### 3. OFFENSE_MULT_MIN / OFFENSE_MULT_MAX (0.90 / 1.10)

- ±10% is reasonable; with lineup at ±10% the combined cap is 1.10 × 1.10 = 1.21.
- Optional tightening: **0.92 / 1.08** if you want to be more conservative; not required.

### 4. VELO_DROP_RUNS_PER_MPH (0.02), cap 1.10

- 2% per mph drop, max +10%. Reasonable; can slightly reduce sensitivity.
- Optional: **0.015** and cap **1.06** to avoid stacking with other boosts.

### 5. BULLPEN_FATIGUE_RUNS_BOOST (0.03), cap 1.08

- +3% per 10 IP over 12, max +8%. Fine as-is.
- Optional: cap at **1.05** if fatigue is noisy.

### 6. STARTER_IP_SHARE / BULLPEN_IP_SHARE (0.60 / 0.40)

- Standard and not a source of over-amplification. **No change.**

### 7. WHIP clamp (0.85–1.20)

- 1.20 allows +20% from WHIP alone; 0.85 allows -15%.
- With other multipliers, 1.20 can stack high.
- Recommendation: **0.90–1.15** so WHIP contributes at most ±15%.

---

## Tuning recommendation (concise)

| Constant | Current | Safer default | Rationale |
|----------|---------|----------------|------------|
| **LINEUP_IMPACT_CAP** | 0.15 | **0.10** | Avoid stacking with offense_mult; keep “offense + lineup” combined cap ~1.21. |
| **OFFENSE_MULT_MIN / MAX** | 0.90 / 1.10 | Keep or **0.92 / 1.08** | Optional; 0.90/1.10 is OK if lineup is reduced. |
| **WHIP clamp** | 0.85, 1.20 | **0.90, 1.15** | Limit WHIP contribution so it doesn’t over-stack with park/offense/lineup. |
| **VELO_DROP_RUNS_PER_MPH** | 0.02, cap 1.10 | Keep or **0.015**, cap **1.06** | Optional; small reduction in velo sensitivity. |
| **BULLPEN_FATIGUE** | 0.03, cap 1.08 | Keep or cap **1.05** | Optional; slightly lower max fatigue effect. |
| **STARTER_IP_SHARE / BULLPEN_IP_SHARE** | 0.60 / 0.40 | **No change** | Not a source of over-amplification. |

**Priority:** Reduce **LINEUP_IMPACT_CAP** to **0.10** and tighten **WHIP** to **0.90–1.15** first; then optionally narrow offense_mult and velo/fatigue caps if totals still look hot.

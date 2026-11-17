# Phase 2.1 — Scouting Signals

These are **descriptive monitoring flags**, not scouting grades, probabilities, causal effects, or estimates of market undervaluation.

## Hitter signals
Compare consecutive seasons within the same competition (MLB vs MiLB). Require >=50 batted balls with measured exit velocity **in each season**. Positive: mean exit velocity +2 mph **and** hard-hit rate +5 percentage points. Negative: both declines of the same magnitude. Otherwise: no material change. Missing or inadequate history: insufficient evidence.

## Pitcher signals
Require >=200 measured pitch velocities and >=100 observed swings **in each season**. Positive: average pitch velocity +1 mph **and** whiff rate +3 percentage points. Negative: both declines of the same magnitude. Otherwise: no material change.

## Out-of-time check (added 2026-11-17, exploratory)
Using MLB season leaderboards 2017–2026: hitters with a POSITIVE_CHANGE flag still **fell back ~15 wOBA points**
the next season (they did not improve), but ~5 points **less than Marcel expected** (p = 0.04, d = 0.17). The
NEGATIVE_CHANGE flag ran −5 points vs Marcel (p = 0.10, not significant). Present the positive flag as
"projection may be low", never as "breakout". Details: `docs/PHASE3_RESULTS.md`.

## Critical limitations
- MiLB coverage is incomplete and variable by year, park and level; do not interpret missing tracking as lack of talent.
- MiLB levels are not yet separated in these tables; do **not** treat a MiLB signal as comparable across A/AA/AAA or as decision-grade until level-specific keys and verified coverage are implemented.
- Velocity across *all pitch types* changes when pitch mix changes. Phase 2.2 should calculate pitch-type-specific velocity, movement and usage before interpreting pitcher flags.
- Hitter launch-speed samples and pitch-level whiff samples are tracking-dependent; thresholds are preliminary analyst settings, not empirically calibrated cutoffs.
- No age, park, opponent quality, role, injury, scouting observation or cost adjustment yet.
- These flags must not be described as breakout forecasts or as proof that a player is undervalued.
- The 2026 strike-zone/location definition changes require harmonization before location-based features are introduced.

## Execution
```bash
python -m src.build_db
python -m src.build_features
python -m src.build_signals
```

Inspect results:
```sql
SELECT * FROM scouting_signal_hitter
WHERE competition = 'MLB' AND signal = 'POSITIVE_CHANGE'
ORDER BY ev_delta_mph DESC;

SELECT * FROM scouting_signal_pitcher
WHERE competition = 'MLB' AND signal = 'POSITIVE_CHANGE'
ORDER BY velocity_delta_mph DESC;
```

**Next:** verified MiLB level, pitch-type-level pitcher features, park/age context, signal stability and out-of-time evaluation.

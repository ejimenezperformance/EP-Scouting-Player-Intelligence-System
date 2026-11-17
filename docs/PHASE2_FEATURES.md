# Phase 2 — descriptive player-season features (initial version)

Run after Phase 1 ingestion and database build:

```bash
python -m src.build_features
python -m pytest -q
```

## Tables

- `feature_hitter_season`: one row per batter, competition, calendar season.
- `feature_pitcher_season`: one row per pitcher, competition, calendar season.

## Definitions and safeguards

- `tracked_pitches` is the count of pitch records **observed in this source**, not guaranteed full-season totals.
- `batted_balls_with_ev` counts records with non-null `launch_speed`; `hard_hit_rate` is the share of those records with exit velocity >=95 mph. Missing exit velocities are excluded from its denominator.
- `avg_estimated_woba_on_contact` averages source `estimated_woba_using_speedangle` on records where present. It is **not** player xwOBA across all plate appearances.
- `whiff_rate_on_swings` divides whiff descriptions by selected swing descriptions; its classification should be audited against current Statcast definitions before executive use.
- Pitcher speed/spin averages include every pitch with available values, irrespective of pitch type. **Do not compare pitch-mix-weighted velocity changes as mechanical improvements.** Pitch-type-specific features are a next step.
- No barrel rate is output until the exact Baseball Savant barrel definition is implemented and validated. A simplified exit velocity/launch angle rule is not an official barrel definition.
- `competition` distinguishes MLB and MiLB, but MiLB level is not yet separated. **Do not rank MLB against MiLB or combine minor-league levels as though they are equivalent.**
- No age, park, level, competition-strength, season coverage, or 2026 ABS location corrections are implemented yet.
- These are descriptive aggregates, **not** breakout probabilities or scouting grades.

## Next gate

Verify live ingestion; audit duplicate pitch records, event taxonomies, source coverage, game identifiers, level labels, and sample thresholds. Then build pitch-type / level adjusted features and forward-looking breakout targets.

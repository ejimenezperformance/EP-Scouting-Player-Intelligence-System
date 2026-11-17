# Phase 1 validation gate

## Local smoke test

Run from the project root after installing dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install --pre sportsdataverse
python -m pytest -q
python -m src.ingest_statcast --start-date 2025-04-01 --end-date 2025-04-02 --competitions MLB --chunk-days 2
python -m src.build_db
python -m src.validate_db
```

MiLB should be tested separately once the optional beta adapter is installed:

```bash
python -m src.ingest_statcast --start-date 2025-04-01 --end-date 2025-04-02 --competitions MiLB --chunk-days 2
python -m src.build_db
python -m src.validate_db
```

## Acceptance criteria

1. Every raw CSV has a successful matching manifest with matching row count and competition.
2. `fact_statcast_pitch` contains a nonzero number of rows for an in-season test period.
3. Competition is populated and never inferred from a mixed query.
4. Player IDs and dates are inspected for missingness.
5. MLB and MiLB coverage is reported separately; MiLB is not treated as a census.
6. 2026 plate location and ABS strike-zone definitions are explicitly versioned before model features are built.
7. No predictive claims until forward-looking labels and temporal validation exist.

## Execution status

Verified live on 2026-11-17 (April 1–2, 2025 window):
- MLB: 7,999 pitches, 351 batters, 221 pitchers; manifest row count matched.
- MiLB (sportsdataverse 0.2.0b1, beta): 8,543 pitches, 327 batters; manifest row count matched.
- `build_db`, `validate_db`, `build_features`, `build_signals` ran end to end; 0 null dates / batters / pitchers.
- `pytest`: all tests pass (one fixture fix: the second row of `test_feature_sql.py` now casts its date to DATE).
- Gap: MiLB rows carry team codes but **no level field**; level must be mapped from team IDs before any MiLB comparison.

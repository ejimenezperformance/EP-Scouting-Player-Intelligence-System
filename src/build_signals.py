"""Phase 2.1: transparent year-over-year scouting signals, not predictions.

A signal requires adjacent seasons, matching competition, sufficient observations,
and a material change. Missing values never become zero. This is a watchlist,
not a scouting grade, breakout probability, or proof of undervaluation.
"""
import duckdb
from src.config import DB_PATH


def build(con):
    tables = {row[0] for row in con.execute('SHOW TABLES').fetchall()}
    required = {'feature_hitter_season', 'feature_pitcher_season'}
    if not required.issubset(tables):
        raise RuntimeError('Missing feature tables: run python -m src.build_features')

    con.execute('''
    CREATE OR REPLACE TABLE scouting_signal_hitter AS
    WITH history AS (
      SELECT *,
        LAG(season) OVER w AS previous_season,
        LAG(batted_balls_with_ev) OVER w AS previous_ev_samples,
        LAG(avg_exit_velocity) OVER w AS previous_avg_ev,
        LAG(hard_hit_rate) OVER w AS previous_hard_hit_rate
      FROM feature_hitter_season
      WINDOW w AS (PARTITION BY player_id, competition ORDER BY season)
    ), changes AS (
      SELECT *,
        avg_exit_velocity - previous_avg_ev AS ev_delta_mph,
        (hard_hit_rate - previous_hard_hit_rate) * 100 AS hard_hit_delta_pp,
        (season = previous_season + 1
          AND batted_balls_with_ev >= 50 AND previous_ev_samples >= 50) AS eligible
      FROM history
    )
    SELECT player_id, competition, season, previous_season,
           batted_balls_with_ev, previous_ev_samples,
           ROUND(avg_exit_velocity, 2) AS avg_exit_velocity,
           ROUND(ev_delta_mph, 2) AS ev_delta_mph,
           ROUND(hard_hit_rate * 100, 2) AS hard_hit_pct,
           ROUND(hard_hit_delta_pp, 2) AS hard_hit_delta_pp,
           eligible,
           CASE
             WHEN NOT eligible OR eligible IS NULL THEN 'INSUFFICIENT_EVIDENCE'
             WHEN ev_delta_mph >= 2 AND hard_hit_delta_pp >= 5 THEN 'POSITIVE_CHANGE'
             WHEN ev_delta_mph <= -2 AND hard_hit_delta_pp <= -5 THEN 'NEGATIVE_CHANGE'
             ELSE 'NO_MATERIAL_CHANGE'
           END AS signal,
           'Descriptive year-over-year signal; not a breakout probability' AS interpretation
    FROM changes
    ''')
    con.execute('''
    CREATE OR REPLACE TABLE scouting_signal_pitcher AS
    WITH history AS (
      SELECT *,
        LAG(season) OVER w AS previous_season,
        LAG(velocity_samples) OVER w AS previous_velocity_samples,
        LAG(avg_release_speed) OVER w AS previous_avg_speed,
        LAG(observed_swings) OVER w AS previous_swings,
        LAG(whiff_rate_on_swings) OVER w AS previous_whiff_rate
      FROM feature_pitcher_season
      WINDOW w AS (PARTITION BY player_id, competition ORDER BY season)
    ), changes AS (
      SELECT *,
        avg_release_speed - previous_avg_speed AS velocity_delta_mph,
        (whiff_rate_on_swings - previous_whiff_rate) * 100 AS whiff_delta_pp,
        (season = previous_season + 1
          AND velocity_samples >= 200 AND previous_velocity_samples >= 200
          AND observed_swings >= 100 AND previous_swings >= 100) AS eligible
      FROM history
    )
    SELECT player_id, competition, season, previous_season,
           velocity_samples, previous_velocity_samples,
           observed_swings, previous_swings,
           ROUND(avg_release_speed, 2) AS avg_release_speed,
           ROUND(velocity_delta_mph, 2) AS velocity_delta_mph,
           ROUND(whiff_rate_on_swings * 100, 2) AS whiff_pct,
           ROUND(whiff_delta_pp, 2) AS whiff_delta_pp,
           eligible,
           CASE
             WHEN NOT eligible OR eligible IS NULL THEN 'INSUFFICIENT_EVIDENCE'
             WHEN velocity_delta_mph >= 1 AND whiff_delta_pp >= 3 THEN 'POSITIVE_CHANGE'
             WHEN velocity_delta_mph <= -1 AND whiff_delta_pp <= -3 THEN 'NEGATIVE_CHANGE'
             ELSE 'NO_MATERIAL_CHANGE'
           END AS signal,
           'Pitch-mix changes may affect mean velocity; review by pitch type' AS interpretation
    FROM changes
    ''')
    return {name: con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
            for name in ('scouting_signal_hitter', 'scouting_signal_pitcher')}


def main():
    if not DB_PATH.exists():
        raise SystemExit('Database not found: complete Phase 1 first')
    con = duckdb.connect(str(DB_PATH))
    try:
        for name, count in build(con).items():
            print(f'{name}: {count} player-season records')
    finally:
        con.close()


if __name__ == '__main__':
    main()

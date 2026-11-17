"""Build descriptive player-season tables. No predictive claims or scouting grades."""
import duckdb
from src.config import DB_PATH


def build(con):
    tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
    if 'fact_statcast_pitch' not in tables:
        raise RuntimeError('Missing fact_statcast_pitch: run python -m src.build_db first')
    con.execute('''
    CREATE OR REPLACE TABLE feature_hitter_season AS
    WITH events AS (
      SELECT batter AS player_id, competition,
             EXTRACT(YEAR FROM game_date)::INTEGER AS season,
             game_pk, game_date, launch_speed, launch_angle,
             estimated_woba_using_speedangle,
             CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END AS hard_hit,
             CASE WHEN launch_speed IS NOT NULL AND launch_angle IS NOT NULL
               AND launch_speed >= 98 AND launch_angle BETWEEN 26 AND 30
               THEN 1 ELSE 0 END AS conservative_barrel_proxy
      FROM fact_statcast_pitch
      WHERE batter IS NOT NULL AND game_date IS NOT NULL
    )
    SELECT player_id, competition, season,
           COUNT(*) AS tracked_pitches,
           COUNT(DISTINCT game_pk) AS observed_games,
           COUNT(launch_speed) AS batted_balls_with_ev,
           AVG(launch_speed) AS avg_exit_velocity,
           SUM(hard_hit) FILTER (WHERE launch_speed IS NOT NULL)::DOUBLE
             / NULLIF(COUNT(launch_speed),0) AS hard_hit_rate,
           AVG(launch_angle) AS avg_launch_angle,
           AVG(estimated_woba_using_speedangle) AS avg_estimated_woba_on_contact,
           COUNT(estimated_woba_using_speedangle) AS xwoba_contact_samples,
           COUNT(launch_angle) AS launch_angle_samples
    FROM events GROUP BY player_id, competition, season
    ''')
    con.execute('''
    CREATE OR REPLACE TABLE feature_pitcher_season AS
    WITH pitches AS (
      SELECT pitcher AS player_id, competition,
             EXTRACT(YEAR FROM game_date)::INTEGER AS season,
             game_pk, pitch_type, release_speed, release_spin_rate,
             release_extension, description,
             CASE WHEN description IN (
               'swinging_strike', 'swinging_strike_blocked',
               'foul', 'foul_tip', 'foul_bunt', 'hit_into_play',
               'hit_into_play_no_out', 'hit_into_play_score',
               'missed_bunt', 'bunt_foul_tip'
             ) THEN 1 ELSE 0 END AS swing,
             CASE WHEN description IN ('swinging_strike',
               'swinging_strike_blocked', 'missed_bunt') THEN 1 ELSE 0 END AS whiff
      FROM fact_statcast_pitch
      WHERE pitcher IS NOT NULL AND game_date IS NOT NULL
    )
    SELECT player_id, competition, season,
           COUNT(*) AS tracked_pitches,
           COUNT(DISTINCT game_pk) AS observed_games,
           COUNT(release_speed) AS velocity_samples,
           AVG(release_speed) AS avg_release_speed,
           COUNT(release_spin_rate) AS spin_samples,
           AVG(release_spin_rate) AS avg_spin_rate,
           AVG(release_extension) AS avg_release_extension,
           SUM(swing) AS observed_swings,
           SUM(whiff) AS observed_whiffs,
           SUM(whiff)::DOUBLE / NULLIF(SUM(swing),0) AS whiff_rate_on_swings
    FROM pitches GROUP BY player_id, competition, season
    ''')
    return {name: con.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
            for name in ('feature_hitter_season','feature_pitcher_season')}


def main():
    if not DB_PATH.exists():
        raise SystemExit('Database not found: run ingestion and build_db first')
    con = duckdb.connect(str(DB_PATH))
    try:
        counts = build(con)
        for name, count in counts.items():
            print(f'{name}: {count} player-season rows')
    finally:
        con.close()


if __name__ == '__main__':
    main()

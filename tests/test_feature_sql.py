import pytest

def test_features_on_synthetic_pitch_data():
    duckdb = pytest.importorskip('duckdb')
    from src.build_features import build
    con = duckdb.connect(':memory:')
    con.execute('''
        CREATE TABLE fact_statcast_pitch AS
        SELECT 1::BIGINT AS batter, 2::BIGINT AS pitcher,
        'MLB'::VARCHAR AS competition, '2025-05-01'::DATE AS game_date,
        101::BIGINT AS game_pk, 'FF'::VARCHAR AS pitch_type,
        100.0::DOUBLE AS launch_speed, 20.0::DOUBLE AS launch_angle,
        0.45::DOUBLE AS estimated_woba_using_speedangle,
        95.0::DOUBLE AS release_speed, 2300.0::DOUBLE AS release_spin_rate,
        6.2::DOUBLE AS release_extension, 'swinging_strike'::VARCHAR AS description
        UNION ALL
        SELECT 1, 2, 'MLB', '2025-05-01'::DATE, 101, 'FF', NULL, NULL, NULL,
        96.0, 2350.0, 6.3, 'ball'
    ''')
    counts = build(con)
    assert counts == {'feature_hitter_season': 1, 'feature_pitcher_season': 1}
    hitter = con.execute('SELECT tracked_pitches, batted_balls_with_ev, hard_hit_rate FROM feature_hitter_season').fetchone()
    assert hitter == (2, 1, 1.0)
    pitcher = con.execute('SELECT tracked_pitches, observed_swings, observed_whiffs, whiff_rate_on_swings FROM feature_pitcher_season').fetchone()
    assert pitcher == (2, 1, 1, 1.0)
    con.close()

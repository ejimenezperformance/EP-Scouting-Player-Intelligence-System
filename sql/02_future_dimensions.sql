-- Phase 2 logical model: player identity, competition and season aggregates.
-- Populate only from a source that provides the relevant fields.

CREATE TABLE IF NOT EXISTS dim_player_identity (
  player_id BIGINT PRIMARY KEY,
  full_name VARCHAR,
  birth_date DATE,
  bats VARCHAR,
  throws VARCHAR,
  primary_position VARCHAR,
  source VARCHAR,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_competition (
  competition_id VARCHAR PRIMARY KEY,
  competition_name VARCHAR,
  competition_level VARCHAR,
  league_name VARCHAR,
  season INTEGER
);

CREATE TABLE IF NOT EXISTS fact_player_season_batting (
  player_id BIGINT,
  season INTEGER,
  competition_id VARCHAR,
  games INTEGER,
  plate_appearances INTEGER,
  at_bats INTEGER,
  hits INTEGER,
  home_runs INTEGER,
  walks INTEGER,
  strikeouts INTEGER,
  batting_avg DOUBLE,
  on_base_pct DOUBLE,
  slugging_pct DOUBLE,
  woba DOUBLE,
  source VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_player_season_pitching (
  player_id BIGINT,
  season INTEGER,
  competition_id VARCHAR,
  games INTEGER,
  games_started INTEGER,
  innings_pitched DOUBLE,
  batters_faced INTEGER,
  strikeouts INTEGER,
  walks INTEGER,
  home_runs_allowed INTEGER,
  era DOUBLE,
  fip DOUBLE,
  source VARCHAR
);

CREATE TABLE IF NOT EXISTS fact_player_physical_assessment (
  player_id BIGINT,
  assessment_date DATE,
  test_name VARCHAR,
  metric_name VARCHAR,
  metric_value DOUBLE,
  unit VARCHAR,
  device VARCHAR,
  protocol VARCHAR,
  source_file VARCHAR
);

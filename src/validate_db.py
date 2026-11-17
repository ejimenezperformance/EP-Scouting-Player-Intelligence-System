import duckdb
from src.config import DB_PATH

def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = set(con.execute("SHOW TABLES").df()["name"])
    required = {"ingestion_log", "raw_statcast", "fact_statcast_pitch", "dim_player"}
    missing = required - tables
    if missing:
        raise SystemExit(f"Missing tables: {sorted(missing)}")
    result = con.execute('''
      SELECT COUNT(*) AS pitch_rows,
             COUNT(*) FILTER (WHERE game_date IS NULL) AS null_dates,
             COUNT(*) FILTER (WHERE batter IS NULL) AS null_batters,
             COUNT(*) FILTER (WHERE pitcher IS NULL) AS null_pitchers
      FROM fact_statcast_pitch
    ''').df()
    print(result.to_string(index=False))
    con.close()

if __name__ == "__main__":
    main()

import json
import duckdb

from src.config import RAW, MANIFESTS, DB_PATH, WAREHOUSE

def main():
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    con.execute('''
    CREATE TABLE IF NOT EXISTS ingestion_log (
      ingestion_id VARCHAR PRIMARY KEY, source VARCHAR, competition VARCHAR,
      level_filter_verified BOOLEAN, start_date DATE, end_date DATE,
      retrieved_at_utc TIMESTAMP, row_count BIGINT, status VARCHAR,
      source_file VARCHAR, error VARCHAR
    )
    ''')

    for mf in MANIFESTS.glob("*.json"):
        m = json.loads(mf.read_text(encoding="utf-8"))
        con.execute("INSERT OR REPLACE INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            mf.stem, m.get("source"), m.get("competition"), m.get("level_filter_verified"),
            m.get("start_date"), m.get("end_date"), m.get("retrieved_at_utc"),
            m.get("rows", 0), m.get("status"), m.get("file"), m.get("error")
        ])

    paths = list(RAW.rglob("*.csv"))
    if not paths:
        print("No raw CSVs. Run the ingestion script first.")
        con.close()
        return

    # Provenance gate: every CSV needs a successful manifest with a matching row count.
    # Row counts and loading are done by DuckDB, so the full history never sits in pandas memory.
    for path in paths:
        competition = path.parent.name.upper()
        if competition not in {"MLB", "MILB"}:
            raise ValueError(f"Unexpected competition folder: {path}")
        manifest_path = MANIFESTS / (path.stem + ".json")
        if not manifest_path.exists():
            raise ValueError(f"Missing provenance manifest: {manifest_path}")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if meta.get("competition", "").upper() != competition:
            raise ValueError(f"Competition mismatch in {manifest_path}")
        if meta.get("status") != "ok":
            raise ValueError(f"Non-successful ingestion manifest: {manifest_path}")
        n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path.as_posix()}', all_varchar=true)").fetchone()[0]
        if int(meta.get("rows", -1)) != n:
            raise ValueError(f"Row count mismatch: {path}")

    pattern = (RAW / "*" / "*.csv").as_posix()
    con.execute(f"""
    CREATE OR REPLACE TABLE raw_statcast AS
    SELECT *, UPPER(regexp_extract(filename, '([^/\\\\]+)[/\\\\][^/\\\\]+$', 1)) AS competition,
           regexp_extract(filename, '([^/\\\\]+[/\\\\][^/\\\\]+)$', 1) AS source_file,
           TRUE AS level_filter_verified
    FROM read_csv_auto('{pattern}', union_by_name=true, all_varchar=true, filename=true)
    """)
    data_cols = [r[0] for r in con.execute("DESCRIBE raw_statcast").fetchall()]

    # Keep all source fields in raw_statcast; create a typed event table for analysis.
    available = {c.lower(): c for c in data_cols}
    fields = {
      "game_pk":"BIGINT", "game_date":"DATE", "game_year":"INTEGER",
      "game_type":"VARCHAR", "batter":"BIGINT", "pitcher":"BIGINT",
      "stand":"VARCHAR", "p_throws":"VARCHAR", "inning":"INTEGER",
      "inning_topbot":"VARCHAR", "outs_when_up":"INTEGER", "balls":"INTEGER",
      "strikes":"INTEGER", "pitch_number":"INTEGER", "events":"VARCHAR",
      "description":"VARCHAR", "type":"VARCHAR", "pitch_type":"VARCHAR",
      "release_speed":"DOUBLE", "release_spin_rate":"DOUBLE", "release_extension":"DOUBLE",
      "release_pos_x":"DOUBLE", "release_pos_y":"DOUBLE", "release_pos_z":"DOUBLE",
      "pfx_x":"DOUBLE", "pfx_z":"DOUBLE", "plate_x":"DOUBLE", "plate_z":"DOUBLE",
      "zone":"INTEGER", "sz_top":"DOUBLE", "sz_bot":"DOUBLE",
      "launch_speed":"DOUBLE", "launch_angle":"DOUBLE", "hit_distance_sc":"DOUBLE",
      "estimated_ba_using_speedangle":"DOUBLE",
      "estimated_woba_using_speedangle":"DOUBLE", "woba_value":"DOUBLE",
      "woba_denom":"DOUBLE", "babip_value":"DOUBLE", "iso_value":"DOUBLE",
      "at_bat_number":"INTEGER", "spin_axis":"DOUBLE",
      "delta_home_win_exp":"DOUBLE", "delta_run_exp":"DOUBLE"
    }
    expr = []
    for field, dtype in fields.items():
        source = available.get(field)
        if source:
            escaped = '"' + source.replace('"', '""') + '"'
            expr.append(f'TRY_CAST({escaped} AS {dtype}) AS "{field}"')
        else:
            expr.append(f'CAST(NULL AS {dtype}) AS "{field}"')
    expr += ['source_file', 'competition', 'level_filter_verified']
    # Regular season only (game_type = 'R'); spring training and postseason stay in raw_statcast.
    con.execute("CREATE OR REPLACE TABLE fact_statcast_pitch AS SELECT " + ", ".join(expr)
                + " FROM raw_statcast WHERE game_type = 'R'")

    con.execute('''
    CREATE OR REPLACE TABLE dim_player AS
    SELECT player_id,
           BOOL_OR(is_batter) AS observed_as_batter,
           BOOL_OR(is_pitcher) AS observed_as_pitcher
    FROM (
      SELECT batter AS player_id, TRUE AS is_batter, FALSE AS is_pitcher
      FROM fact_statcast_pitch WHERE batter IS NOT NULL
      UNION ALL
      SELECT pitcher AS player_id, FALSE, TRUE
      FROM fact_statcast_pitch WHERE pitcher IS NOT NULL
    )
    GROUP BY player_id
    ''')

    print("Database:", DB_PATH)
    print(con.execute("SELECT COUNT(*) pitches, COUNT(DISTINCT batter) batters, COUNT(DISTINCT pitcher) pitchers FROM fact_statcast_pitch").df())
    con.close()

if __name__ == "__main__":
    main()

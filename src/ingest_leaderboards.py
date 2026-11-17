"""Phase 2 ingestion: season-level hitter leaderboards from Baseball Savant.

One CSV per season (regular season, all hitters with at least 1 PA) plus a
JSON manifest with the same provenance fields used in Phase 1.

Usage:
    python -m src.ingest_leaderboards                 # 2015-2026
    python -m src.ingest_leaderboards --years 2025 2026 --overwrite
"""
import argparse
import io
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import RAW, MANIFESTS

URL = "https://baseballsavant.mlb.com/leaderboard/custom"
FIELDS = [
    "pa", "player_age", "woba", "xwoba", "xslg", "xba", "barrel_batted_rate",
    "hard_hit_percent", "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent",
    "k_percent", "bb_percent", "whiff_percent", "oz_swing_percent", "sprint_speed",
]
FOLDER = RAW.parent / "leaderboards" / "hitters"


def fetch(year: int, overwrite: bool = False, pause: float = 1.5) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    stem = f"leaderboard_hitters_{year}"
    csv_path, manifest_path = FOLDER / f"{stem}.csv", MANIFESTS / f"{stem}.json"
    if csv_path.exists() and csv_path.stat().st_size and not overwrite:
        print(f"SKIP {csv_path.name}")
        return
    params = {"year": year, "type": "batter", "filter": "", "min": 1,
              "selections": ",".join(FIELDS), "chart": "false", "x": "pa", "y": "pa",
              "r": "no", "chartType": "beeswarm", "sort": "pa", "sortDir": "desc", "csv": "true"}
    meta = {"source": "Baseball Savant custom leaderboard (regular season)", "competition": "MLB",
            "season": year, "fields": FIELDS,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": 0, "status": "started", "file": csv_path.name, "error": None}
    try:
        r = requests.get(URL, params=params, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
        df = df.rename(columns={"last_name, first_name": "player_name"})
        missing = [f for f in FIELDS if f not in df.columns]
        if missing:
            raise ValueError(f"missing fields from source: {missing}")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        meta["rows"], meta["status"] = len(df), "ok" if len(df) else "empty"
    except Exception as exc:  # recorded in the manifest, never silently dropped
        meta["status"], meta["error"] = "error", repr(exc)
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{year}: {meta['status']} rows={meta['rows']}")
    time.sleep(pause)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2015, 2027)))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for y in a.years:
        fetch(y, a.overwrite)


if __name__ == "__main__":
    main()

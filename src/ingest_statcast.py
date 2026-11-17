import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from pybaseball import statcast

from src.config import RAW, MANIFESTS

def windows(start, end, days=7):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)

def fetch_mlb(start, end, overwrite=False, pause=1.5):
    folder = RAW / "mlb"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"statcast_mlb_{start:%Y%m%d}_{end:%Y%m%d}"
    csv_path, manifest_path = folder / f"{stem}.csv", MANIFESTS / f"{stem}.json"
    if csv_path.exists() and csv_path.stat().st_size and not overwrite:
        print(f"SKIP {csv_path}")
        return
    meta = {
        "source": "Baseball Savant MLB Statcast via pybaseball",
        "competition": "MLB", "level_filter_verified": True,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 0, "status": "started", "file": str(csv_path.relative_to(RAW)),
        "error": None
    }
    try:
        frame = statcast(start_dt=start.isoformat(), end_dt=end.isoformat(),
                         verbose=False, parallel=True)
        if frame is None:
            frame = pd.DataFrame()
        meta["rows"] = len(frame)
        meta["status"] = "ok" if len(frame) else "empty"
        if len(frame):
            frame.to_csv(csv_path, index=False)
    except Exception as exc:
        meta["status"], meta["error"] = "error", repr(exc)
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"MLB {meta['status']}: {start}..{end}, rows={meta['rows']}")
    time.sleep(pause)

def fetch_milb(start, end, overwrite=False, pause=1.5):
    # Minor League Statcast coverage begins in 2021 and is limited to
    # specified levels/parks. The source helper pins minors=true and wbc=false.
    if end < date(2021, 1, 1):
        print(f"MiLB Statcast unavailable before 2021 in this source: {start}..{end}")
        return
    from sportsdataverse.mlb import mlb_statcast_search_minors
    start = max(start, date(2021, 1, 1))
    folder = RAW / "milb"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"statcast_milb_{start:%Y%m%d}_{end:%Y%m%d}"
    csv_path, manifest_path = folder / f"{stem}.csv", MANIFESTS / f"{stem}.json"
    if csv_path.exists() and csv_path.stat().st_size and not overwrite:
        print(f"SKIP {csv_path}")
        return
    meta = {
        "source": "Baseball Savant Minor League Statcast via sportsdataverse-py",
        "competition": "MiLB", "level_filter_verified": True,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 0, "status": "started", "file": str(csv_path.relative_to(RAW)),
        "coverage_note": "Only tracked levels/parks; not complete MiLB coverage.",
        "error": None
    }
    try:
        frame = mlb_statcast_search_minors(
            start.isoformat(), end.isoformat(), chunk_days=7, return_as_pandas=True
        )
        if frame is None:
            frame = pd.DataFrame()
        meta["rows"] = len(frame)
        meta["status"] = "ok" if len(frame) else "empty"
        if len(frame):
            frame.to_csv(csv_path, index=False)
    except Exception as exc:
        meta["status"], meta["error"] = "error", repr(exc)
    manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"MiLB {meta['status']}: {start}..{end}, rows={meta['rows']}")
    time.sleep(pause)

def main():
    parser = argparse.ArgumentParser(description="Download MLB and/or tracked MiLB Statcast.")
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument("--competitions", default="MLB,MiLB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--pause", type=float, default=1.5)
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("end-date must be on or after start-date")
    if args.end_date > date.today():
        parser.error("end-date cannot be in the future")
    competitions = {x.strip().upper() for x in args.competitions.split(",")}
    if not competitions.issubset({"MLB", "MILB"}):
        parser.error("competitions must contain MLB and/or MiLB")
    if "MILB" in competitions:
        try:
            import sportsdataverse  # noqa: F401
        except ImportError:
            parser.error("MiLB adapter requires: pip install --pre sportsdataverse")
    for start, end in windows(args.start_date, args.end_date, args.chunk_days):
        if "MLB" in competitions:
            fetch_mlb(start, end, args.overwrite, args.pause)
        if "MILB" in competitions and end >= date(2021, 1, 1):
            fetch_milb(start, end, args.overwrite, args.pause)

if __name__ == "__main__":
    main()

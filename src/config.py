from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "statcast"
MANIFESTS = DATA / "manifests"
WAREHOUSE = DATA / "warehouse"
DB_PATH = WAREHOUSE / "ep_scouting.duckdb"

for folder in (RAW, MANIFESTS, WAREHOUSE):
    folder.mkdir(parents=True, exist_ok=True)

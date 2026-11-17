"""EP — International pipeline data pull (MLB Stats API, public).

Run on your computer (Windows/Mac), in any folder:
    pip install requests
    python ep_dsl_pull.py

Takes ~10-20 minutes. When it finishes it creates  ep_dsl_raw.zip  in the same
folder. Upload that zip to Claude. Nothing else is needed.

What it downloads (raw JSON, unmodified):
  1) Season stats (hitting + pitching) for MLB, AAA, AA, High-A, Single-A and
     Rookie levels (Rookie includes the Dominican Summer League), 2012-2026.
  2) Team lists per level and season (to know each team's parent club).
  3) Biographical data (birth date, country, position, MLB debut) for every
     player who appeared in the Dominican Summer League.
"""
import json
import time
import zipfile
from pathlib import Path

import requests

BASE = "https://statsapi.mlb.com/api/v1"
SPORTS = {1: "MLB", 11: "AAA", 12: "AA", 13: "HighA", 14: "SingleA", 16: "Rookie"}
SEASONS = [y for y in range(2012, 2027) if y != 2020]  # no minor-league season in 2020
OUT = Path("ep_dsl_raw")
OUT.mkdir(exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "EP-research (public data, low rate)"


def get(url, params, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  status {r.status_code}, retrying...")
        except Exception as exc:
            print(f"  error {exc!r}, retrying...")
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url} {params}")


def save(name, data):
    (OUT / name).write_text(json.dumps(data), encoding="utf-8")


def pull_stats():
    for season in SEASONS:
        for sid, sname in SPORTS.items():
            for group in ("hitting", "pitching"):
                name = f"stats_{season}_{sname}_{group}.json"
                if (OUT / name).exists():
                    continue
                pages, offset = [], 0
                while True:
                    d = get(f"{BASE}/stats", {"stats": "season", "group": group, "sportIds": sid,
                                              "season": season, "playerPool": "ALL", "gameType": "R",
                                              "limit": 2000, "offset": offset})
                    pages.append(d)
                    splits = sum(len(s.get("splits", [])) for s in d.get("stats", []))
                    total = max([s.get("totalSplits", 0) for s in d.get("stats", [])] + [0])
                    offset += 2000
                    if splits == 0 or offset >= total:
                        break
                    time.sleep(0.5)
                save(name, pages)
                n = sum(len(s.get("splits", [])) for p in pages for s in p.get("stats", []))
                print(f"{season} {sname:7s} {group:8s} rows={n}")
                time.sleep(0.5)
            tname = f"teams_{season}_{sname}.json"
            if not (OUT / tname).exists():
                save(tname, get(f"{BASE}/teams", {"sportId": sid, "season": season}))


def dsl_player_ids():
    ids = set()
    for f in OUT.glob("stats_*_Rookie_*.json"):
        for page in json.loads(f.read_text(encoding="utf-8")):
            for s in page.get("stats", []):
                for sp in s.get("splits", []):
                    league = (sp.get("league") or {}).get("name", "") or ""
                    team = (sp.get("team") or {}).get("name", "") or ""
                    if "Dominican" in league or team.startswith("DSL"):
                        pid = (sp.get("player") or {}).get("id")
                        if pid:
                            ids.add(pid)
    return sorted(ids)


def pull_people(ids):
    batch, out = 150, []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        d = get(f"{BASE}/people", {"personIds": ",".join(map(str, chunk))})
        out.extend(d.get("people", []))
        print(f"people {min(i + batch, len(ids))}/{len(ids)}")
        time.sleep(0.5)
    save("people_dsl.json", out)


if __name__ == "__main__":
    t0 = time.time()
    pull_stats()
    ids = dsl_player_ids()
    print(f"DSL players found: {len(ids)}")
    pull_people(ids)
    with zipfile.ZipFile("ep_dsl_raw.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT.glob("*.json"):
            z.write(f, f.name)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> ep_dsl_raw.zip (upload this file)")

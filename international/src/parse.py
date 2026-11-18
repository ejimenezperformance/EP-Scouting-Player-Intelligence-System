"""Parse raw MLB Stats API JSON into tidy parquet tables."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAW, WORK = Path(sys.argv[1]), Path(sys.argv[2])
WORK.mkdir(parents=True, exist_ok=True)

HIT = ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
       "intentionalWalks", "hitByPitch", "strikeOuts", "sacFlies", "stolenBases", "caughtStealing",
       "gamesPlayed"]
PIT = ["battersFaced", "outs", "strikeOuts", "baseOnBalls", "hitByPitch", "homeRuns", "hits",
       "earnedRuns", "gamesPlayed", "gamesStarted", "wildPitches"]

rows = {"hitting": [], "pitching": []}
for f in sorted(RAW.glob("stats_*.json")):
    m = re.match(r"stats_(\d{4})_(\w+?)_(hitting|pitching)\.json", f.name)
    season, level, group = int(m.group(1)), m.group(2), m.group(3)
    for page in json.loads(f.read_text(encoding="utf-8")):
        for s in page.get("stats", []):
            for sp in s.get("splits", []):
                st = sp.get("stat", {})
                r = {"season": season, "level": level,
                     "player_id": (sp.get("player") or {}).get("id"),
                     "player_name": (sp.get("player") or {}).get("fullName"),
                     "team_id": (sp.get("team") or {}).get("id"),
                     "team_name": (sp.get("team") or {}).get("name"),
                     "league_id": (sp.get("league") or {}).get("id"),
                     "league_name": (sp.get("league") or {}).get("name"),
                     "age_api": st.get("age")}
                for k in (HIT if group == "hitting" else PIT):
                    v = st.get(k)
                    r[k] = float(v) if isinstance(v, (int, float)) else (float(v) if v not in (None, "", "-.--", ".---") and str(v).replace(".", "", 1).lstrip("-").isdigit() else 0.0)
                rows[group].append(r)

for g, rs in rows.items():
    df = pd.DataFrame(rs)
    df.to_parquet(WORK / f"{g}.parquet", index=False)
    print(g, df.shape, df.groupby("level").size().to_dict())

teams = []
for f in sorted(RAW.glob("teams_*.json")):
    for t in json.loads(f.read_text(encoding="utf-8")).get("teams", []):
        teams.append({"season": t.get("season"), "team_id": t.get("id"), "team_name": t.get("name"),
                      "parent_org_id": t.get("parentOrgId"), "parent_org": t.get("parentOrgName"),
                      "league_name": (t.get("league") or {}).get("name"),
                      "sport_id": (t.get("sport") or {}).get("id")})
teams = pd.DataFrame(teams).drop_duplicates(["season", "team_id"])
teams.to_parquet(WORK / "teams.parquet", index=False)
print("teams", teams.shape)

people = json.loads((RAW / "people_dsl.json").read_text(encoding="utf-8"))
pp = pd.DataFrame([{
    "player_id": p.get("id"), "full_name": p.get("fullName"), "birth_date": p.get("birthDate"),
    "birth_country": p.get("birthCountry"), "height": p.get("height"), "weight": p.get("weight"),
    "position": (p.get("primaryPosition") or {}).get("abbreviation"),
    "bats": (p.get("batSide") or {}).get("code"), "throws": (p.get("pitchHand") or {}).get("code"),
    "mlb_debut": p.get("mlbDebutDate"), "active": p.get("active")} for p in people])
pp.to_parquet(WORK / "people.parquet", index=False)
print("people", pp.shape, "with MLB debut:", pp["mlb_debut"].notna().sum())
print(pp["birth_country"].value_counts().head(8).to_dict())

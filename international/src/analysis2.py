"""Follow-ups (v1.1): out-of-sample expectations, markets by country, organizations with a null simulation.

Expected values are cross-fitted: each signing class (first DSL season) is scored by a model
trained on the OTHER classes 2012-2018, so no player's own outcome is used to set his expectation.
Organization and country gaps are then compared against 5,000 simulations of the model's own
expectations (what spread would appear by chance alone).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(sys.argv[1])
RNG = np.random.default_rng(2026)
PERF = {"hitters": ["z_bb_pct", "z_k_pct", "z_obp", "z_slg", "z_iso", "z_sb_rate", "plateAppearances"],
        "pitchers": ["z_k_pct", "z_bb_pct", "z_kbb_pct", "z_hr_pct", "battersFaced"]}
COUNTRY = {"DOM": "Dominican Republic", "VEN": "Venezuela", "PAN": "Panama", "COL": "Colombia",
           "MEX": "Mexico", "CUB": "Cuba", "NIC": "Nicaragua", "CUW": "Curaçao", "Curacao": "Curaçao"}
N_SIM = 5000


def crossfit(d, perf, outcome="reach_aa"):
    d = d[d["season"].between(2012, 2018)].copy()
    cols = ["age"] + perf
    d["expected"] = np.nan
    for yr in range(2012, 2019):
        tr, te = d[d["season"] != yr], d["season"] == yr
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        m.fit(tr[cols].fillna(0), tr[outcome])
        d.loc[te, "expected"] = m.predict_proba(d.loc[te, cols].fillna(0))[:, 1]
    return d


def group_gap(d, key, outcome="reach_aa"):
    g = d.groupby(key).agg(n=(outcome, "size"), reached=(outcome, "sum"), expected=("expected", "sum"))
    g["rate"] = g["reached"] / g["n"]
    g["gap_per_100"] = 100 * (g["reached"] - g["expected"]) / g["n"]
    # null simulation: outcomes drawn from the model's own expectations
    p = d["expected"].values
    keys = d[key].values
    groups = {k: np.where(keys == k)[0] for k in g.index}
    sims = RNG.random((N_SIM, len(p))) < p
    sim_gap = pd.DataFrame({k: 100 * (sims[:, idx].sum(1) - p[idx].sum()) / len(idx) for k, idx in groups.items()})
    g["p_low"] = [(sim_gap[k] <= g.loc[k, "gap_per_100"]).mean() for k in g.index]
    g["p_high"] = [(sim_gap[k] >= g.loc[k, "gap_per_100"]).mean() for k in g.index]
    g["sim_lo"] = [np.percentile(sim_gap[k], 2.5) for k in g.index]
    g["sim_hi"] = [np.percentile(sim_gap[k], 97.5) for k in g.index]
    return g.sort_values("gap_per_100"), sim_gap


def main():
    res = {}
    allrows = []
    for kind, perf in PERF.items():
        d = pd.read_parquet(OUT / f"{kind}_first_dsl.parquet")
        d["country"] = d["birth_country"].replace(COUNTRY).fillna("Unknown")
        cf = crossfit(d, perf)
        cf["kind"] = kind
        allrows.append(cf[["player_id", "kind", "season", "org", "country", "age", "reach_aa", "reach_mlb", "expected"]])
        # organizations
        org = cf.dropna(subset=["org"])
        g, sim = group_gap(org, "org")
        g["rank_low_to_high"] = range(1, len(g) + 1)
        # chance check: in a world where every club converts exactly as expected,
        # how often is the LOWEST club's gap at least this low?
        b = g.loc["Milwaukee Brewers"]
        min_sim = sim.min(axis=1)
        res[f"{kind}_brewers"] = {**{k: float(v) for k, v in b.items()},
                                  "p_any_club_this_low_by_chance": float((min_sim <= b["gap_per_100"]).mean()),
                                  "rank_of_30": int(g.index.get_loc("Milwaukee Brewers") + 1)}
        g.to_csv(OUT / f"{kind}_org_crossfit.csv")
    allp = pd.concat(allrows)
    allp.to_parquet(OUT / "crossfit_all.parquet", index=False)
    # markets: hitters + pitchers together
    top = allp["country"].value_counts()
    keep = top[top >= 60].index
    allp["market"] = np.where(allp["country"].isin(keep), allp["country"], "Other countries")
    mk, _ = group_gap(allp, "market")
    mk["age_mean"] = allp.groupby("market")["age"].mean()
    mk["mlb_rate"] = allp.groupby("market")["reach_mlb"].mean()
    mk.to_csv(OUT / "markets_crossfit.csv")
    # Brewers mix by market: classes 2012-2018 and recent classes 2021-2026
    mix = []
    for kind in PERF:
        d = pd.read_parquet(OUT / f"{kind}_first_dsl.parquet")
        d["country"] = d["birth_country"].replace(COUNTRY).fillna("Unknown")
        mix.append(d[["player_id", "season", "org", "country"]])
    mix = pd.concat(mix)
    mix["market"] = np.where(mix["country"].isin(keep), mix["country"], "Other countries")
    mix["era"] = np.select([mix["season"].between(2012, 2018), mix["season"].between(2021, 2026)],
                           ["2012–2018", "2021–2026"], "other")
    mix = mix[mix["era"] != "other"]
    sh = (mix.assign(club=np.where(mix["org"] == "Milwaukee Brewers", "Brewers", "Rest of MLB"))
          .groupby(["era", "club"])["market"].value_counts(normalize=True).rename("share").reset_index())
    sh.to_csv(OUT / "market_mix.csv", index=False)
    (OUT / "summary_v11.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(res, indent=1))
    print(mk.round(3).to_string())
    print(sh[sh.era == "2021–2026"].pivot(index="market", columns="club", values="share").round(3))


if __name__ == "__main__":
    main()

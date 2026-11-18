"""International pipeline: what in a first DSL season predicts reaching AA / MLB?

Pre-registered (written before fitting):
  Unit: player, first Dominican Summer League season (min 100 PA hitters / 100 BF pitchers).
  Outcomes: reached AA by 2026; reached MLB by 2026.
  Cohorts: first DSL season 2012-2018 (8+ years of follow-up).
  Validation: train cohorts 2012-2015, test cohorts 2016-2018 (used once).
  Question: does DSL performance add to AGE alone?
  CONFIRMED if test AUC(age + performance) - AUC(age only) >= 0.02 and its 95% bootstrap CI > 0.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
H = pd.read_parquet(WORK / "hitting.parquet")
P = pd.read_parquet(WORK / "pitching.parquet")
PEOPLE = pd.read_parquet(WORK / "people.parquet")
TEAMS = pd.read_parquet(WORK / "teams.parquet")

def is_dsl(df):
    return (df["level"] == "Rookie") & (df["league_name"].fillna("").str.upper().eq("DSL") |
                                         df["team_name"].fillna("").str.startswith("DSL"))

# The Mexican League (LMB) is filed under AAA in the API; it is not an MLB affiliate level.
AFF_H, AFF_P = H[H["league_name"] != "MEX"], P[P["league_name"] != "MEX"]
reached = {lvl: set(AFF_H.loc[AFF_H.level == lvl, "player_id"]) | set(AFF_P.loc[AFF_P.level == lvl, "player_id"])
           for lvl in ["SingleA", "HighA", "AA", "AAA", "MLB"]}
reach_aa = reached["AA"] | reached["AAA"] | reached["MLB"]
reach_mlb = reached["MLB"]
reach_fullseason = reached["SingleA"] | reached["HighA"] | reach_aa

ORG = TEAMS.set_index(["season", "team_id"])["parent_org"].to_dict()
SUM_H = ["plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns", "baseOnBalls",
         "hitByPitch", "strikeOuts", "sacFlies", "stolenBases", "caughtStealing"]
SUM_P = ["battersFaced", "outs", "strikeOuts", "baseOnBalls", "hitByPitch", "homeRuns", "hits"]


def first_season(df, cols, vol, min_vol):
    d = df[is_dsl(df)].copy()
    d["org"] = [ORG.get((s, t)) for s, t in zip(d["season"], d["team_id"])]
    first = d.groupby("player_id")["season"].min().rename("first_dsl")
    d = d.merge(first, on="player_id")
    f = d[d["season"] == d["first_dsl"]]
    agg = f.groupby(["player_id", "season"]).agg(
        **{c: (c, "sum") for c in cols}, org=("org", lambda s: s.mode().iloc[0] if s.notna().any() else None),
        player_name=("player_name", "first")).reset_index()
    agg = agg[agg[vol] >= min_vol]
    agg = agg.merge(PEOPLE, on="player_id", how="left")
    bd = pd.to_datetime(agg["birth_date"], errors="coerce")
    ref = pd.to_datetime(agg["season"].astype(str) + "-07-01")
    agg["age"] = (ref - bd).dt.days / 365.25
    agg["reach_aa"] = agg["player_id"].isin(reach_aa).astype(int)
    agg["reach_mlb"] = agg["player_id"].isin(reach_mlb).astype(int)
    agg["reach_fullseason"] = agg["player_id"].isin(reach_fullseason).astype(int)
    agg["dsl_seasons"] = agg["player_id"].map(d.groupby("player_id")["season"].nunique())
    return agg.dropna(subset=["age"])


def zscore_by_season(df, cols):
    for c in cols:
        g = df.groupby("season")[c]
        df[f"z_{c}"] = (df[c] - g.transform("mean")) / g.transform("std")
    return df


def hitters():
    d = first_season(H, SUM_H, "plateAppearances", 100)
    pa = d["plateAppearances"]
    d["bb_pct"] = d["baseOnBalls"] / pa
    d["k_pct"] = d["strikeOuts"] / pa
    singles = d["hits"] - d["doubles"] - d["triples"] - d["homeRuns"]
    d["obp"] = (d["hits"] + d["baseOnBalls"] + d["hitByPitch"]) / (d["atBats"] + d["baseOnBalls"] + d["hitByPitch"] + d["sacFlies"]).clip(lower=1)
    tb = singles + 2 * d["doubles"] + 3 * d["triples"] + 4 * d["homeRuns"]
    d["slg"] = tb / d["atBats"].clip(lower=1)
    d["iso"] = d["slg"] - d["hits"] / d["atBats"].clip(lower=1)
    d["sb_rate"] = (d["stolenBases"] + d["caughtStealing"]) / pa
    d = zscore_by_season(d, ["bb_pct", "k_pct", "obp", "slg", "iso", "sb_rate"])
    d["pos_group"] = d["position"].map({"C": "C", "SS": "MIF", "2B": "MIF", "3B": "CIF", "1B": "CIF",
                                        "CF": "OF", "LF": "OF", "RF": "OF", "OF": "OF"}).fillna("OTHER")
    perf = ["z_bb_pct", "z_k_pct", "z_obp", "z_slg", "z_iso", "z_sb_rate", "plateAppearances"]
    return d, perf


def pitchers():
    d = first_season(P, SUM_P, "battersFaced", 100)
    bf = d["battersFaced"]
    d["k_pct"] = d["strikeOuts"] / bf
    d["bb_pct"] = d["baseOnBalls"] / bf
    d["kbb_pct"] = d["k_pct"] - d["bb_pct"]
    d["hr_pct"] = d["homeRuns"] / bf
    d["ip"] = d["outs"] / 3
    d = zscore_by_season(d, ["k_pct", "bb_pct", "kbb_pct", "hr_pct"])
    perf = ["z_k_pct", "z_bb_pct", "z_kbb_pct", "z_hr_pct", "battersFaced"]
    return d, perf


def fit_eval(d, perf, outcome):
    coh = d[d["season"].between(2012, 2018)].copy()
    tr, te = coh[coh["season"] <= 2015], coh[coh["season"] >= 2016]
    def model(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        m.fit(tr[cols].fillna(0), tr[outcome])
        return m, m.predict_proba(te[cols].fillna(0))[:, 1]
    m_age, p_age = model(["age"])
    m_full, p_full = model(["age"] + perf)
    y = te[outcome].values
    a0, a1 = roc_auc_score(y, p_age), roc_auc_score(y, p_full)
    bs = []
    n = len(te)
    for _ in range(2000):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        bs.append(roc_auc_score(y[i], p_full[i]) - roc_auc_score(y[i], p_age[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res = {"outcome": outcome, "n_train": len(tr), "n_test": len(te), "rate_test": float(y.mean()),
           "auc_age_only": a0, "auc_age_perf": a1, "auc_gain": a1 - a0, "ci_lo": lo, "ci_hi": hi,
           "label": "CONFIRMED" if (a1 - a0 >= 0.02 and lo > 0) else "NOT CONFIRMED"}
    coefs = dict(zip(["age"] + perf, m_full.named_steps["logisticregression"].coef_[0]))
    return res, coefs, m_full


def age_table(d, outcome):
    coh = d[d["season"].between(2012, 2018)].copy()
    coh["age_band"] = pd.cut(coh["age"], [0, 17.5, 18.5, 19.5, 20.5, 40],
                             labels=["≤17.5", "17.5–18.5", "18.5–19.5", "19.5–20.5", ">20.5"])
    return coh.groupby("age_band", observed=True).agg(n=(outcome, "size"), rate=(outcome, "mean")).reset_index()


def perf_by_age(d, outcome, perfcol):
    coh = d[d["season"].between(2012, 2018)].copy()
    coh["young"] = np.where(coh["age"] < 18.5, "Under 18.5", "18.5 and older")
    coh["perf_tier"] = coh.groupby("young")[perfcol].transform(lambda s: pd.qcut(s, 3, labels=["Bottom third", "Middle third", "Top third"]))
    return coh.groupby(["young", "perf_tier"], observed=True).agg(n=(outcome, "size"), rate=(outcome, "mean")).reset_index()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def org_table(d, m, perf, outcome):
    coh = d[d["season"].between(2012, 2018)].dropna(subset=["org"]).copy()
    coh["expected"] = m.predict_proba(coh[["age"] + perf].fillna(0))[:, 1]
    g = coh.groupby("org").agg(n=(outcome, "size"), reached=(outcome, "sum"), expected=("expected", "sum")).reset_index()
    g["rate"] = g["reached"] / g["n"]
    g[["ci_lo", "ci_hi"]] = [wilson(k, n) for k, n in zip(g["reached"], g["n"])]
    g["over_expected"] = g["reached"] - g["expected"]
    g["over_expected_per_100"] = 100 * g["over_expected"] / g["n"]
    # Poisson-binomial approx z for observed vs expected
    var = coh.assign(v=coh["expected"] * (1 - coh["expected"])).groupby("org")["v"].sum()
    g["z"] = g["over_expected"] / g["org"].map(np.sqrt(var))
    return g.sort_values("over_expected_per_100", ascending=False)


def main():
    out = {}
    for kind, (d, perf) in {"hitters": hitters(), "pitchers": pitchers()}.items():
        d.to_parquet(OUT / f"{kind}_first_dsl.parquet", index=False)
        res = {}
        for oc in ["reach_aa", "reach_mlb"]:
            r, coefs, m = fit_eval(d, perf, oc)
            res[oc] = {**r, "coefs": coefs}
            if oc == "reach_aa":
                org_table(d, m, perf, oc).to_csv(OUT / f"{kind}_org_reach_aa.csv", index=False)
                cur = d[d["season"] >= 2025].copy()
                cur["p_reach_aa"] = m.predict_proba(cur[["age"] + perf].fillna(0))[:, 1]
                cur.sort_values("p_reach_aa", ascending=False).to_csv(OUT / f"{kind}_current_scored.csv", index=False)
        age_table(d, "reach_aa").to_csv(OUT / f"{kind}_age_reach_aa.csv", index=False)
        perfcol = "z_obp" if kind == "hitters" else "z_kbb_pct"
        perf_by_age(d, "reach_aa", perfcol).to_csv(OUT / f"{kind}_perf_by_age.csv", index=False)
        coh = d[d["season"].between(2012, 2018)]
        out[kind] = {"n_cohort": int(len(coh)), "rate_fullseason": float(coh["reach_fullseason"].mean()),
                     "rate_aa": float(coh["reach_aa"].mean()), "rate_mlb": float(coh["reach_mlb"].mean()),
                     "models": res}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in out.items()}, indent=1))
    for k, v in out.items():
        for oc, r in v["models"].items():
            print(k, oc, {x: round(r[x], 3) if isinstance(r[x], float) else r[x] for x in
                          ["n_train", "n_test", "rate_test", "auc_age_only", "auc_age_perf", "auc_gain", "ci_lo", "ci_hi", "label"]})


if __name__ == "__main__":
    main()

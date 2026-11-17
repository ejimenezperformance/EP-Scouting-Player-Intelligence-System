"""Out-of-time check of the Phase 2.1 hitter signal rule (EXPLORATORY).

Rule (docs/PHASE2_1_SIGNALS.md): POSITIVE_CHANGE if average EV +2 mph AND
Hard-Hit% +5 pts vs the prior season; NEGATIVE_CHANGE if both fall by the same.
Question: after a flag in season t, did the hitter beat (or trail) the
bias-corrected Marcel projection in season t+1?

Uses season leaderboards (MLB) as a stand-in for the pitch-level feature table.
Output: results/signal_evaluation.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA
import src.model_hitters as mh

RES = DATA.parent / "results"


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s


def main():
    d, _ = mh.prepare(pd.read_parquet(DATA / "processed" / "model_hitters.parquet"))
    lb = d[["player_id", "year", "hard_hit_percent"]].copy()
    lb["year"] += 1
    d = d.merge(lb.rename(columns={"hard_hit_percent": "hh_prev"}), on=["player_id", "year"], how="left")
    d["d_hh"] = d["hard_hit_percent"] - d["hh_prev"]
    d = d[d["lost_role"] == 0].dropna(subset=["d_exit_velocity_avg", "d_hh", "resid_bc"])
    d["signal"] = np.select(
        [(d["d_exit_velocity_avg"] >= 2) & (d["d_hh"] >= 5),
         (d["d_exit_velocity_avg"] <= -2) & (d["d_hh"] <= -5)],
        ["POSITIVE_CHANGE", "NEGATIVE_CHANGE"], "NO_MATERIAL_CHANGE")
    d["raw_change_next"] = d["woba_next"] - d["woba"]
    base = d[d["signal"] == "NO_MATERIAL_CHANGE"]
    rows = []
    for sig in ["POSITIVE_CHANGE", "NEGATIVE_CHANGE", "NO_MATERIAL_CHANGE"]:
        g = d[d["signal"] == sig]
        r = {"signal": sig, "n": len(g),
             "mean_next_woba_minus_marcel": g["resid_bc"].mean(),
             "mean_next_woba_minus_current": g["raw_change_next"].mean(),
             "pct_beat_marcel_by_020": 100 * (g["resid_bc"] >= 0.020).mean()}
        if sig != "NO_MATERIAL_CHANGE" and len(g) > 2:
            r["mw_p_vs_no_change"] = stats.mannwhitneyu(g["resid_bc"], base["resid_bc"]).pvalue
            r["cohens_d_vs_no_change"] = cohens_d(g["resid_bc"], base["resid_bc"])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "signal_evaluation.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()

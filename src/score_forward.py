"""Score the frozen 2027 predictions once the 2027 regular season is complete.

1) python -m src.ingest_leaderboards --years 2027
2) python -m src.score_forward
Checks the SHA-256 hashes first: if any frozen file changed, scoring stops.
Uses the same pre-registered criteria as the 2025–26 test.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA, RAW
import src.model_hitters as mh

FROZEN = DATA.parent / "results" / "frozen_2026-11-17"


def main():
    man = json.loads((FROZEN / "FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, h in man["sha256"].items():
        if name.startswith("input_"):
            continue
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != h:
            raise SystemExit(f"Frozen file changed after the freeze: {name}. Forward test invalid.")
    lb = RAW.parent / "leaderboards" / "hitters" / "leaderboard_hitters_2027.csv"
    if not lb.exists():
        raise SystemExit("2027 leaderboard not found. Run: python -m src.ingest_leaderboards --years 2027")
    actual = pd.read_csv(lb)[["player_id", "woba", "pa"]].rename(columns={"woba": "woba_2027", "pa": "pa_2027"})
    p = pd.read_csv(FROZEN / "predictions_2027.csv").merge(actual, on="player_id", how="left")
    p["pa_2027"] = p["pa_2027"].fillna(0)
    p["lost_role"] = (p["pa_2027"] < 200).astype(int)
    s = p[p["lost_role"] == 0].copy()
    e_m = np.abs(s["woba_2027"] - s["marcel_2027"])
    e_r = np.abs(s["woba_2027"] - s["ridge_2027"])
    imp, lo, hi = mh.paired_boot(e_m, e_r, s["player_id"])
    resid = s["woba_2027"] - s["marcel_2027"]
    out = {"n_scored": int(len(s)), "lost_role_rate": float(p["lost_role"].mean()),
           "marcel_mae": float(e_m.mean()), "ridge_mae": float(e_r.mean()),
           "ridge_improvement": imp, "ci": [lo, hi],
           "ridge_label": "CONFIRMED" if imp >= mh.MAE_PRACTICAL and lo > 0 else "NOT CONFIRMED"}
    for t, cond in (("breakout", resid >= mh.BREAK), ("regression", resid <= -mh.BREAK)):
        auc, alo, ahi = mh.auc_boot(cond.astype(int), s[f"p_{t}_experimental"], s["player_id"])
        out[f"{t}_auc"] = [auc, alo, ahi]
    (FROZEN / "FORWARD_TEST_RESULT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

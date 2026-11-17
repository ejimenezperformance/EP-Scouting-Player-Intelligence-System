"""Freeze the Phase 3 models for a forward test on the 2027 season.

The models are refit on every labeled season (target seasons 2017–2026) with the
hyperparameters chosen on validation, then saved together with their 2027
predictions and SHA-256 hashes. Nothing here may be edited after the freeze date;
`src/score_forward.py` scores the frozen predictions once 2027 is complete.

Output: results/frozen_2026-11-17/
"""
import hashlib
import json
from datetime import date

import joblib
import pandas as pd

import src.model_hitters as mh
from src.config import DATA

FREEZE = "2026-11-17"
OUT = DATA.parent / "results" / f"frozen_{FREEZE}"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src_path = DATA / "processed" / "model_hitters.parquet"
    raw = pd.read_parquet(src_path)
    summary = json.loads((DATA.parent / "results" / "phase3_summary.json").read_text(encoding="utf-8"))
    # offset from all labeled seasons (train + valid + test)
    lab = raw[(raw["split"].isin(["train", "valid", "test"])) & (raw["lost_role"] == 0)]
    offset = float((lab["woba_next"] - lab["marcel_woba"]).mean())
    d = raw.copy()
    d["marcel_bc"] = d["marcel_woba"] + offset
    d["resid_bc"] = d["woba_next"] - d["marcel_bc"]
    d["breakout"] = (d["resid_bc"] >= mh.BREAK).astype(float).where(d["resid_bc"].notna())
    d["regression"] = (d["resid_bc"] <= -mh.BREAK).astype(float).where(d["resid_bc"].notna())
    fit = d[(d["split"].isin(["train", "valid", "test"])) & (d["lost_role"] == 0)]

    ridge = mh.ridge(summary["regression_meta"]["ridge_alpha"]).fit(mh.X(fit), fit["resid_bc"])
    clfs = {t: mh.logit(summary["classifier_meta"][t]["C"]).fit(mh.X(fit), fit[t])
            for t in ("breakout", "regression")}

    fc = d[d["split"] == "forecast"].copy()
    fc["marcel_2027"] = fc["marcel_bc"]
    fc["ridge_2027"] = fc["marcel_bc"] + ridge.predict(mh.X(fc))
    for t, m in clfs.items():
        fc[f"p_{t}_experimental"] = m.predict_proba(mh.X(fc))[:, 1]
    preds = fc[["player_id", "player_name", "player_age", "pa", "woba", "marcel_2027", "ridge_2027",
                "p_breakout_experimental", "p_regression_experimental"]]
    preds_path = OUT / "predictions_2027.csv"
    preds.to_csv(preds_path, index=False, encoding="utf-8")

    files = {"ridge.joblib": ridge, "logit_breakout.joblib": clfs["breakout"],
             "logit_regression.joblib": clfs["regression"]}
    for name, obj in files.items():
        joblib.dump(obj, OUT / name)

    manifest = {
        "frozen_on": FREEZE, "written_on": date.today().isoformat(),
        "rule": "Do not refit or edit. Score once with src/score_forward.py after the 2027 regular season.",
        "pre_registered_success_criteria": "docs/PHASE3_PREREGISTRATION.md (same thresholds as the 2025–26 test)",
        "marcel_offset": offset, "marcel_K": int(raw["marcel_K"].iloc[0]),
        "features": mh.FEATURES, "n_training_rows": int(len(fit)), "n_predictions": int(len(preds)),
        "status_of_probabilities": "EXPERIMENTAL: not validated in 2025–26 test; not shown on dashboards",
        "sha256": {"input_model_hitters.parquet": sha(src_path), "predictions_2027.csv": sha(preds_path),
                   **{n: sha(OUT / n) for n in files}},
    }
    (OUT / "FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["frozen_on", "n_training_rows", "n_predictions"]}, indent=1))


if __name__ == "__main__":
    main()

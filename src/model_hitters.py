"""Phase 3: does a Statcast model beat Marcel at projecting next-season wOBA?

Pre-registered design: docs/PHASE3_PREREGISTRATION.md
Input : data/processed/model_hitters.parquet
Output: results/*.csv, results/phase3_summary.json, data/processed/predictions.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA

ROOT = DATA.parent
PROC, RES = DATA / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
SEED, N_BOOT, BREAK = 2026, 2000, 0.020
MAE_PRACTICAL = 0.002

FEATURES = ["woba", "xwoba", "xwoba_gap", "xslg", "xba", "barrel_batted_rate", "hard_hit_percent",
            "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent", "k_percent", "bb_percent",
            "whiff_percent", "oz_swing_percent", "sprint_speed", "player_age", "pa",
            "d_xwoba", "d_barrel_batted_rate", "d_exit_velocity_avg", "d_k_percent", "marcel_woba"]


# ------------------------------------------------------------------ helpers
def prepare(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    d = d.copy()
    tr = d[(d["split"] == "train") & (d["lost_role"] == 0)]
    offset = float((tr["woba_next"] - tr["marcel_woba"]).mean())
    d["marcel_bc"] = d["marcel_woba"] + offset
    d["resid_bc"] = d["woba_next"] - d["marcel_bc"]
    d["breakout"] = (d["resid_bc"] >= BREAK).astype(float).where(d["resid_bc"].notna())
    d["regression"] = (d["resid_bc"] <= -BREAK).astype(float).where(d["resid_bc"].notna())
    return d, offset


def X(d):
    return d[FEATURES].astype(float)


def ridge(alpha):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def logit(C):
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def xgb_reg(p):
    return xgb.XGBRegressor(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"],
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                            reg_lambda=5, random_state=SEED, n_jobs=4)


def xgb_clf(p):
    return xgb.XGBClassifier(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"],
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                             reg_lambda=5, random_state=SEED, n_jobs=4, eval_metric="logloss")


GRID_XGB = [{"n": n, "depth": dp, "lr": lr} for n in (150, 400) for dp in (2, 3) for lr in (0.02, 0.05)]


def paired_boot(err_a, err_b, groups, n=N_BOOT, seed=SEED):
    """Cluster bootstrap (by player) of mean(err_a - err_b). Positive = b is better."""
    df = pd.DataFrame({"g": groups, "d": np.asarray(err_a) - np.asarray(err_b)})
    per = df.groupby("g")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    ids = per.index.values
    bs = []
    for _ in range(n):
        s = per.loc[rng.choice(ids, len(ids))]
        bs.append(s["sum"].sum() / s["count"].sum())
    return float(df["d"].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def auc_boot(y, p, groups, n=N_BOOT, seed=SEED):
    df = pd.DataFrame({"g": groups, "y": y, "p": p})
    rng = np.random.default_rng(seed)
    ids = df["g"].unique()
    grp = {k: v for k, v in df.groupby("g")}
    bs = []
    for _ in range(n):
        s = pd.concat([grp[i] for i in rng.choice(ids, len(ids))])
        if s["y"].nunique() == 2:
            bs.append(roc_auc_score(s["y"], s["p"]))
    return float(roc_auc_score(y, p)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ------------------------------------------------------------------ regression track
def regression_track(d):
    lab = d[d["lost_role"] == 0]
    tr, va, te = (lab[lab["split"] == s] for s in ("train", "valid", "test"))
    trva = pd.concat([tr, va])
    # choose hyperparameters on validation
    alphas = [1, 10, 30, 100, 300, 1000]
    va_mae = {a: np.abs(va["resid_bc"] - ridge(a).fit(X(tr), tr["resid_bc"]).predict(X(va))).mean() for a in alphas}
    a_best = min(va_mae, key=va_mae.get)
    xg_mae = {}
    for i, p in enumerate(GRID_XGB):
        m = xgb_reg(p).fit(X(tr), tr["resid_bc"])
        xg_mae[i] = np.abs(va["resid_bc"] - m.predict(X(va))).mean()
    p_best = GRID_XGB[min(xg_mae, key=xg_mae.get)]
    valid_table = {"marcel_bc": float(np.abs(va["resid_bc"]).mean()),
                   "ridge": float(va_mae[a_best]), "xgb": float(min(xg_mae.values()))}
    # refit on train+valid, evaluate once on test
    r = ridge(a_best).fit(X(trva), trva["resid_bc"])
    g = xgb_reg(p_best).fit(X(trva), trva["resid_bc"])
    te = te.copy()
    te["pred_ridge"] = te["marcel_bc"] + r.predict(X(te))
    te["pred_xgb"] = te["marcel_bc"] + g.predict(X(te))
    e_m = np.abs(te["woba_next"] - te["marcel_bc"])
    rows = [{"model": "Marcel (bias-corrected)", "test_mae": e_m.mean(),
             "test_rmse": np.sqrt(((te["woba_next"] - te["marcel_bc"]) ** 2).mean()),
             "improvement_vs_marcel": 0.0, "ci_lo": np.nan, "ci_hi": np.nan}]
    for name, col in (("Ridge", "pred_ridge"), ("XGBoost", "pred_xgb")):
        e = np.abs(te["woba_next"] - te[col])
        imp, lo, hi = paired_boot(e_m, e, te["player_id"])
        rows.append({"model": name, "test_mae": e.mean(),
                     "test_rmse": np.sqrt(((te["woba_next"] - te[col]) ** 2).mean()),
                     "improvement_vs_marcel": imp, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res["label"] = np.where(res["model"].str.startswith("Marcel"), "BASELINE",
                   np.where((res["improvement_vs_marcel"] >= MAE_PRACTICAL) & (res["ci_lo"] > 0),
                            "CONFIRMED", "NOT CONFIRMED"))
    res["n_test"] = len(te)
    res.to_csv(RES / "regression_track.csv", index=False)
    return res, te, r, g, {"ridge_alpha": a_best, "xgb_params": p_best, "valid_mae": valid_table}


# ------------------------------------------------------------------ classification track
def classification_track(d, target):
    lab = d[d["lost_role"] == 0]
    tr, va, te = (lab[lab["split"] == s] for s in ("train", "valid", "test"))
    trva = pd.concat([tr, va])
    Cs = [0.01, 0.03, 0.1, 0.3, 1]
    va_ll = {}
    for C in Cs:
        p = logit(C).fit(X(tr), tr[target]).predict_proba(X(va))[:, 1]
        va_ll[C] = brier_score_loss(va[target], p)
    C_best = min(va_ll, key=va_ll.get)
    xg_b = {}
    for i, p in enumerate(GRID_XGB):
        pr = xgb_clf(p).fit(X(tr), tr[target]).predict_proba(X(va))[:, 1]
        xg_b[i] = brier_score_loss(va[target], pr)
    p_best = GRID_XGB[min(xg_b, key=xg_b.get)]
    base = trva[target].mean()
    lm = logit(C_best).fit(X(trva), trva[target])
    gm = xgb_clf(p_best).fit(X(trva), trva[target])
    te = te.copy()
    te[f"p_{target}_logit"] = lm.predict_proba(X(te))[:, 1]
    te[f"p_{target}_xgb"] = gm.predict_proba(X(te))[:, 1]
    b_ref = brier_score_loss(te[target], np.full(len(te), base))
    rows = []
    for name, col in (("Logistic", f"p_{target}_logit"), ("XGBoost", f"p_{target}_xgb")):
        auc, lo, hi = auc_boot(te[target], te[col], te["player_id"])
        b = brier_score_loss(te[target], te[col])
        rows.append({"target": target, "model": name, "auc": auc, "auc_lo": lo, "auc_hi": hi,
                     "brier": b, "brier_base_rate": b_ref, "brier_skill": 1 - b / b_ref,
                     "base_rate_trainvalid": base, "test_rate": te[target].mean(), "n_test": len(te)})
    out = pd.DataFrame(rows)
    out["label"] = np.where((out["auc_lo"] > 0.55) & (out["brier_skill"] > 0), "CONFIRMED", "NOT CONFIRMED")
    # calibration bins for the logistic model
    te["bin"] = pd.qcut(te[f"p_{target}_logit"], 5, labels=False, duplicates="drop")
    cal = te.groupby("bin").agg(pred=(f"p_{target}_logit", "mean"), obs=(target, "mean"), n=(target, "size"))
    cal.to_csv(RES / f"calibration_{target}.csv")
    return out, te, lm, gm, {"C": C_best, "xgb_params": p_best}


def main():
    raw = pd.read_parquet(PROC / "model_hitters.parquet")
    d, offset = prepare(raw)
    reg, te_reg, r_model, g_model, reg_meta = regression_track(d)
    clf_rows, models = [], {}
    for t in ("breakout", "regression"):
        out, te_c, lm, gm, meta = classification_track(d, t)
        clf_rows.append(out)
        models[t] = (lm, gm, meta)
    clf = pd.concat(clf_rows)
    clf.to_csv(RES / "classification_track.csv", index=False)

    # survivorship: does the regression probability also flag hitters who lost their role?
    te_all = d[d["split"] == "test"].copy()
    lm_reg = models["regression"][0]
    te_all["p_regression"] = lm_reg.predict_proba(X(te_all))[:, 1]
    surv = {"n_test_all": int(len(te_all)), "lost_role_rate": float(te_all["lost_role"].mean())}
    surv["auc_p_regression_vs_lost_role"], surv["auc_lo"], surv["auc_hi"] = auc_boot(
        te_all["lost_role"], te_all["p_regression"], te_all["player_id"])

    # explainability (XGBoost regression model, test set)
    import shap
    sv = shap.TreeExplainer(g_model).shap_values(X(te_reg))
    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values(
        "mean_abs_shap", ascending=False)
    imp.to_csv(RES / "shap_importance_xgb_regression.csv", index=False)
    coefs = pd.DataFrame({"feature": FEATURES,
                          "ridge_coef_std": r_model.named_steps["ridge"].coef_}).sort_values(
        "ridge_coef_std", key=np.abs, ascending=False)
    coefs.to_csv(RES / "ridge_coefficients.csv", index=False)

    # 2027 forecast lists, only from models that passed
    fc = d[d["split"] == "forecast"].copy()
    fc["marcel_bc_2027"] = fc["marcel_bc"]
    passed = {}
    for t in ("breakout", "regression"):
        ok = clf[(clf["target"] == t) & (clf["label"] == "CONFIRMED")]
        if len(ok):
            best = ok.sort_values("auc", ascending=False).iloc[0]["model"]
            lm, gm, _ = models[t]
            fc[f"p_{t}"] = (lm if best == "Logistic" else gm).predict_proba(X(fc))[:, 1]
            passed[t] = best
    keep = ["player_name", "player_id", "player_age", "pa", "woba", "xwoba", "xwoba_gap",
            "barrel_batted_rate", "k_percent", "marcel_bc_2027"] + [f"p_{t}" for t in passed]
    fc[keep].to_csv(RES / "forecast_2027_all.csv", index=False)
    pd.concat([te_reg]).to_parquet(PROC / "predictions_test.parquet", index=False)

    summary = {"marcel_offset": offset, "marcel_K": int(raw["marcel_K"].iloc[0]),
               "n": d.groupby("split").size().to_dict(), "regression_meta": reg_meta,
               "classifier_meta": {k: v[2] for k, v in models.items()},
               "survivorship": surv, "models_passed_for_forecast": passed}
    (RES / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(reg.round(4).to_string())
    print(clf.round(3).to_string())
    print(json.dumps(surv, indent=1))
    print(imp.head(8).round(4).to_string())


if __name__ == "__main__":
    main()

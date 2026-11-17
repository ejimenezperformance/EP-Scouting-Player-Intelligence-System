"""Phase 2: hitter-season table in DuckDB + Marcel baseline + modeling dataset.

Input : data/raw/leaderboards/hitters/*.csv (+ manifests)
Output: DuckDB tables fact_hitter_season, model_hitters ; data/processed/model_hitters.parquet
"""
import json

import duckdb
import numpy as np
import pandas as pd

from src.config import DB_PATH, MANIFESTS, RAW, WAREHOUSE, DATA

LB = RAW.parent / "leaderboards" / "hitters"
PROC = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

TRAIN = range(2017, 2023)   # target seasons
VALID = range(2023, 2025)
TEST = range(2025, 2027)
BREAK = 0.020


def min_pa(season: int) -> int:
    return 80 if season == 2020 else 200


def load_seasons(con) -> pd.DataFrame:
    paths = sorted(LB.glob("leaderboard_hitters_*.csv"))
    for p in paths:  # provenance gate, same rule as Phase 1
        m = json.loads((MANIFESTS / f"{p.stem}.json").read_text(encoding="utf-8"))
        n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{p.as_posix()}')").fetchone()[0]
        if m.get("status") != "ok" or int(m["rows"]) != n:
            raise ValueError(f"manifest mismatch: {p.name}")
    glob = (LB / "leaderboard_hitters_*.csv").as_posix()
    con.execute(f"""CREATE OR REPLACE TABLE fact_hitter_season AS
        SELECT * FROM read_csv_auto('{glob}', union_by_name=true)""")
    return con.execute("SELECT * FROM fact_hitter_season").df()


def marcel(df: pd.DataFrame, K: float) -> pd.DataFrame:
    """Marcel-style wOBA projection for season t+1 from t, t-1, t-2."""
    s = df.set_index(["player_id", "year"])
    ok = df.dropna(subset=["woba"])
    lg = ok.groupby("year").apply(lambda g: np.average(g["woba"], weights=g["pa"]), include_groups=False)
    rows = []
    for (pid, t), r in s.iterrows():
        num = den = lgnum = 0.0
        for w, k in zip((5, 4, 3), (0, 1, 2)):
            key = (pid, t - k)
            if key in s.index and pd.notna(s.loc[key, "woba"]):
                x = s.loc[key]
                num += w * x["pa"] * x["woba"]
                den += w * x["pa"]
                lgnum += w * x["pa"] * lg.get(t - k, lg[t])
        lg_ref = lgnum / den if den else lg[t]
        proj = (num + K * lg_ref) / (den + K)
        age_next = r["player_age"] + 1
        adj = 1 + 0.006 * (29 - age_next) if age_next < 29 else 1 - 0.003 * (age_next - 29)
        rows.append({"player_id": pid, "year": t, "marcel_woba": proj * adj,
                     "lg_woba_t": lg[t], "weighted_pa": den})
    return pd.DataFrame(rows)


def build_dataset(df: pd.DataFrame, K: float) -> pd.DataFrame:
    df = df.copy()
    df["xwoba_gap"] = df["xwoba"] - df["woba"]
    prev = df[["player_id", "year", "xwoba", "barrel_batted_rate", "exit_velocity_avg", "k_percent"]].copy()
    prev["year"] += 1
    df = df.merge(prev, on=["player_id", "year"], how="left", suffixes=("", "_prev"))
    for c in ["xwoba", "barrel_batted_rate", "exit_velocity_avg", "k_percent"]:
        df[f"d_{c}"] = df[c] - df[f"{c}_prev"]
    df = df.merge(marcel(df, K), on=["player_id", "year"], how="left")
    nxt = df[["player_id", "year", "woba", "pa"]].rename(columns={"woba": "woba_next", "pa": "pa_next"})
    nxt["year"] -= 1
    out = df.merge(nxt, on=["player_id", "year"], how="left")
    out["target_season"] = out["year"] + 1
    out = out[out.apply(lambda r: r["pa"] >= min_pa(r["year"]), axis=1)]
    out = out[(out["target_season"] >= 2017) & (out["target_season"] <= 2027)]
    thr = out["target_season"].map(min_pa)
    out["pa_next"] = out["pa_next"].fillna(0)
    out["lost_role"] = (out["pa_next"] < thr).astype(int)
    out["resid"] = np.where(out["lost_role"] == 0, out["woba_next"] - out["marcel_woba"], np.nan)
    out["breakout"] = (out["resid"] >= BREAK).astype(float).where(out["resid"].notna())
    out["regression"] = (out["resid"] <= -BREAK).astype(float).where(out["resid"].notna())
    out["split"] = np.select([out["target_season"].isin(TRAIN), out["target_season"].isin(VALID),
                              out["target_season"].isin(TEST)], ["train", "valid", "test"], "forecast")
    return out


def fit_K(df: pd.DataFrame) -> float:
    """Choose Marcel's regression constant on TRAIN target seasons only."""
    best = None
    for K in [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]:
        d = build_dataset(df, K)
        tr = d[(d["split"] == "train") & (d["lost_role"] == 0)]
        mae = (tr["woba_next"] - tr["marcel_woba"]).abs().mean()
        print(f"K={K}: train MAE {mae:.4f}")
        if best is None or mae < best[1]:
            best = (K, mae)
    return best[0]


def main():
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    df = load_seasons(con)
    df = df[df["year"] >= 2015]
    K = fit_K(df)
    data = build_dataset(df, K)
    data["marcel_K"] = K
    data.to_parquet(PROC / "model_hitters.parquet", index=False)
    con.register("m", data)
    con.execute("CREATE OR REPLACE TABLE model_hitters AS SELECT * FROM m")
    print(f"Marcel K = {K}")
    print(data.groupby("split").agg(n=("player_id", "size"), lost_role=("lost_role", "mean"),
                                     breakout=("breakout", "mean"), regression=("regression", "mean")).round(3))
    con.close()


if __name__ == "__main__":
    main()

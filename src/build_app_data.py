"""Small, shareable table for the Streamlit app (no raw data needed to run the app).

Output: results/app_hitters_2027.csv
"""
import numpy as np
import pandas as pd

from src.config import DATA

RES = DATA.parent / "results"
PCT_COLS = {"xwoba": "xwOBA", "exit_velocity_avg": "Avg EV", "hard_hit_percent": "Hard-Hit%",
            "barrel_batted_rate": "Barrel%", "sweet_spot_percent": "Sweet-Spot%",
            "k_percent": "K%", "bb_percent": "BB%", "whiff_percent": "Whiff%",
            "oz_swing_percent": "Chase%", "sprint_speed": "Sprint speed"}
LOWER_IS_BETTER = {"k_percent", "whiff_percent", "oz_swing_percent"}


def main():
    d = pd.read_parquet(DATA / "processed" / "model_hitters.parquet")
    frozen = pd.read_csv(RES / "frozen_2026-11-17" / "predictions_2027.csv")[["player_id", "marcel_2027"]]
    fc = d[d["split"] == "forecast"].merge(frozen, on="player_id", how="left")
    prev = d[d["year"] == 2025][["player_id", "hard_hit_percent", "exit_velocity_avg"]]
    allprev = pd.read_parquet(DATA / "processed" / "model_hitters.parquet")
    fc = fc.merge(prev.rename(columns={"hard_hit_percent": "hh_2025", "exit_velocity_avg": "ev_2025"}),
                  on="player_id", how="left")
    fc["d_ev"] = fc["exit_velocity_avg"] - fc["ev_2025"]
    fc["d_hh"] = fc["hard_hit_percent"] - fc["hh_2025"]
    fc["signal"] = np.select(
        [fc["d_ev"].isna() | fc["d_hh"].isna(),
         (fc["d_ev"] >= 2) & (fc["d_hh"] >= 5), (fc["d_ev"] <= -2) & (fc["d_hh"] <= -5)],
        ["No 2025 baseline", "Projection may be low", "Projection may be high"], "No material change")
    for c in PCT_COLS:
        r = fc[c].rank(pct=True)
        fc[f"pct_{c}"] = (100 * (1 - r if c in LOWER_IS_BETTER else r)).round(0)
    keep = (["player_id", "player_name", "player_age", "pa", "woba", "xwoba", "xwoba_gap", "marcel_2027",
             "exit_velocity_avg", "hard_hit_percent", "barrel_batted_rate", "sweet_spot_percent",
             "k_percent", "bb_percent", "whiff_percent", "oz_swing_percent", "sprint_speed",
             "d_ev", "d_hh", "signal"] + [f"pct_{c}" for c in PCT_COLS])
    fc[keep].sort_values("marcel_2027", ascending=False).to_csv(RES / "app_hitters_2027.csv", index=False,
                                                                  encoding="utf-8")
    print(len(fc), fc["signal"].value_counts().to_dict())


if __name__ == "__main__":
    main()

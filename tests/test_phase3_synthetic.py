"""Known-answer test for the Phase 3 pipeline.

1) A planted signal (next-season residual depends on xwOBA - wOBA) must be found:
   model beats Marcel, CI above zero, label CONFIRMED.
2) Pure noise must NOT be confirmed.
Run: python -m pytest -q tests/test_phase3_synthetic.py
"""
import numpy as np
import pandas as pd

import src.model_hitters as mh


def fake(signal: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(900):
        for t in range(2016, 2026):
            if rng.random() < 0.35:
                continue
            row = {f: rng.normal(0, 1) for f in mh.FEATURES}
            row.update(player_id=pid, year=t, target_season=t + 1, lost_role=0,
                       marcel_woba=0.315 + rng.normal(0, 0.02))
            row["xwoba_gap"] = rng.normal(0, 0.02)
            row["woba_next"] = row["marcel_woba"] + signal * row["xwoba_gap"] + rng.normal(0, 0.03)
            ts = t + 1
            row["split"] = "train" if ts <= 2022 else "valid" if ts <= 2024 else "test"
            rows.append(row)
    return pd.DataFrame(rows)


def run(signal, tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "RES", tmp_path)
    monkeypatch.setattr(mh, "N_BOOT", 300)
    d, _ = mh.prepare(fake(signal))
    res, *_ = mh.regression_track(d)
    return res.set_index("model")


def test_planted_signal_is_found(tmp_path, monkeypatch):
    res = run(1.0, tmp_path, monkeypatch)
    assert res.loc["Ridge", "label"] == "CONFIRMED"
    assert res.loc["Ridge", "ci_lo"] > 0


def test_noise_is_not_confirmed(tmp_path, monkeypatch):
    res = run(0.0, tmp_path, monkeypatch)
    assert res.loc["Ridge", "label"] == "NOT CONFIRMED"
    assert res.loc["XGBoost", "label"] == "NOT CONFIRMED"

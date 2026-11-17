"""EP Scouting & Player Intelligence — executive view (hitters MVP).

Run from the repository root:
    streamlit run app/streamlit_app.py
Reads only small files in results/ and figures/ — no raw data or database needed.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
NAVY, GOLD, BLUE, BG = "#0B1B33", "#B8862A", "#2A62A8", "#F5F3EC"

st.set_page_config(page_title="EP Scouting Intelligence", page_icon=str(ROOT / "app" / "ep_logo.png"),
                   layout="wide")
st.markdown(f"""<style>
h1, h2, h3 {{ color: {NAVY}; }}
div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
.ep-note {{ background: {BG}; border-left: 4px solid {GOLD}; padding: 10px 14px; margin: 8px 0 14px; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    players = pd.read_csv(RES / "app_hitters_2027.csv")
    reg = pd.read_csv(RES / "regression_track.csv")
    clf = pd.read_csv(RES / "classification_track.csv")
    sig = pd.read_csv(RES / "signal_evaluation.csv")
    man = json.loads((RES / "frozen_2026-11-17" / "FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    fwd = RES / "frozen_2026-11-17" / "FORWARD_TEST_RESULT.json"
    return players, reg, clf, sig, man, (json.loads(fwd.read_text()) if fwd.exists() else None)


players, reg, clf, sig, man, fwd = load()

c1, c2 = st.columns([1, 6])
c1.image(str(ROOT / "app" / "ep_logo.png"), width=110)
c2.title("EP Scouting & Player Intelligence")
c2.caption("Hitters MVP · public Statcast 2015–2026 · Emerson Jimenez, Emerson Performance")

tab_over, tab_board, tab_player, tab_valid = st.tabs(
    ["Executive overview", "2027 board", "Player profile", "Validation"])

# ------------------------------------------------------------------ overview
with tab_over:
    ridge = reg[reg["model"] == "Ridge"].iloc[0]
    best_auc = clf.sort_values("auc", ascending=False).iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hitters projected for 2027", f"{len(players)}")
    k2.metric("Best model vs Marcel", f"{ridge['improvement_vs_marcel']*1000:+.1f}", "needed +2.0", delta_color="off")
    k3.metric("Best breakout AUC (test)", f"{best_auc['auc']:.2f}", "needed CI > 0.55", delta_color="off")
    k4.metric("Models validated", "0 of 6", delta_color="off")
    st.markdown("""<div class="ep-note"><b>What this view can and cannot say.</b> No model beat a bias-corrected
    Marcel projection by the pre-registered margin on the 2025–26 test, so this dashboard shows <b>Marcel 2027
    projections only</b> — no breakout or regression probabilities. Models are frozen (Nov 17, 2026) and will be
    scored on the 2027 season.</div>""", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.image(str(FIG / "phase3_models_vs_marcel.png"))
    b.image(str(FIG / "phase3_classifier_auc.png"))
    st.subheader("Watch list: projection may be low")
    st.caption("Hitters whose avg exit velocity rose ≥ 2 mph and Hard-Hit% ≥ 5 pts from 2025 to 2026. "
               "Historically such hitters still fell back ~15 wOBA points the next season, but ~5 points less "
               "than Marcel expected (exploratory, p = 0.04). Not a breakout forecast.")
    low = players[players["signal"] == "Projection may be low"]
    st.dataframe(low[["player_name", "player_age", "pa", "woba", "xwoba", "marcel_2027", "d_ev", "d_hh"]]
                 .rename(columns={"player_name": "Player", "player_age": "Age", "pa": "PA 2026", "woba": "wOBA 2026",
                                  "xwoba": "xwOBA 2026", "marcel_2027": "Marcel 2027", "d_ev": "Δ EV (mph)",
                                  "d_hh": "Δ Hard-Hit%"}).round(3),
                 hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ board
with tab_board:
    f1, f2, f3 = st.columns(3)
    sigs = f1.multiselect("Signal", sorted(players["signal"].unique()), default=None)
    age = f2.slider("Age", int(players["player_age"].min()), int(players["player_age"].max()),
                    (int(players["player_age"].min()), int(players["player_age"].max())))
    min_pa = f3.slider("Minimum PA in 2026", 200, 700, 200, step=50)
    v = players[(players["player_age"].between(*age)) & (players["pa"] >= min_pa)]
    if sigs:
        v = v[v["signal"].isin(sigs)]
    v = v.assign(gap_to_marcel=v["marcel_2027"] - v["woba"])
    st.dataframe(
        v[["player_name", "player_age", "pa", "woba", "xwoba", "xwoba_gap", "marcel_2027", "gap_to_marcel",
           "barrel_batted_rate", "k_percent", "signal"]].rename(columns={
            "player_name": "Player", "player_age": "Age", "pa": "PA", "woba": "wOBA 2026", "xwoba": "xwOBA 2026",
            "xwoba_gap": "xwOBA − wOBA", "marcel_2027": "Marcel 2027", "gap_to_marcel": "2027 vs 2026",
            "barrel_batted_rate": "Barrel%", "k_percent": "K%", "signal": "Signal"}).round(3),
        hide_index=True, use_container_width=True, height=520)
    st.caption("Marcel 2027 = 3-year PA-weighted wOBA, regressed to league average, age-adjusted, bias-corrected. "
               "It is the baseline every model must beat.")

# ------------------------------------------------------------------ player
with tab_player:
    name = st.selectbox("Player", players["player_name"].sort_values())
    p = players[players["player_name"] == name].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Marcel 2027 wOBA", f"{p['marcel_2027']:.3f}", f"{p['marcel_2027']-p['woba']:+.3f} vs 2026")
    m2.metric("wOBA 2026", f"{p['woba']:.3f}")
    m3.metric("xwOBA 2026", f"{p['xwoba']:.3f}", f"{p['xwoba_gap']:+.3f} vs wOBA")
    m4.metric("Age · PA", f"{int(p['player_age'])} · {int(p['pa'])}")
    labels = {"xwoba": "xwOBA", "exit_velocity_avg": "Avg EV", "hard_hit_percent": "Hard-Hit%",
              "barrel_batted_rate": "Barrel%", "sweet_spot_percent": "Sweet-Spot%", "k_percent": "K% (low = good)",
              "bb_percent": "BB%", "whiff_percent": "Whiff% (low = good)", "oz_swing_percent": "Chase% (low = good)",
              "sprint_speed": "Sprint speed"}
    vals = [p[f"pct_{k}"] for k in labels]
    fig = go.Figure(go.Bar(x=vals, y=list(labels.values()), orientation="h",
                           marker_color=[BLUE if x >= 50 else GOLD for x in vals],
                           text=[f"{x:.0f}" for x in vals], textposition="outside"))
    fig.update_layout(title="2026 percentile among 2026 qualified hitters (≥ 200 PA)", xaxis_range=[0, 105],
                      height=420, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor=BG, paper_bgcolor="white",
                      font=dict(color=NAVY), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    note = {"Projection may be low": "EV and Hard-Hit% both rose sharply in 2026. Historically, such hitters "
                                     "beat Marcel by ~5 wOBA points (exploratory) — but usually still fell back.",
            "Projection may be high": "EV and Hard-Hit% both fell sharply in 2026. Historically −5 points vs "
                                      "Marcel, not statistically significant (p = 0.10).",
            "No material change": "No large change in contact quality from 2025 to 2026.",
            "No 2025 baseline": "Not enough 2025 MLB data to compare."}[p["signal"]]
    st.markdown(f'<div class="ep-note"><b>Signal: {p["signal"]}.</b> {note}</div>', unsafe_allow_html=True)
    st.caption("Physical testing, bat tracking and scouting observations are not yet integrated (roadmap V3–V4).")

# ------------------------------------------------------------------ validation
with tab_valid:
    st.subheader("Pre-registered test (target seasons 2025–2026, used once)")
    st.dataframe(reg[["model", "test_mae", "improvement_vs_marcel", "ci_lo", "ci_hi", "label"]].round(4),
                 hide_index=True, use_container_width=True)
    st.dataframe(clf[["target", "model", "auc", "auc_lo", "auc_hi", "brier_skill", "label"]].round(3),
                 hide_index=True, use_container_width=True)
    st.subheader("Phase 2.1 signal rule, out of time (exploratory)")
    st.dataframe(sig.round(4), hide_index=True, use_container_width=True)
    st.subheader("Forward test 2027")
    if fwd:
        st.json(fwd)
    else:
        st.info(f"Models frozen on {man['frozen_on']} ({man['n_predictions']} hitters). Scored after the 2027 "
                "regular season with src/score_forward.py; SHA-256 hashes guarantee nothing was changed.")

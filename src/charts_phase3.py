"""EP-branded charts for Phase 3 results (palette validated for color-vision deficiency)."""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA

ROOT = DATA.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAVY, GOLD, BLUE, BG, MUTED = "#0B1B33", "#B8862A", "#2A62A8", "#F5F3EC", "#6B7280"
plt.rcParams.update({"font.family": ["Poppins", "DejaVu Sans"], "axes.edgecolor": MUTED,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "text.color": NAVY, "figure.facecolor": BG, "axes.facecolor": BG,
                     "font.size": 12, "axes.titlesize": 14})


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color="#D9D6CC", lw=0.8)
    ax.set_axisbelow(True)


def model_chart():
    r = pd.read_csv(RES / "regression_track.csv")
    r = r[r["model"] != "Marcel (bias-corrected)"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, row in r.iterrows():
        y = len(r) - 1 - i
        ax.plot([row["ci_lo"] * 1000, row["ci_hi"] * 1000], [y, y], color=NAVY, lw=2)
        ax.plot(row["improvement_vs_marcel"] * 1000, y, "o", ms=11, color=BLUE, mec=BG, mew=2)
        ax.text(2.2, y, f"{row['improvement_vs_marcel']*1000:+.1f}  {row['label']}",
                va="center", fontsize=10)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(2, color=GOLD, ls="--", lw=1.5)
    ax.text(1.95, -0.5, "pre-registered practical bar →", color=GOLD, fontsize=10, ha="right")
    ax.set_yticks(range(len(r)), r["model"][::-1])
    ax.set_xlim(-1, 4.6)
    ax.set_ylim(-0.6, len(r) - 0.2)
    ax.set_xlabel("MAE improvement over Marcel (wOBA points × 1000, 95% CI)")
    ax.set_title("Statcast models barely beat a simple projection", loc="left", fontweight="bold")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "phase3_models_vs_marcel.png", dpi=200)


def auc_chart():
    c = pd.read_csv(RES / "classification_track.csv")
    c["name"] = c["target"].str.capitalize() + " · " + c["model"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, row in c.reset_index(drop=True).iterrows():
        y = len(c) - 1 - i
        ax.plot([row["auc_lo"], row["auc_hi"]], [y, y], color=NAVY, lw=2)
        ax.plot(row["auc"], y, "o", ms=11, color=BLUE if row["model"] == "Logistic" else GOLD, mec=BG, mew=2)
        ax.text(row["auc_hi"] + 0.005, y, f"{row['auc']:.3f}", va="center", fontsize=11)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.axvline(0.55, color=GOLD, ls="--", lw=1.5)
    ax.text(0.552, len(c) - 0.45, "required lower bound (0.55)", color=GOLD, fontsize=10)
    ax.set_yticks(range(len(c)), c["name"][::-1])
    ax.set_xlim(0.45, 0.72)
    ax.set_ylim(-0.6, len(c) - 0.1)
    ax.set_xlabel("Test AUC 2025–2026 (95% CI)")
    ax.set_title("Breakout / regression flags: close, but not confirmed", loc="left", fontweight="bold")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "phase3_classifier_auc.png", dpi=200)


def shap_chart():
    s = pd.read_csv(RES / "shap_importance_xgb_regression.csv").head(10)[::-1]
    nice = {"exit_velocity_avg": "Avg exit velocity", "xwoba_gap": "xwOBA − wOBA", "pa": "Plate appearances",
            "marcel_woba": "Marcel projection", "sweet_spot_percent": "Sweet-Spot%", "whiff_percent": "Whiff%",
            "woba": "wOBA", "barrel_batted_rate": "Barrel%", "player_age": "Age", "xwoba": "xwOBA",
            "k_percent": "K%", "bb_percent": "BB%", "launch_angle_avg": "Launch angle", "xslg": "xSLG",
            "xba": "xBA", "sprint_speed": "Sprint speed", "hard_hit_percent": "Hard-Hit%",
            "oz_swing_percent": "Chase%", "d_xwoba": "Δ xwOBA", "d_exit_velocity_avg": "Δ exit velocity",
            "d_barrel_batted_rate": "Δ Barrel%", "d_k_percent": "Δ K%"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([nice.get(f, f) for f in s["feature"]], s["mean_abs_shap"] * 1000, color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP| (wOBA points × 1000)")
    ax.set_title("What the XGBoost model leaned on", loc="left", fontweight="bold")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIG / "phase3_shap.png", dpi=200)


def signal_chart():
    s = pd.read_csv(RES / "signal_evaluation.csv")
    lab = {"POSITIVE_CHANGE": "Positive flag", "NEGATIVE_CHANGE": "Negative flag",
           "NO_MATERIAL_CHANGE": "No flag"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, title in ((axes[0], "mean_next_woba_minus_current", "Next season vs this season"),
                           (axes[1], "mean_next_woba_minus_marcel", "Next season vs Marcel")):
        vals = s[col] * 1000
        ax.bar([lab[x] for x in s["signal"]], vals,
               color=[BLUE if v >= 0 else GOLD for v in vals], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11)
        ax.axhline(0, color=NAVY, lw=1)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylim(-19, 9)
        clean(ax, "y")
    axes[0].set_ylabel("wOBA points × 1000")
    fig.suptitle("Phase 2.1 hitter flag (EV +2 mph & Hard-Hit% +5): what came next", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "phase21_signal_check.png", dpi=200)


if __name__ == "__main__":
    model_chart(); auc_chart(); shap_chart(); signal_chart()
    print("figures saved")

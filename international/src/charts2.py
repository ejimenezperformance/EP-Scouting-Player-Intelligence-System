import sys, pandas as pd, numpy as np, matplotlib.pyplot as plt
OUT,FIG=sys.argv[1],sys.argv[2]
NAVY,GOLD,BLUE,MUTED,GRID="#0B1B33","#B8862A","#2A62A8","#6B7280","#E5E1D6"
plt.rcParams.update({"font.family":["Poppins","DejaVu Sans"],"axes.edgecolor":MUTED,"text.color":NAVY,
 "xtick.color":NAVY,"ytick.color":NAVY,"axes.labelcolor":NAVY,"font.size":14})
def clean(ax):
    for s in("top","right","left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x",color=GRID,lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y",length=0)
# org, cross-fitted, with chance range
o=pd.read_csv(f"{OUT}/hitters_org_crossfit.csv").sort_values("gap_per_100",ascending=False).reset_index(drop=True)
fig,ax=plt.subplots(figsize=(11,8.6)); y=np.arange(len(o))
for i,r in o.iterrows():
    br="Brewers" in r.org
    ax.plot([r.sim_lo,r.sim_hi],[i,i],color="#C9CFD8",lw=6,solid_capstyle="round",zorder=1)
    ax.scatter(r.gap_per_100,i,s=150 if br else 70,color=GOLD if br else NAVY,edgecolor="white",lw=1.5,zorder=3)
ax.set_yticks(y,[n.replace(" "," ") for n in o.org],fontsize=11)
for t in ax.get_yticklabels():
    if "Brewers" in t.get_text(): t.set_fontweight("bold"); t.set_color(GOLD)
ax.axvline(0,color=NAVY,lw=1); ax.set_ylim(len(o)-0.4,-0.6); ax.set_xlim(-14,17)
b=o[o.org.str.contains("Brewers")].iloc[0]; bi=o.index[o.org.str.contains("Brewers")][0]
ax.text(b.sim_hi+0.4,bi,f"{int(b.reached)} of {int(b.n)} (exp. {b.expected:.0f})",ha="left",va="center",fontsize=11,fontweight="bold",color=GOLD)
ax.text(0.99,1.01,"Grey band = range expected by chance alone (95%)",transform=ax.transAxes,ha="right",fontsize=11,color=MUTED)
ax.set_xlabel("DSL hitters reaching AA vs out-of-sample expectation (per 100), classes 2012–2018")
clean(ax); fig.tight_layout(); fig.savefig(f"{FIG}/org_hitters_crossfit.png",dpi=200)
# markets
m=pd.read_csv(f"{OUT}/markets_crossfit.csv").sort_values("gap_per_100",ascending=False).reset_index(drop=True)
fig,ax=plt.subplots(figsize=(10,5.4))
for i,r in m.iterrows():
    ax.plot([r.sim_lo,r.sim_hi],[i,i],color="#C9CFD8",lw=10,solid_capstyle="round",zorder=1)
    ax.scatter(r.gap_per_100,i,s=170,color=BLUE,edgecolor="white",lw=2,zorder=3)
    ax.text(9.6,i,f"{100*r.rate:.0f}% reached AA  ·  n = {int(r.n):,}",va="center",fontsize=12,color=NAVY)
ax.set_yticks(range(len(m)),m.market,fontsize=13); ax.axvline(0,color=NAVY,lw=1)
ax.set_xlim(-10,20); ax.set_xticks([-8,-4,0,4,8]); ax.set_ylim(len(m)-0.5,-0.95)
ax.set_xlabel("Reached AA vs expectation for age + DSL performance (per 100)",loc="left")
ax.text(-10,-0.75,"Grey band = chance range (95%)",fontsize=11,color=MUTED)
clean(ax); fig.tight_layout(); fig.savefig(f"{FIG}/markets.png",dpi=200)
print(m[["market","n","rate","gap_per_100","p_low","p_high"]].round(3))

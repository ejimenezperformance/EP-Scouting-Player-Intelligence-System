import sys, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
OUT,FIG=sys.argv[1],sys.argv[2]
NAVY,GOLD,BLUE,BG,MUTED="#0B1B33","#B8862A","#2A62A8","#FFFFFF","#6B7280"
plt.rcParams.update({"font.family":["Poppins","DejaVu Sans"],"axes.edgecolor":MUTED,"axes.labelcolor":NAVY,
 "xtick.color":NAVY,"ytick.color":NAVY,"text.color":NAVY,"figure.facecolor":BG,"axes.facecolor":BG,"font.size":14})
def clean(ax,g="y"):
    for s in("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis=g,color="#E5E1D6",lw=0.8); ax.set_axisbelow(True)
# 1 age x performance
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,k,lab in zip(axes,["hitters","pitchers"],["Hitters (performance = OBP vs league)","Pitchers (performance = K−BB% vs league)"]):
    d=pd.read_csv(f"{OUT}/{k}_perf_by_age.csv")
    tiers=["Bottom third","Middle third","Top third"]; x=np.arange(3); w=0.38
    for j,(grp,c) in enumerate([("Under 18.5",BLUE),("18.5 and older",GOLD)]):
        v=[100*d[(d.young==grp)&(d.perf_tier==t)].rate.iloc[0] for t in tiers]
        b=ax.bar(x+(j-0.5)*w,v,w,color=c,label=grp)
        for xi,vi in zip(x+(j-0.5)*w,v): ax.text(xi,vi+0.8,f"{vi:.0f}%",ha="center",fontsize=12)
    ax.set_xticks(x,["Bottom third","Middle third","Top third"]); ax.set_title(lab,loc="left",fontsize=14,fontweight="bold"); clean(ax)
axes[0].set_ylabel("Reached AA or higher (%)"); axes[0].legend(frameon=False,title="Age in first DSL season",loc="upper left")
axes[0].set_ylim(0,42)
fig.tight_layout(); fig.savefig(f"{FIG}/age_x_performance.png",dpi=200)
# 2 AUC gain
s=json.load(open(f"{OUT}/summary.json"))
fig,ax=plt.subplots(figsize=(11,4.6))
rows=[("Hitters → AA","hitters","reach_aa"),("Hitters → MLB","hitters","reach_mlb"),("Pitchers → AA","pitchers","reach_aa"),("Pitchers → MLB","pitchers","reach_mlb")]
y=np.arange(len(rows))[::-1]
for yi,(lab,k,oc) in zip(y,rows):
    r=s[k]["models"][oc]
    ax.plot([r["auc_age_only"],r["auc_age_perf"]],[yi,yi],color=MUTED,lw=2,zorder=1)
    ax.scatter(r["auc_age_only"],yi,s=140,color=GOLD,zorder=2,edgecolor="white",lw=2)
    ax.scatter(r["auc_age_perf"],yi,s=140,color=BLUE,zorder=2,edgecolor="white",lw=2)
    ax.text(r["auc_age_only"]-0.01,yi,f"{r['auc_age_only']:.2f}",ha="right",va="center",fontsize=12)
    ax.text(r["auc_age_perf"]+0.01,yi,f"{r['auc_age_perf']:.2f}  (+{r['auc_gain']:.2f})",ha="left",va="center",fontsize=12,fontweight="bold")
ax.set_yticks(y,[r[0] for r in rows]); ax.set_xlim(0.45,0.9); ax.axvline(0.5,color=MUTED,lw=1,ls=":")
ax.set_xlabel("Test AUC, cohorts 2016–2018 (0.5 = coin flip)")
ax.text(0.455,len(rows)-0.35,"● Age only",color=GOLD,fontsize=12); ax.text(0.56,len(rows)-0.35,"● Age + DSL performance",color=BLUE,fontsize=12)
ax.set_ylim(-0.6,len(rows)-0.1); clean(ax,"x"); fig.tight_layout(); fig.savefig(f"{FIG}/auc_gain.png",dpi=200)
# 3 org dot plot hitters
o=pd.read_csv(f"{OUT}/hitters_org_reach_aa.csv").sort_values("over_expected_per_100")
fig,ax=plt.subplots(figsize=(11,8.5))
yy=np.arange(len(o))
cols=[GOLD if "Brewers" in n else "#9AA7B8" for n in o.org]
ax.barh(yy,o.over_expected_per_100,color=cols,height=0.7)
ax.set_yticks(yy,o.org,fontsize=11); ax.axvline(0,color=NAVY,lw=1)
b=o[o.org.str.contains("Brewers")].iloc[0]
ax.text(0.5,list(o.org).index(b.org),f"Brewers: {int(b.reached)} of {int(b.n)} (expected {b.expected:.0f})",ha="left",va="center",fontsize=12,fontweight="bold")
ax.set_xlabel("DSL hitters reaching AA vs model expectation (per 100 players), first DSL season 2012–2018")
clean(ax,"x"); fig.tight_layout(); fig.savefig(f"{FIG}/org_hitters.png",dpi=200)
print("ok")

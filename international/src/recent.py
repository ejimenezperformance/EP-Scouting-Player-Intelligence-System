import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
OUT=sys.argv[1]
P={"hitters":["z_bb_pct","z_k_pct","z_obp","z_slg","z_iso","z_sb_rate","plateAppearances"],
   "pitchers":["z_k_pct","z_bb_pct","z_kbb_pct","z_hr_pct","battersFaced"]}
rows=[]; cur_out={}
for k,perf in P.items():
    d=pd.read_parquet(f'{OUT}/{k}_first_dsl.parquet'); cols=["age"]+perf
    tr=d[d.season.between(2012,2018)]
    m=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)).fit(tr[cols].fillna(0),tr.reach_fullseason)
    rec=d[d.season.between(2021,2023)].dropna(subset=["org"]).copy()
    rec["exp"]=m.predict_proba(rec[cols].fillna(0))[:,1]
    lg=rec.reach_fullseason.mean()
    g=rec.groupby("org").agg(n=("exp","size"),reached=("reach_fullseason","sum"),expected=("exp","sum"))
    g["over_per_100"]=100*(g.reached-g.expected)/g.n
    var=rec.assign(v=rec.exp*(1-rec.exp)).groupby("org").v.sum(); g["z"]=(g.reached-g.expected)/np.sqrt(var)
    g=g.sort_values("over_per_100",ascending=False); g["rank"]=range(1,len(g)+1)
    g.to_csv(f'{OUT}/{k}_org_recent_fullseason.csv')
    b=g.loc["Milwaukee Brewers"]; print(k,"recent 2021-23 Brewers:",b.round(2).to_dict(),"league full-season rate",round(lg,3))
    # out-of-time check of the full-season model on 2021-23
    from sklearn.metrics import roc_auc_score
    print(k,"AUC full-season model on 2021-23:",round(roc_auc_score(rec.reach_fullseason,rec.exp),3))

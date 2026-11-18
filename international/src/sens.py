import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
OUT=sys.argv[1]; rng=np.random.default_rng(7)
def ev(d, cols, oc, base=["age"]):
    c=d[d.season.between(2012,2018)]; tr,te=c[c.season<=2015],c[c.season>=2016]
    def p(cs):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000)).fit(tr[cs].fillna(0),tr[oc]); return m.predict_proba(te[cs].fillna(0))[:,1]
    a,b=p(base),p(base+cols); y=te[oc].values; bs=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),len(y))
        if y[i].min()==y[i].max(): continue
        bs.append(roc_auc_score(y[i],b[i])-roc_auc_score(y[i],a[i]))
    return round(roc_auc_score(y,b),3), round(roc_auc_score(y,b)-roc_auc_score(y,a),3), np.round(np.percentile(bs,[2.5,97.5]),3).tolist()
res={}
h=pd.read_parquet(f'{OUT}/hitters_first_dsl.parquet'); p=pd.read_parquet(f'{OUT}/pitchers_first_dsl.parquet')
hs={"performance only (no playing time)":["z_bb_pct","z_k_pct","z_obp","z_slg","z_iso","z_sb_rate"],
    "K% only":["z_k_pct"], "playing time only":["plateAppearances"]}
ps={"performance only (no playing time)":["z_k_pct","z_bb_pct","z_kbb_pct","z_hr_pct"],
    "K-BB% only":["z_kbb_pct"], "playing time only":["battersFaced"]}
for kind,d,sets in [("hitters",h,hs),("pitchers",p,ps)]:
    for name,cols in sets.items():
        res[f"{kind} | {name}"]=ev(d,cols,"reach_aa")
for k,v in res.items(): print(k, "AUC",v[0],"gain vs age",v[1],"CI",v[2])
json.dump(res, open(f'{OUT}/sensitivity.json','w'), indent=1)

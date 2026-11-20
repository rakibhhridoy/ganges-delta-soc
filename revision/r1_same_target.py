"""Same-target evaluation of cross-delta transfer (RC1 point 1, RC2 G1). Every model is GBR (200 trees,
random_state 42) with in-fold scaling. Groups = sampling site (identical coordinates), so layers of one
profile, and duplicate profiles stored under two IDs, never straddle train and test.
Usage: python3 r1_same_target.py v4 [v4_wosis_only] [polygon]"""
import sys,numpy as np,pandas as pd,warnings,json
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor as G
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline as mp
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
DATA=Path(__file__).resolve().parent.parent/'data';HERE=Path(__file__).resolve().parent/'outputs';HERE.mkdir(exist_ok=True)
P=['ph','clay','silt','sand'];gb=lambda:mp(StandardScaler(),G(random_state=42,n_estimators=200))
def ganges():
    g=pd.read_csv(DATA/'GangesSOC.csv');g.columns=[c.strip() for c in g.columns]
    return pd.DataFrame({'delta':'Ganges','ph':g.pH,'clay':g.Clay,'silt':g.Silt,'sand':g.Sand,'soc':g.SOCC*10,'site':'G_'+g.Location,'source':'field'})
def panel(name):
    # data/h4_delta_samples.csv (v4) is already restricted to 0-100 cm, texture-checked and de-duplicated
    # (provenance/assemble_panel.py); the drop below is a no-op kept as a guard.
    d=pd.read_csv(DATA/('h4_delta_samples_POLY.csv' if name=='polygon' else 'h4_delta_samples.csv'))
    if name=='v4_wosis_only':d=d[d.source=='WoSIS']
    d=d.copy();d['site']=d.delta+'_'+d.lat.round(5).astype(str)+'_'+d.lon.round(5).astype(str)
    n0=len(d);d=d.drop_duplicates(['delta','lat','lon','depth_top_cm','depth_bot_cm','soc']+P)
    out=pd.concat([d[['delta','site','soc','source']+P],ganges()],ignore_index=True)
    return out,n0-len(d)
def site_folds(sites,k,seed):
    u=np.random.default_rng(seed).permutation(np.unique(sites));k=min(k,len(u));f=dict(zip(u,np.arange(len(u))%k));return np.array([f[s] for s in sites]),k
def metrics(y,p):return dict(R2=r2_score(y,p),MAE=float(np.mean(np.abs(p-y))),bias=float(np.mean(p-y)))
def run(name,reps=10):
    D,dropped=panel(name);names=sorted(D.delta.unique());data={t:D[D.delta==t] for t in names}
    res={'panel':name,'exact_duplicate_rows_dropped':int(dropped),'rows_external':int((D.delta!='Ganges').sum()),
         'sites_external':int(D[D.delta!='Ganges'].site.nunique())}
    # (0) published method on this panel: random layer 5-fold diagonal, whole-delta off-diagonal
    src={t:gb().fit(data[t][P].values,data[t].soc.values) for t in names}
    diag_rand={}
    for t in names:
        X,y=data[t][P].values,data[t].soc.values;p=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=42).split(X):p[b]=gb().fit(X[a],y[a]).predict(X[b])
        diag_rand[t]=r2_score(y,p)
    off=[r2_score(data[t].soc.values,src[s].predict(data[t][P].values)) for s in names for t in names if s!=t]
    res['published_method']={'within_median_R2':float(np.median(list(diag_rand.values()))),'cross_median_R2':float(np.median(off)),
        'cross_frac_positive':float(np.mean(np.array(off)>0)),'n_pairs':len(off)}
    # (1) same-target evaluation on site-held-out test folds
    rows=[];pairs=[]
    for t in names:
        q=data[t];X,y,S=q[P].values,q.soc.values,q.site.values
        oth=D[D.delta!=t];pooled=gb().fit(oth[P].values,oth.soc.values)
        tr_pred={s:src[s].predict(X) for s in names if s!=t}  # source models never saw target rows
        acc={'local':[],'baseline':[],'pooled':[],**{f'from_{s}':[] for s in tr_pred}}
        R=3 if len(y)>2000 else reps
        for seed in range(R):
            f,k=site_folds(S,5,seed);pl=np.zeros(len(y));pb=np.zeros(len(y))
            for i in range(k):
                tr,te=f!=i,f==i;pl[te]=gb().fit(X[tr],y[tr]).predict(X[te]);pb[te]=y[tr].mean()
            acc['local'].append(metrics(y,pl));acc['baseline'].append(metrics(y,pb))
        acc['pooled']=[metrics(y,pooled.predict(X))]
        for s,p in tr_pred.items():acc[f'from_{s}']=[metrics(y,p)]
        avg={m:{k2:float(np.mean([a[k2] for a in v])) for k2 in('R2','MAE','bias')} for m,v in acc.items()}
        loc_sd=float(np.std([a['R2'] for a in acc['local']]))
        tr=[avg[f'from_{s}'] for s in tr_pred]
        for s in tr_pred:
            pairs.append(dict(source=s,target=t,R2=avg[f'from_{s}']['R2'],dR2_vs_local=avg[f'from_{s}']['R2']-avg['local']['R2'],
                              MAE_ratio_vs_local=avg[f'from_{s}']['MAE']/avg['local']['MAE'],beats_local_MAE=avg[f'from_{s}']['MAE']<avg['local']['MAE'],
                              beats_baseline_MAE=avg[f'from_{s}']['MAE']<avg['baseline']['MAE'],bias=avg[f'from_{s}']['bias']))
        rows.append(dict(delta=t,rows=len(y),sites=len(np.unique(S)),local_R2=avg['local']['R2'],local_R2_sd=loc_sd,baseline_R2=avg['baseline']['R2'],
             pooled_R2=avg['pooled']['R2'],transfer_R2_median=float(np.median([x['R2'] for x in tr])),
             local_MAE=avg['local']['MAE'],baseline_MAE=avg['baseline']['MAE'],pooled_MAE=avg['pooled']['MAE'],
             transfer_MAE_median=float(np.median([x['MAE'] for x in tr])),pooled_bias=avg['pooled']['bias'],
             transfer_bias_median=float(np.median([x['bias'] for x in tr])),soc_mean=float(y.mean()),soc_sd=float(y.std())))
    T=pd.DataFrame(rows);Pr=pd.DataFrame(pairs)
    res['same_target']={'within_local_median_R2':float(T.local_R2.median()),'n_deltas_local_R2_pos':int((T.local_R2>0).sum()),
        'n_deltas_local_beats_baseline_MAE':int((T.local_MAE<T.baseline_MAE).sum()),'n_deltas':len(T),
        'baseline_median_R2':float(T.baseline_R2.median()),'cross_median_R2':float(Pr.R2.median()),'cross_frac_positive':float((Pr.R2>0).mean()),
        'median_dR2_transfer_minus_local':float(Pr.dR2_vs_local.median()),'median_MAE_ratio_transfer_over_local':float(Pr.MAE_ratio_vs_local.median()),
        'frac_pairs_transfer_beats_local_MAE':float(Pr.beats_local_MAE.mean()),'frac_pairs_transfer_beats_baseline_MAE':float(Pr.beats_baseline_MAE.mean()),
        'median_abs_bias_transfer':float(Pr.bias.abs().median()),'lodo_R2_negative':int((T.pooled_R2<0).sum()),
        'median_MAE_ratio_pooled_over_local':float((T.pooled_MAE/T.local_MAE).median())}
    T.to_csv(HERE/f'r1_{name}_by_delta.csv',index=False);Pr.to_csv(HERE/f'r1_{name}_pairs.csv',index=False)
    (HERE/f'r1_{name}_summary.json').write_text(json.dumps(res,indent=2))
    print(f'\n######## {name}');print(json.dumps(res,indent=1));print(T.round(2).to_string(index=False))
for n in sys.argv[1:]:run(n)

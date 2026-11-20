"""Referee point 1, few-shot: local-only and external+local get the SAME k local sites and are scored on
the SAME remaining test sites; identical GBR settings (200 trees) for every model."""
import sys,numpy as np,pandas as pd,warnings,json
warnings.filterwarnings('ignore')
import os,pathlib;PANEL=os.environ.get('P1_PANEL','v4');OUTDIR=pathlib.Path(__file__).resolve().parent;sys.argv=[sys.argv[0]];exec(open(OUTDIR/'r1_same_target.py').read().split('def run(')[0])
from sklearn.metrics import r2_score
name=PANEL;D,_=panel(name);out=[]
for t in sorted(D.delta.unique()):
    q=D[D.delta==t];u=q.site.unique()
    if len(u)<20:continue
    o=D[D.delta!=t];Xo,yo=o[P].values,o.soc.values;ext=gb().fit(Xo,yo)
    for k in (5,10,15):
        for rep in range(10):
            pp=np.random.default_rng(1000*k+rep).permutation(u);tr=q[q.site.isin(pp[:k])];te=q[~q.site.isin(pp[:k])];y=te.soc.values
            r=dict(delta=t,k=k,rep=rep,n_train_rows=len(tr))
            for m,p in {'local':gb().fit(tr[P].values,tr.soc.values).predict(te[P].values),
                        'external_plus_local':gb().fit(np.vstack([Xo,tr[P].values]),np.r_[yo,tr.soc.values]).predict(te[P].values),
                        'external_only':ext.predict(te[P].values),'baseline_local_mean':np.full(len(y),tr.soc.mean())}.items():
                r[f'{m}_R2']=r2_score(y,p);r[f'{m}_MAE']=float(np.mean(np.abs(p-y)));r[f'{m}_bias']=float(np.mean(p-y))
            out.append(r)
    print('done',t,flush=True)
F=pd.DataFrame(out);F.to_csv(OUTDIR/'outputs'/f'r1_fewshot_{PANEL}.csv',index=False)
S=F.groupby(['delta','k']).mean(numeric_only=True).drop(columns='rep')
print(S[[c for c in S.columns if c.endswith('R2') or c.endswith('MAE')]].round(2).to_string())
M=S.groupby('k').median()
print('\nMEDIAN across deltas'),print(M[[c for c in M.columns if c.endswith('R2') or c.endswith('MAE')]].round(3).to_string())
W=S.assign(ext_beats_local_MAE=S.external_plus_local_MAE<S.local_MAE,ext_beats_local_R2=S.external_plus_local_R2>S.local_R2,local_beats_base_MAE=S.local_MAE<S.baseline_local_mean_MAE)
print('\nshare of deltas:'),print(W.groupby('k')[['ext_beats_local_MAE','ext_beats_local_R2','local_beats_base_MAE']].mean().round(2).to_string())

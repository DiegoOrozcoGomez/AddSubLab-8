import os
import pandas as pd, numpy as np, glob, sys
from sklearn.metrics import f1_score
R=sys.argv[1] if len(sys.argv)>1 else 'results'
cls=pd.read_csv(f'{R}/classes.csv').sort_values('class_id').class_folder.tolist()
UNS=['delta_3d_printer','resin_3d_printer','bench_drill']
SH=[c for c in cls if c not in UNS]
rows=[]
for p in sorted(glob.glob(f'{R}/training/*/seed_*/test_predictions.csv')):
    m,s=p.replace('\\','/').split('/')[-3],p.replace('\\','/').split('/')[-2]
    d=pd.read_csv(p)
    f=f1_score(d.true,d.predicted,labels=cls,average=None,zero_division=0)
    pc=dict(zip(cls,f))
    # 5-class-only problem: restrict to shifted-class frames, labels among shifted
    sub=d[d.true.isin(SH)]
    rows.append(dict(model=m,seed=s,f1_all=f.mean(),f1_shifted=np.mean([pc[c] for c in SH]),
        f1_unshifted=np.mean([pc[c] for c in UNS]),acc_shifted=(sub.true==sub.predicted).mean(),
        acc_unshifted=d[d.true.isin(UNS)].pipe(lambda x:(x.true==x.predicted).mean())))
r=pd.DataFrame(rows)
print(r.head(2))
g=r.groupby('model')[['f1_all','f1_shifted','f1_unshifted','acc_shifted','acc_unshifted']].agg(['mean','std'])*100
print(g.round(2).sort_values(('f1_all','mean'),ascending=False).to_string())
os.makedirs(f'{R}/revision_analysis',exist_ok=True); r.to_csv(f'{R}/revision_analysis/relocated_f1_per_seed.csv',index=False)

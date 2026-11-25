import sys
from pathlib import Path
# Usage: python <script>.py [results_folder]   (default: results)
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = RES / "revision_analysis"; OUT.mkdir(parents=True, exist_ok=True)
MAP = RES / "video_mapping" / "image_to_video.csv"
import pandas as pd, numpy as np, glob, itertools
rng=np.random.default_rng(7)
M=pd.read_csv(MAP); M=M[M.split=='test'][['file','video']]
models=['xception','maxvit_tiny','convnext_tiny','densenet121','swin_tiny','resnet50','efficientnet_b0','vit_base']
def per_video(d):
    A=[]
    for p in sorted(glob.glob(f'{d}/seed_*/test_predictions.csv')):
        x=pd.read_csv(p).merge(M,on='file'); A.append(x.assign(ok=x.true==x.predicted).groupby('video').ok.mean())
    return pd.concat(A,axis=1).mean(1)   # mean over seeds of per-video frame accuracy
acc={m:per_video(RES/'training'/m) for m in models}
acc['vit_base_in21k']=per_video(RES/'training_vit21k'/'vit_base_in21k')
vids=acc['xception'].index
NP=20000
signs=rng.choice([-1,1],size=(NP,len(vids)))
def test(a,b):
    d=(acc[a]-acc[b]).loc[vids].values
    t=d.mean(); null=(signs*d).mean(1)
    return t*100,(np.abs(null)>=abs(t)-1e-12).mean(), (d>0).sum(), (d<0).sum()
rows=[]
for a,b in itertools.combinations(models,2):
    t,p,na,nb=test(a,b); rows.append(dict(A=a,B=b,diff_pts=t,p=p,videos_A_better=na,videos_B_better=nb))
R=pd.DataFrame(rows).sort_values('p')
# Holm
m=len(R); R['p_holm']=np.minimum(1,np.maximum.accumulate((m-np.arange(m))*R.p.values))
R['sig']=R.p_holm<0.05
print(R.round(4).to_string(index=False)); print('significant pairs:',R.sig.sum(),'of',m)
t,p,na,nb=test('vit_base_in21k','vit_base'); print('ViT21k vs ViT1k',round(t,2),p,na,nb)
R.to_csv(OUT/'video_permutation_tests.csv',index=False)

import sys
from pathlib import Path
# Usage: python <script>.py [results_folder]   (default: results)
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = RES / "revision_analysis"; OUT.mkdir(parents=True, exist_ok=True)
MAP = RES / "video_mapping" / "image_to_video.csv"
import pandas as pd, numpy as np, glob, itertools, warnings
warnings.filterwarnings('ignore')
rng=np.random.default_rng(11)
M=pd.read_csv(MAP); M=M[M.split=='test'][['file','video','class_folder']]
V=M.drop_duplicates('video').set_index('video').class_folder; vids=V.index.values
models=['xception','maxvit_tiny','convnext_tiny','densenet121','swin_tiny','resnet50','efficientnet_b0','vit_base']
def mat(d):
    A=[]
    for p in sorted(glob.glob(f'{d}/seed_*/test_predictions.csv')):
        x=pd.read_csv(p).merge(M,on='file'); A.append(x.assign(ok=x.true==x.predicted).groupby('video').ok.mean().reindex(vids).values)
    return np.array(A)
A={m:mat(RES/'training'/m) for m in models}
A['vit_base_in21k']=mat(RES/'training_vit21k'/'vit_base_in21k')
cls=V.unique(); idx={c:np.where(V.values==c)[0] for c in cls}
B=20000
VS=np.concatenate([rng.choice(idx[c],(B,len(idx[c]))) for c in cls],axis=1)  # B x 66
def boot(a,b):
    na,nb=A[a].shape[0],A[b].shape[0]
    sa=rng.integers(0,na,(B,na)); sb=rng.integers(0,nb,(B,nb))
    # mean over sampled seeds then over sampled videos
    ma=A[a][sa].mean(1); mb=A[b][sb].mean(1)          # B x 66
    d=(np.take_along_axis(ma,VS,1)-np.take_along_axis(mb,VS,1)).mean(1)*100
    obs=(A[a].mean(0)-A[b].mean(0)).mean()*100
    p=min(1,2*min((d<=0).mean(),(d>=0).mean()))
    return obs,np.percentile(d,2.5),np.percentile(d,97.5),p
rows=[dict(A=a,B=b,**dict(zip(['diff','lo','hi','p'],boot(a,b)))) for a,b in itertools.combinations(models,2)]
R=pd.DataFrame(rows).sort_values('p'); m=len(R)
R['p_holm']=np.minimum(1,np.maximum.accumulate((m-np.arange(m))*R.p.values)); R['sig']=R.p_holm<0.05
print(R.round(4).to_string(index=False)); print('sig',R.sig.sum())
print('vit21k vs vit1k', [round(x,4) for x in boot('vit_base_in21k','vit_base')])
R.to_csv(OUT/'video_hierarchical_bootstrap.csv',index=False)

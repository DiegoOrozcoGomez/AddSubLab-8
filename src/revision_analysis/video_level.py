import sys
from pathlib import Path
# Usage: python <script>.py [results_folder]   (default: results)
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = RES / "revision_analysis"; OUT.mkdir(parents=True, exist_ok=True)
MAP = RES / "video_mapping" / "image_to_video.csv"
import pandas as pd, numpy as np, glob, itertools, json
from sklearn.metrics import f1_score
rng=np.random.default_rng(2025)
M=pd.read_csv(MAP)
M=M[M.split=='test'][['file','video','class_folder']]
cls=pd.read_csv(RES/'classes.csv').sort_values('class_id').class_folder.tolist()
UNS={'delta_3d_printer','resin_3d_printer','bench_drill'}
src={m:RES/'training'/m for m in ['xception','maxvit_tiny','convnext_tiny','densenet121','swin_tiny','resnet50','efficientnet_b0','vit_base']}
src['vit_base_in21k']=RES/'training_vit21k'/'vit_base_in21k'
P={}
for m,d in src.items():
    for p in sorted(glob.glob(f'{d}/seed_*/test_predictions.csv')):
        s=int(p.replace('\\','/').split('seed_')[1].split('/')[0]); x=pd.read_csv(p)[['file','true','predicted']].merge(M,on='file',how='left')
        assert x.video.notna().all(); P[(m,s)]=x
seeds=sorted({s for _,s in P}); models=list(src)
# videos list
V=M.drop_duplicates('video').set_index('video').class_folder
vids=V.index.values
rows=[]
for (m,s),x in P.items():
    x=x.assign(ok=x.true==x.predicted)
    vote=x.groupby('video').predicted.agg(lambda v:v.value_counts().index[0])
    vacc=(vote==V[vote.index]).mean()
    fr=x.groupby('video').ok.mean()
    rows.append(dict(model=m,seed=s,frame_f1=f1_score(x.true,x.predicted,average='macro')*100,
        frame_acc=x.ok.mean()*100,video_acc=vacc*100,videos_wrong=int((vote!=V[vote.index]).sum()),
        video_f1=f1_score(V[vote.index],vote,average='macro',labels=cls,zero_division=0)*100,
        mean_video_acc=fr.mean()*100))
R=pd.DataFrame(rows)
S=R.groupby('model').agg(frame_f1=('frame_f1','mean'),frame_f1_sd=('frame_f1','std'),video_acc=('video_acc','mean'),
    video_acc_sd=('video_acc','std'),videos_wrong=('videos_wrong','mean'),wrong_max=('videos_wrong','max'),video_f1=('video_f1','mean'))
# cluster bootstrap CI (videos resampled within class) of seed-averaged frame macro F1 and seed-averaged video accuracy
byclass={c:V[V==c].index.values for c in cls}
B=2000
ci={}
for m in models:
    xs=[P[(m,s)] for s in seeds if (m,s) in P]
    # per video: per seed list of (true,pred) arrays
    groups=[{v:g for v,g in x.groupby('video')} for x in xs]
    vote_ok=np.array([[ (g[v].predicted.value_counts().index[0]==V[v]) for v in vids] for g in groups])  # seeds x videos
    boots_f1=[]; boots_v=[]
    for b in range(B):
        samp=np.concatenate([rng.choice(byclass[c],len(byclass[c])) for c in cls])
        f=[]
        for g in groups:
            t=np.concatenate([g[v].true.values for v in samp]); p=np.concatenate([g[v].predicted.values for v in samp])
            f.append(f1_score(t,p,average='macro'))
        boots_f1.append(np.mean(f)*100)
        idx=pd.Index(vids).get_indexer(samp); boots_v.append(vote_ok[:,idx].mean()*100)
    ci[m]=dict(f1_lo=np.percentile(boots_f1,2.5),f1_hi=np.percentile(boots_f1,97.5),v_lo=np.percentile(boots_v,2.5),v_hi=np.percentile(boots_v,97.5))
S=S.join(pd.DataFrame(ci).T)
print(S.sort_values('frame_f1',ascending=False).round(2).to_string())
R.to_csv(OUT/'video_level_per_seed.csv',index=False); S.to_csv(OUT/'video_level_summary.csv')
# wrong videos detail for top models
for m in ['xception','maxvit_tiny','vit_base','vit_base_in21k']:
    w=[]
    for s in seeds:
        x=P[(m,s)]; vote=x.groupby('video').predicted.agg(lambda v:v.value_counts().index[0])
        w+= [f'{v}->{vote[v]}' for v in vote.index if vote[v]!=V[v]]
    print(m, pd.Series(w).value_counts().head(8).to_dict())

import sys, json, glob
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Usage: python per_epoch.py [results_folder]   (default: results)
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
OUT = RES / "revision_analysis"; OUT.mkdir(parents=True, exist_ok=True)
T = RES / "training"
rows = []
for p in sorted(T.glob("*/seed_*/per_epoch_test.csv")):
    d = pd.read_csv(p); d["model"] = p.parent.parent.name; d["seed"] = int(p.parent.name.split("_")[1]); rows.append(d)
pe = pd.concat(rows, ignore_index=True); pe.to_csv(OUT / "per_epoch_all.csv", index=False)
pe.groupby(["model", "epoch"]).mean(numeric_only=True).drop(columns="seed").to_csv(OUT / "per_epoch_mean.csv")
ORDER = ["maxvit_tiny","convnext_tiny","densenet121","swin_tiny","resnet50","efficientnet_b0","xception","vit_base"]
NAME = dict(resnet50="ResNet-50",densenet121="DenseNet-121",xception="Xception",efficientnet_b0="EfficientNet-B0",
            convnext_tiny="ConvNeXt-T",maxvit_tiny="MaxViT-T",swin_tiny="Swin-T",vit_base="ViT-B/16")
C_CNN, C_HYB, C_TR, C_GR = "#2a6fb0", "#c26a1b", "#7a4bb0", "#6b6b6b"
FAM = dict(maxvit_tiny=C_HYB, swin_tiny=C_TR, vit_base=C_TR)
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.6), sharex=True, sharey=True)
for k, (ax, m) in enumerate(zip(axs.flat, ORDER)):
    c = FAM.get(m, C_CNN); d = pe[pe.model == m]
    for s, g in d.groupby("seed"):
        ax.plot(g.epoch, g.test_f1_macro*100, color=c, lw=0.5, alpha=0.45)
        be = json.load(open(T / m / f"seed_{s}" / "metrics.json"))["best_epoch"]
        ax.plot(be, g[g.epoch == be].test_f1_macro.iloc[0]*100, "o", ms=3, color=c, mec="white", mew=0.4, zorder=4)
    ax.plot(*zip(*(d.groupby("epoch").test_f1_macro.mean()*100).items()), color=c, lw=1.6)
    ax.plot(*zip(*(d.groupby("epoch").val_f1_macro.mean()*100).items()), color=C_GR, lw=1.0, ls="--")
    ax.set_title(NAME[m], fontsize=8, pad=3); ax.set_ylim(20, 101); ax.set_xlim(0.5, 15.5); ax.set_xticks([1,3,5,7,9,11,13,15])
    if k % 4 == 0: ax.set_ylabel("Macro F1-score (%)")
    if k >= 4: ax.set_xlabel("Epoch")
hs = [Line2D([],[],color="k",lw=0.5,alpha=0.6,label="Test, each seed"), Line2D([],[],color="k",lw=1.6,label="Test, mean"),
      Line2D([],[],color=C_GR,lw=1.0,ls="--",label="Validation, mean"),
      Line2D([],[],marker="o",ls="",ms=3,color="k",label="Checkpoint selected by early stopping")]
fig.legend(handles=hs, loc="lower center", ncol=4); fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_test_f1_per_epoch.{ext}", dpi=600)

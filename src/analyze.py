# -*- coding: utf-8 -*-
"""
AddSubLab-8: statistical analysis, tables, and figures of the article.

Reads the outputs of train.py (the results/ folder of this repository by
default) and writes tables and IEEE-formatted figures. It does not train
anything and does not need a GPU.

Usage:
    python src/analyze.py [--results results] [--output analysis]

Outputs:
    <output>/tables/   CSV tables and a text report with the statistical tests
    <output>/figures/  figNN_<name>.pdf / .png and figure_index.csv
"""

import argparse
import csv
import json
import math
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import minimize_scalar

import figure_style as F

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results"
OUTPUT_DIR = REPO_DIR / "analysis"

ALPHA = 0.05
ECE_BINS = 15

# Subsampled McNemar: the test images are consecutive video frames, strongly
# correlated, whereas McNemar assumes independent observations. One frame out
# of every STEP frames is taken per class (in frame order), for every possible
# offset (0..STEP-1); a pair is significant in a seed if the MEDIAN of the
# Holm-adjusted p over the offsets is < ALPHA. Several steps are reported as a
# sensitivity analysis.
SUBSAMPLING_STEPS = [10, 20, 40]

# Presentation order and display names.
MODELS = {
    "resnet50":        ("ResNet-50",       "cnn"),
    "densenet121":     ("DenseNet-121",    "cnn"),
    "xception":        ("Xception",        "cnn"),
    "efficientnet_b0": ("EfficientNet-B0", "cnn"),
    "convnext_tiny":   ("ConvNeXt-T",      "cnn"),
    "maxvit_tiny":     ("MaxViT-T",        "hybrid"),
    "swin_tiny":       ("Swin-T",          "transformer"),
    "vit_base":        ("ViT-B/16",        "transformer"),
}
FAMILY_COLOR = {"cnn": F.C_1, "hybrid": F.C_2, "transformer": F.C_3}
FAMILY_NAME = {"cnn": "CNN", "hybrid": "Hybrid CNN + Transformer",
               "transformer": "Transformer"}
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("blue", F.SEQUENTIAL)


# =============================================================================
# UTILITIES
# =============================================================================

def save_table(df, name, index=False):
    d = OUTPUT_DIR / "tables"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv", index=index, encoding="utf-8")
    print(f"  table:  {name}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean_sd(s, dec=2, scale=1.0):
    s = pd.Series(s).dropna() * scale
    if len(s) == 0:
        return ""
    if len(s) == 1:
        return f"{s.iloc[0]:.{dec}f}"
    return f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}"


def name(key):
    return MODELS.get(key, (key, "cnn"))[0]


def family(key):
    return MODELS.get(key, (key, "cnn"))[1]


def family_legend(ax, **kw):
    hs = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markersize=5,
                 label=FAMILY_NAME[f]) for f in ("cnn", "hybrid", "transformer")]
    ax.legend(handles=hs, handletextpad=0.2, borderaxespad=0.2, **kw)


def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


# =============================================================================
# LOADING
# =============================================================================

def load_classes():
    """Class folders (label order), long names, and short names."""
    with open(RESULTS_DIR / "classes.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["class_id"]))
    return ([r["class_folder"] for r in rows], [r["class_name"] for r in rows],
            [r["abbreviation"] for r in rows])


def load_runs(base_folder, class_folders):
    """Finished runs: metrics, history, and per-image predictions."""
    base = RESULTS_DIR / base_folder
    runs = []
    if not base.exists():
        return runs
    for metrics in sorted(base.glob("*/seed_*/metrics.json")):
        d = metrics.parent
        r = read_json(metrics)
        c = {"model": r["model"], "seed": int(r["seed"]), "info": r,
             "history": pd.read_csv(d / "history.csv")}
        for split in ("test", "validation"):
            path = d / f"{split}_predictions.csv"
            if path.exists():
                p = pd.read_csv(path).sort_values("file")
                cols = [f"prob_{k}" for k in class_folders]
                c[f"probs_{split}"] = p[cols].to_numpy(dtype=float)
                idx = {k: i for i, k in enumerate(class_folders)}
                c[f"y_{split}"] = p["true"].map(idx).to_numpy()
                c[f"files_{split}"] = p["file"].to_numpy()
        runs.append(c)
    return runs


# =============================================================================
# METRICS
# =============================================================================

def ece(probs, y, bins=ECE_BINS):
    conf, pred = probs.max(1), probs.argmax(1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def nll(probs, y):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))))


def scale_probs(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(probs_val, y_val):
    """Temperature that minimizes the validation NLL (log-probabilities as logits)."""
    res = minimize_scalar(lambda lt: nll(scale_probs(probs_val, math.exp(lt)), y_val),
                          bounds=(math.log(0.05), math.log(50)), method="bounded")
    return math.exp(res.x)


def f1_per_class(probs, y, k):
    pred = probs.argmax(1)
    out = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return np.array(out)


def confusion(probs, y, k):
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y, probs.argmax(1)), 1)
    return m


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def friedman_nemenyi(table):
    """table: rows = blocks (seeds), columns = models; higher is better."""
    table = table.dropna()
    n, k = table.shape
    chi2, p = stats.friedmanchisquare(*[table[c] for c in table.columns])
    ranks = table.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * n))
    se = math.sqrt(k * (k + 1) / (6 * n))
    pv = pd.DataFrame(1.0, index=table.columns, columns=table.columns)
    for a, b in combinations(table.columns, 2):
        z = abs(ranks[a] - ranks[b]) / se
        pv.loc[a, b] = pv.loc[b, a] = float(stats.studentized_range.sf(
            z * math.sqrt(2), k, np.inf))
    return {"n": n, "k": k, "chi2": chi2, "p": p, "ranks": ranks, "cd": cd, "p_pair": pv}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acc = 0.0
    for i, j in enumerate(order):
        acc = max(acc, min(1.0, (m - i) * p[j]))
        adj[j] = acc
    return adj


def exact_mcnemar(ok_a, ok_b):
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


# =============================================================================
# ANALYSES
# =============================================================================

def search_table(models):
    rows, trials = [], []
    for m in models:
        d = RESULTS_DIR / "hyperparameter_search" / m
        if not (d / "trials_summary.csv").exists():
            continue
        r = pd.read_csv(d / "trials_summary.csv")
        r["model"] = m
        trials.append(r)
        sel = read_json(d / "best_hyperparameters.json")
        rows.append({
            "Model": name(m), "lr": f"{sel['lr']:.2e}",
            "weight_decay": sel["weight_decay"],
            "label_smoothing": sel["label_smoothing"],
            "val_F1_last5": round(sel.get("val_f1_last_epochs", np.nan), 4),
            "trials_complete": int((r.status == "complete").sum()),
            "trials_pruned": int((r.status == "pruned").sum()),
            "search_time_h": round(r.total_time_s.sum() / 3600, 2),
        })
    save_table(pd.DataFrame(rows), "hyperparameter_search")
    return pd.concat(trials, ignore_index=True) if trials else None


def main_analysis(runs, k, long_names):
    rows, per_class = [], []
    for c in runs:
        pt, yt = c["probs_test"], c["y_test"]
        i = c["info"]
        mt = i["test_metrics"]
        row = {"model": c["model"], "seed": c["seed"],
               "accuracy": mt["accuracy"], "precision_macro": mt["precision_macro"],
               "recall_macro": mt["recall_macro"], "f1_macro": mt["f1_macro"],
               "mcc": mt["mcc"], "kappa": mt["kappa"],
               "roc_auc": mt["roc_auc_ovr_macro"], "ece": ece(pt, yt),
               "nll": nll(pt, yt), "val_f1": i["validation_metrics"]["f1_macro"],
               "best_epoch": i["best_epoch"], "epochs": i["epochs_trained"],
               "training_time_s": i["training_time_s"],
               "label_smoothing": i["hyperparameters"]["label_smoothing"]}
        if "probs_validation" in c:
            T = fit_temperature(c["probs_validation"], c["y_validation"])
            pts = scale_probs(pt, T)
            row.update({"T": T, "ece_ts": ece(pts, yt), "nll_ts": nll(pts, yt)})
        rows.append(row)
        for j, f in enumerate(f1_per_class(pt, yt, k)):
            per_class.append({"model": c["model"], "seed": c["seed"],
                              "class": long_names[j], "f1": f})
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def summary_by_model(df, eff):
    rows = []
    for m in [x for x in MODELS if x in set(df.model)]:
        s = df[df.model == m]
        row = {"Model": name(m), "Family": FAMILY_NAME[family(m)], "n_seeds": len(s),
               "Accuracy (%)": mean_sd(s.accuracy, 2, 100),
               "Precision (%)": mean_sd(s.precision_macro, 2, 100),
               "Recall (%)": mean_sd(s.recall_macro, 2, 100),
               "Macro F1 (%)": mean_sd(s.f1_macro, 2, 100),
               "MCC": mean_sd(s.mcc, 3), "Kappa": mean_sd(s.kappa, 3),
               "ROC-AUC": mean_sd(s.roc_auc, 4), "ECE": mean_sd(s.ece, 3)}
        if "ece_ts" in s:
            row["ECE after TS"] = mean_sd(s.ece_ts, 3)
        row["Label smoothing"] = s.label_smoothing.iloc[0]
        row["Best epoch"] = mean_sd(s.best_epoch, 1)
        row["Training time (min)"] = mean_sd(s.training_time_s, 1, 1 / 60)
        if eff is not None and m in eff.index:
            e = eff.loc[m]
            row.update({"Params (M)": round(e.parameters / 1e6, 1),
                        "GFLOPs": round(e.gflops, 2),
                        "GPU latency (ms)": round(e.gpu_fp32_latency_ms, 2),
                        "CPU latency (ms)": round(e.cpu_latency_ms, 1),
                        "Throughput (img/s)": round(e.gpu_bf16_throughput_img_s)})
        rows.append(row)
    return pd.DataFrame(rows)


def subsample_indices(files, y, step, offset):
    """One out of every `step` frames per class, in natural file-name order."""
    idx = []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        pos = pos[sorted(range(len(pos)), key=lambda i: natural_key(files[pos[i]]))]
        idx.extend(pos[offset::step])
    return np.sort(np.array(idx))


def summarize_mcnemar(det, p_col="p_holm"):
    res = []
    for (a, b), g in det.groupby(["A", "B"], sort=False):
        sig = g[g[p_col] < ALPHA]
        res.append({"A": a, "B": b, "seeds": len(g),
                    "seeds_significant": len(sig),
                    "A_better_significant": int((sig.better == a).sum()),
                    "B_better_significant": int((sig.better == b).sum())})
    r = pd.DataFrame(res)
    r["consistent"] = ((r.seeds_significant >= r.seeds - 1) &
                       ((r.A_better_significant == 0) | (r.B_better_significant == 0)))
    return r


def subsampled_mcnemar(runs, step):
    by_seed = {}
    for c in runs:
        by_seed.setdefault(c["seed"], {})[c["model"]] = c
    rows = []
    for s, d in sorted(by_seed.items()):
        ms = [m for m in MODELS if m in d]
        pairs = list(combinations(ms, 2))
        ref = d[ms[0]]
        hits = {m: d[m]["probs_test"].argmax(1) == d[m]["y_test"] for m in ms}
        p_holm = np.zeros((step, len(pairs)))
        n_ab = np.zeros((step, len(pairs), 2))
        for off in range(step):
            idx = subsample_indices(ref["files_test"], ref["y_test"], step, off)
            raw = [exact_mcnemar(hits[a][idx], hits[b][idx]) for a, b in pairs]
            p_holm[off] = holm([p for _, _, p in raw])
            n_ab[off] = [(x, y) for x, y, _ in raw]
        for k, (a, b) in enumerate(pairs):
            na, nb = n_ab[:, k, 0].sum(), n_ab[:, k, 1].sum()
            rows.append({"seed": s, "A": name(a), "B": name(b), "step": step,
                         "images_per_subsample": len(idx),
                         "median_p_holm": float(np.median(p_holm[:, k])),
                         "fraction_subsamples_significant":
                             float(np.mean(p_holm[:, k] < ALPHA)),
                         "better": name(a) if na > nb else name(b)})
    det = pd.DataFrame(rows)
    return det, summarize_mcnemar(det, "median_p_holm")


def mcnemar_tests(runs):
    by_seed = {}
    for c in runs:
        by_seed.setdefault(c["seed"], {})[c["model"]] = c
    rows = []
    for s, d in sorted(by_seed.items()):
        ms = [m for m in MODELS if m in d]
        pairs = list(combinations(ms, 2))
        raw = []
        for a, b in pairs:
            ca, cb = d[a], d[b]
            assert np.array_equal(ca["files_test"], cb["files_test"]), \
                "The test predictions do not correspond to the same images."
            ok_a = ca["probs_test"].argmax(1) == ca["y_test"]
            ok_b = cb["probs_test"].argmax(1) == cb["y_test"]
            raw.append(exact_mcnemar(ok_a, ok_b))
        adj = holm([p for _, _, p in raw])
        for (a, b), (n01, n10, p), pa in zip(pairs, raw, adj):
            rows.append({"seed": s, "A": name(a), "B": name(b),
                         "only_A_correct": n01, "only_B_correct": n10,
                         "p": p, "p_holm": pa,
                         "better": name(a) if n01 > n10 else name(b)})
    det = pd.DataFrame(rows)
    if det.empty:
        return det, det
    return det, summarize_mcnemar(det)


def compare_pairs(base, other, tag):
    """Paired comparison (model, seed) of two sets of runs."""
    b = base.set_index(["model", "seed"])
    o = other.set_index(["model", "seed"])
    common = b.index.intersection(o.index)
    if len(common) == 0:
        return None, None
    t = pd.DataFrame({
        "f1_15": b.loc[common, "f1_macro"] * 100,
        f"f1_{tag}": o.loc[common, "f1_macro"] * 100,
        "acc_15": b.loc[common, "accuracy"] * 100,
        f"acc_{tag}": o.loc[common, "accuracy"] * 100,
        "best_epoch_15": b.loc[common, "best_epoch"],
        f"best_epoch_{tag}": o.loc[common, "best_epoch"],
    }).reset_index()
    t["diff_f1"] = t[f"f1_{tag}"] - t["f1_15"]
    t["model"] = t.model.map(name)
    w = stats.wilcoxon(t.diff_f1) if len(t) >= 6 and (t.diff_f1 != 0).any() else None
    return t, w


def correlations(df):
    d = df.copy()
    g = d.groupby("model")
    rows = []
    for x, desc in [("val_f1", "Validation F1 vs test F1 (within model)"),
                    ("best_epoch", "Best epoch vs test F1 (within model)")]:
        xc = d[x] - g[x].transform("mean")
        yc = d.f1_macro - g.f1_macro.transform("mean")
        r = stats.spearmanr(xc, yc)
        rows.append({"analysis": desc, "n": len(d), "spearman_rho": round(r.statistic, 3),
                     "p": round(r.pvalue, 4), "type": "exploratory"})
    return pd.DataFrame(rows)


# =============================================================================
# FIGURES (IEEE Access format)
# =============================================================================

def fig_test_results(df, order):
    fig, ax = plt.subplots(figsize=(F.COLUMN_WIDTH, 62 * F.MM), layout="constrained")
    rng = np.random.default_rng(0)
    top = max(max(df[df.model == m].f1_macro.max() * 100,
                  (df[df.model == m].f1_macro * 100).mean()
                  + (df[df.model == m].f1_macro * 100).std(ddof=1)) for m in order)
    for i, m in enumerate(order):
        v = df[df.model == m].f1_macro.to_numpy() * 100
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=8,
                   color=FAMILY_COLOR[family(m)], alpha=0.6, linewidths=0, zorder=3)
        sd = v.std(ddof=1) if len(v) > 1 else 0
        ax.errorbar(i, v.mean(), yerr=sd, fmt="_", color=F.C_TXT, markersize=9,
                    lw=0.8, markeredgewidth=1.2, zorder=4)
        ax.text(i, top + 1.0, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_ylim(top=top + 4.0)
    ax.set_yticks([t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= 100])
    ax.spines["left"].set_bounds(ax.get_ylim()[0], 100)
    ax.set_xticks(range(len(order)), [name(m) for m in order], rotation=35, ha="right")
    ax.set_ylabel("Test macro F1-score (%)")
    ax.grid(axis="x", visible=False)
    family_legend(ax, loc="lower left", fontsize=6)
    F.save(fig, "test_results")


def fig_cd(fn):
    r = fn["ranks"].sort_values()
    k, cd = fn["k"], fn["cd"]
    fig, ax = plt.subplots(figsize=(F.PAGE_WIDTH, 48 * F.MM))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.set_xlim(-1.4, k + 2.6)
    ax.set_ylim(-0.02, 1)
    ax.axis("off")
    y0 = 0.74
    ax.plot([1, k], [y0, y0], color=F.C_TXT, lw=0.8)
    for t in range(1, k + 1):
        ax.plot([t, t], [y0, y0 + 0.03], color=F.C_TXT, lw=0.8)
        ax.text(t, y0 + 0.06, str(t), ha="center", fontsize=8)
    ax.text(0.6, y0 + 0.06, "Average rank", ha="right", fontsize=8)
    ax.plot([1, 1 + cd], [0.95, 0.95], color=F.C_TXT, lw=1)
    for xx in (1, 1 + cd):
        ax.plot([xx, xx], [0.93, 0.97], color=F.C_TXT, lw=1)
    ax.text(1 + cd + 0.1, 0.95, f"CD = {cd:.2f}", ha="left", va="center", fontsize=8)
    half = math.ceil(len(r) / 2)
    for j, (m, v) in enumerate(r.items()):
        left = j < half
        yy = 0.50 - 0.13 * (j if left else len(r) - 1 - j)
        xx = 0.6 if left else k + 0.4
        ax.plot([v, v, xx], [y0, yy, yy], color=FAMILY_COLOR[family(m)], lw=0.8)
        ax.text(xx + (-0.05 if left else 0.05), yy, f"{name(m)} ({v:.1f})",
                ha="right" if left else "left", va="center", fontsize=8)
    # Groups without significant differences (horizontal bars).
    groups = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r.iloc[j + 1] - r.iloc[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for n, (i, j) in enumerate(groups):
        yy = y0 - 0.05 - 0.04 * n
        ax.plot([r.iloc[i] - 0.03, r.iloc[j] + 0.03], [yy, yy], color=F.C_TXT, lw=1.8,
                solid_capstyle="round")
    F.save(fig, "critical_difference")


def fig_confusion(runs, order, short):
    k = len(short)
    fig, axs = plt.subplots(2, 4, figsize=(F.PAGE_WIDTH, 112 * F.MM))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.995, bottom=0.19,
                        wspace=0.08, hspace=0.75)
    for n, (ax, m) in enumerate(zip(axs.ravel(), order)):
        cm = sum(confusion(c["probs_test"], c["y_test"], k)
                 for c in runs if c["model"] == m)
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100)
        ax.grid(False)
        for i in range(k):
            for j in range(k):
                if pct[i, j] >= 0.5:
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                            fontsize=5, color="white" if pct[i, j] > 55 else F.C_TXT)
        first_col = n % 4 == 0
        ax.set_xticks(range(k), short, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(k), short if first_col else [], fontsize=5.5)
        ax.tick_params(length=1.5, pad=1)
        if not first_col:
            ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        F.subfig_xlabel(ax, n, name(m))
    fig.text(0.55, 0.005, "Columns: predicted class; rows: true class; "
             "values: percentage of the true class", ha="center", va="bottom", fontsize=8)
    F.save(fig, "confusion_matrices")


def fig_f1_per_class(pc, order, long_names, short):
    t = pc.groupby(["model", "class"]).f1.mean().unstack()[long_names].loc[order] * 100
    fig, ax = plt.subplots(figsize=(F.PAGE_WIDTH, 72 * F.MM), layout="constrained")
    vmin = max(0, math.floor(t.values.min() / 10) * 10)
    im = ax.imshow(t.values, cmap=CMAP, vmin=vmin, vmax=100, aspect="auto")
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if (v - vmin) / (100 - vmin) > 0.55 else F.C_TXT)
    ax.set_xticks(range(len(short)), short, rotation=25, ha="right")
    ax.set_yticks(range(len(order)), [name(m) for m in order])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Per-class F1-score (%)")
    cb.outline.set_visible(False)
    F.save(fig, "f1_per_class")


# Label offsets in the efficiency figure (points: dx, dy, alignment). If |dx| or
# |dy| > 12, a leader line is drawn to the point. Adjust if labels overlap.
LABEL_OFFSETS = {
    ("gpu_latency", "maxvit_tiny"): (-6, 5, "right"),
    ("gpu_latency", "xception"): (5, 3, "left"),
    ("gflops", "densenet121"): (-6, 3, "right"),
    ("gflops", "resnet50"): (-6, -7, "right"),
    ("gflops", "xception"): (0, 7, "center"),
    ("gflops", "vit_base"): (-6, 3, "right"),
    ("gflops", "maxvit_tiny"): (28, -14, "left"),
    ("gflops", "convnext_tiny"): (28, -20, "left"),
    ("gflops", "swin_tiny"): (28, -24, "left"),
}


def fig_efficiency(df, eff, order):
    if eff is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(F.PAGE_WIDTH, 72 * F.MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27, wspace=0.08)
    for n, (ax, col, xl, tag) in enumerate([
            (axs[0], "gpu_fp32_latency_ms", "GPU latency, batch size 1 (ms)", "gpu_latency"),
            (axs[1], "gflops", "Computational cost (GFLOPs)", "gflops")]):
        for m in order:
            if m not in eff.index:
                continue
            v = df[df.model == m].f1_macro * 100
            x = eff.loc[m, col]
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=4,
                        color=FAMILY_COLOR[family(m)], mec=F.C_BG, mew=0.8,
                        elinewidth=0.7, capsize=0)
            dx, dy, ha = LABEL_OFFSETS.get((tag, m), (5, 3, "left"))
            leader = abs(dx) > 12 or abs(dy) > 12
            ax.annotate(name(m), (x, v.mean()), xytext=(dx, dy), ha=ha,
                        va="center" if leader else "baseline",
                        textcoords="offset points", fontsize=6.5,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=F.C_TXT2,
                                        shrinkA=1, shrinkB=2.5) if leader else None)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(f"{xl}, log scale")
        F.subfig_label(ax, n, dy=-0.22)
    axs[0].set_ylabel("Test macro F1-score (%)")
    hs = [Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], markersize=5,
                 label=FAMILY_NAME[f]) for f in ("cnn", "hybrid", "transformer")]
    fig.legend(handles=hs, loc="upper center", ncol=3, handletextpad=0.2,
               columnspacing=1.5, bbox_to_anchor=(0.53, 1.0))
    F.save(fig, "efficiency_vs_performance")


def fig_curves(runs, order):
    for metric, (col_tr, col_va), key, ylab in [
            ("loss", ("train_loss", "val_loss"), "loss_curves", "Loss"),
            ("acc", ("train_acc", "val_acc"), "accuracy_curves", "Accuracy (%)")]:
        fig, axs = plt.subplots(2, 4, figsize=(F.PAGE_WIDTH, 100 * F.MM),
                                sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.14,
                            wspace=0.08, hspace=0.45)
        for n, (ax, m) in enumerate(zip(axs.ravel(), order)):
            sc = 100 if metric == "acc" else 1
            for c in [c for c in runs if c["model"] == m]:
                h = c["history"]
                ax.plot(h.epoch, h[col_tr] * sc, color=F.C_1, lw=0.7, alpha=0.7)
                ax.plot(h.epoch, h[col_va] * sc, color=F.C_2, lw=0.7, alpha=0.7)
                b = c["info"]["best_epoch"]
                ax.plot(b, h.loc[h.epoch == b, col_va].iloc[0] * sc, "o", ms=2.5,
                        color=F.C_2, mec=F.C_BG, mew=0.5)
            F.subfig_xlabel(ax, n, name(m))
            ax.tick_params(labelbottom=True)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab)
        fig.supxlabel("Epoch", fontsize=8, y=0.01)
        hs = [Line2D([], [], color=F.C_1, lw=1, label="Training"),
              Line2D([], [], color=F.C_2, lw=1, label="Validation"),
              Line2D([], [], marker="o", ls="", ms=3, color=F.C_2,
                     label="Selected epoch (lowest validation loss)")]
        fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0))
        F.save(fig, key)


def fig_search(trials, order):
    if trials is None:
        return
    fig, axs = plt.subplots(2, 4, figsize=(F.PAGE_WIDTH, 100 * F.MM),
                            sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.14,
                        wspace=0.08, hspace=0.5)
    for n, (ax, m) in enumerate(zip(axs.ravel(), order)):
        r = trials[trials.model == m]
        sel = read_json(RESULTS_DIR / "hyperparameter_search" / m /
                        "best_hyperparameters.json")["trial"]
        for ls, col in [(0.0, F.C_1), (0.1, F.C_2)]:
            for status, filled in [("complete", True), ("pruned", False)]:
                s = r[(r.label_smoothing == ls) & (r.status == status)]
                ax.scatter(s.lr, s.val_f1_last_epochs * 100, s=12,
                           facecolors=col if filled else "none", edgecolors=col,
                           linewidths=0.7, zorder=3)
        # The star is filled with the color of the selected label smoothing, so that
        # a nearby trial with the other value cannot show through it.
        s = r[r.trial == sel]
        star_col = F.C_1 if float(s.label_smoothing.iloc[0]) == 0.0 else F.C_2
        ax.scatter(s.lr, s.val_f1_last_epochs * 100, s=70, marker="*",
                   facecolors=star_col, edgecolors=F.C_TXT, linewidths=0.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(5e-6, 2e-3)
        ax.set_xticks([1e-5, 1e-4, 1e-3])
        ax.tick_params(labelbottom=True)
        F.subfig_xlabel(ax, n, name(m))
    fig.supxlabel("Learning rate (log scale)", fontsize=8, y=0.01)
    fig.text(0.01, 0.55, "Validation macro F1-score, mean of last 5 epochs (%)",
             rotation=90, ha="left", va="center", fontsize=8)
    hs = [Line2D([], [], marker="o", ls="", ms=3.5, color=F.C_1, label="Label smoothing 0"),
          Line2D([], [], marker="o", ls="", ms=3.5, color=F.C_2, label="Label smoothing 0.1"),
          Line2D([], [], marker="o", ls="", ms=3.5, mfc="none", mec=F.C_TXT2,
                 label="Pruned trial (hollow)"),
          Line2D([], [], marker="*", ls="", ms=7, mfc="none", mec=F.C_TXT,
                 label="Selected configuration (filled with its color)")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0))
    F.save(fig, "hyperparameter_search")


def fig_calibration(df, order):
    if "ece_ts" not in df:
        return
    g = df.groupby("model")
    a, b = g.ece.mean().reindex(order), g.ece_ts.mean().reindex(order)
    sa, sb = g.ece.std().reindex(order), g.ece_ts.std().reindex(order)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(F.COLUMN_WIDTH, 60 * F.MM), layout="constrained")
    ek = {"elinewidth": 0.6, "ecolor": F.C_TXT2}
    ax.bar(x - w / 2 - 0.01, a, w, yerr=sa, color=F.C_1, label="Before temperature scaling",
           error_kw=ek)
    ax.bar(x + w / 2 + 0.01, b, w, yerr=sb, color=F.C_2, label="After temperature scaling",
           error_kw=ek)
    ax.set_xticks(x, [name(m) for m in order], rotation=35, ha="right")
    ax.set_ylabel("Expected calibration error (test)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside upper center", ncol=2, fontsize=6.5, handlelength=1.2,
               columnspacing=1.0)
    F.save(fig, "calibration_ece")


def fig_pairs(t, tag, legend_text, key):
    if t is None:
        return
    mods = [name(m) for m in MODELS if name(m) in set(t.model)]
    if len(mods) <= 5:
        fig, axs = plt.subplots(1, len(mods), figsize=(F.PAGE_WIDTH, 62 * F.MM),
                                sharey=True)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.25, wspace=0.12)
    else:
        cols = int(np.ceil(len(mods) / 2))
        fig, axs = plt.subplots(2, cols, figsize=(F.PAGE_WIDTH, 105 * F.MM),
                                sharey=True)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.14, wspace=0.12,
                            hspace=0.5)
        for ax in axs.ravel()[len(mods):]:
            ax.set_visible(False)
        for ax in axs[:, 0]:
            ax.set_ylabel("Test macro F1-score (%)")
    fig.supxlabel("Training epochs", fontsize=8, y=0.01)
    axs = np.atleast_1d(axs).ravel()
    for n, (ax, m) in enumerate(zip(axs, mods)):
        for _, f in t[t.model == m].iterrows():
            ax.plot([0, 1], [f.f1_15, f[f"f1_{tag}"]], color=F.C_TXT2, lw=0.7, zorder=2)
            ax.plot(0, f.f1_15, "o", color=F.C_1, ms=4, mec=F.C_BG, mew=0.6, zorder=3)
            ax.plot(1, f[f"f1_{tag}"], "o", color=F.C_2, ms=4, mec=F.C_BG, mew=0.6,
                    zorder=3)
        ax.set_xticks([0, 1], ["15", tag])
        ax.set_xlim(-0.4, 1.4)
        ax.grid(axis="x", visible=False)
        F.subfig_xlabel(ax, n, m)
    axs[0].set_ylabel("Test macro F1-score (%)")
    hs = [Line2D([], [], marker="o", ls="", ms=4, color=F.C_1,
                 label="15 epochs (main protocol)"),
          Line2D([], [], marker="o", ls="", ms=4, color=F.C_2, label=legend_text)]
    fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.0))
    F.save(fig, key)


def fig_val_test(df):
    fig, ax = plt.subplots(figsize=(F.COLUMN_WIDTH, 70 * F.MM), layout="constrained")
    for m in MODELS:
        s = df[df.model == m]
        ax.scatter(s.val_f1 * 100, s.f1_macro * 100, s=12,
                   color=FAMILY_COLOR[family(m)], edgecolors=F.C_BG, linewidths=0.4,
                   zorder=3)
    lo = min(df.val_f1.min(), df.f1_macro.min()) * 100 - 2
    ax.plot([lo, 100.5], [lo, 100.5], color=F.C_TXT2, lw=0.6, zorder=1)
    ax.set_xlim(lo, 100.5)
    ax.set_ylim(lo, 100.5)
    ax.set_xlabel("Validation macro F1-score (%)")
    ax.set_ylabel("Test macro F1-score (%)")
    family_legend(ax, loc="upper left", fontsize=6.5)
    F.save(fig, "validation_vs_test")


# =============================================================================
# MAIN
# =============================================================================

def main():
    F.style()
    print(f"Reading results from: {RESULTS_DIR}")
    folders, long_names, short = load_classes()
    k = len(folders)
    report = []

    def line(s=""):
        report.append(s)
        print(s)

    eff = None
    if (RESULTS_DIR / "efficiency.csv").exists():
        eff = pd.read_csv(RESULTS_DIR / "efficiency.csv").set_index("model")

    # ---------------------------------------------------------------- Phase 2
    runs = load_runs("training", folders)
    if not runs:
        raise SystemExit(f"No runs found in {RESULTS_DIR / 'training'}.")
    df, pc = main_analysis(runs, k, long_names)
    save_table(df, "per_run_details")
    order = list(df.groupby("model").f1_macro.mean().sort_values(ascending=False).index)

    trials = search_table([m for m in MODELS if m in set(df.model)])
    save_table(summary_by_model(df, eff), "main_results")

    t_class = (pc.assign(f1=pc.f1 * 100).groupby(["model", "class"]).f1
               .agg(lambda s: mean_sd(s, 2)).unstack()[long_names])
    t_class.index = [name(m) for m in t_class.index]
    save_table(t_class.loc[[name(m) for m in order]], "f1_per_class", index=True)

    # ---------------------------------------------------------------- Statistics
    line("\n=== Friedman + Nemenyi (test macro F1; blocks = seeds) ===")
    fn = friedman_nemenyi(df.pivot(index="seed", columns="model", values="f1_macro"))
    line(f"n = {fn['n']} seeds, k = {fn['k']} models")
    line(f"Friedman chi2 = {fn['chi2']:.3f}, p = {fn['p']:.4g}")
    line(f"Nemenyi critical difference (alpha = {ALPHA}): {fn['cd']:.3f}")
    line("Average ranks: " + ", ".join(f"{name(m)} {v:.2f}"
                                       for m, v in fn["ranks"].sort_values().items()))
    sig = [(a, b, fn["p_pair"].loc[a, b]) for a, b in combinations(fn["p_pair"].index, 2)
           if fn["p_pair"].loc[a, b] < ALPHA]
    line("Significant pairs: " + ("; ".join(f"{name(a)} vs {name(b)} (p = {p:.4f})"
                                           for a, b, p in sig) or "none"))
    pp = fn["p_pair"].copy()
    pp.index = pp.columns = [name(m) for m in pp.index]
    save_table(pp.round(4), "nemenyi_p", index=True)

    line("\n=== Exact McNemar per seed (Holm correction, 28 pairs) ===")
    det, res = mcnemar_tests(runs)
    save_table(det, "mcnemar_detail")
    save_table(res, "mcnemar_summary")
    if not res.empty:
        cons = res[res.seeds_significant == res.seeds]
        line(f"Pairs significant in all seeds: {len(cons)} of {len(res)}")
        line("WARNING: treats all frames as independent (optimistic p-values).")

    line("\n=== McNemar with frame subsampling (median p_Holm over offsets) ===")
    line("Consistent = significant in >= 4 of 5 seeds, with no reversals.")
    sens = []
    for step in SUBSAMPLING_STEPS:
        det_s, res_s = subsampled_mcnemar(runs, step)
        if res_s.empty:
            continue
        save_table(det_s, f"mcnemar_subsampled_{step}_detail")
        save_table(res_s, f"mcnemar_subsampled_{step}_summary")
        n_img = int(det_s.images_per_subsample.iloc[0])
        cons = res_s[res_s.consistent]
        line(f"Step {step} (~{n_img} images per subsample): {len(cons)} of "
             f"{len(res_s)} consistent pairs")
        for _, f in cons.iterrows():
            win = f.A if f.A_better_significant > 0 else f.B
            lose = f.B if win == f.A else f.A
            line(f"    {win} > {lose} ({max(f.A_better_significant, f.B_better_significant)}"
                 f"/{f.seeds} seeds)")
        sens.append(res_s.assign(step=step)[["step", "A", "B", "seeds_significant",
                                             "A_better_significant",
                                             "B_better_significant", "consistent"]])
    if sens:
        save_table(pd.concat(sens), "mcnemar_sensitivity")

    line("\n=== Correlations (exploratory) ===")
    cor = correlations(df)
    for _, f in cor.iterrows():
        line(f"{f.analysis}: rho = {f.spearman_rho}, p = {f.p} (n = {f.n})")
    save_table(cor, "correlations")

    if "T" in df:
        line("\n=== Calibration (temperature scaling fitted on validation) ===")
        cal = df.groupby("model")[["ece", "ece_ts", "nll", "nll_ts", "T"]].mean().loc[order]
        cal.index = [name(m) for m in cal.index]
        save_table(cal.round(4), "calibration", index=True)
        for m, f in cal.iterrows():
            line(f"{m:<16} ECE {f.ece:.4f} -> {f.ece_ts:.4f}   T = {f['T']:.2f}")

    # ---------------------------------------------------------------- Ablations
    print("\nGenerating figures...")
    for folder, tag, text in [
            ("ablation_30_epochs", "30", "30 epochs, no early stopping"),
            ("protocol_50_epochs", "50", "50-epoch learning-rate schedule")]:
        others = load_runs(folder, folders)
        if not others:
            continue
        dfo, _ = main_analysis(others, k, long_names)
        t, w = compare_pairs(df, dfo, tag)
        if t is None:
            continue
        save_table(t.round(3), folder)
        line(f"\n=== {folder}: paired comparison with the 15-epoch protocol ===")
        line(f"Pairs: {len(t)}; mean F1 difference = {t.diff_f1.mean():+.2f} points "
             f"(median {t.diff_f1.median():+.2f})")
        if w is not None:
            line(f"Wilcoxon signed-rank: W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
        fig_pairs(t, tag, text, folder)

    # ---------------------------------------------------------------- Figures
    present = [m for m in MODELS if m in set(df.model)]
    fig_test_results(df, order)
    fig_cd(fn)
    fig_confusion(runs, order, short)
    fig_f1_per_class(pc, order, long_names, short)
    fig_efficiency(df, eff, order)
    fig_curves(runs, present)
    fig_search(trials, present)
    fig_calibration(df, order)
    fig_val_test(df)

    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "tables" / "statistical_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Statistical analysis and figures.")
    p.add_argument("--results", default=None, help=f"results folder (default: {RESULTS_DIR})")
    p.add_argument("--output", default=None, help=f"output folder (default: {OUTPUT_DIR})")
    a = p.parse_args()
    if a.results:
        RESULTS_DIR = Path(a.results).resolve()
    if a.output:
        OUTPUT_DIR = Path(a.output).resolve()
    F.FIGURE_DIR = OUTPUT_DIR / "figures"
    main()

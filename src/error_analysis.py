# -*- coding: utf-8 -*-
"""
AddSubLab-8: error analysis (Section VI of the article).

Part A. Per-image error table and error maps (which test frames each model
        misclassifies, in temporal order).
Part B. Mosaics of the most frequently misclassified test frames and training
        references of the confused classes.
Part C. Grad-CAM of the eight models (seed 42) on misclassified frames.
Part D. The longest segment of test frames that one model (ViT-B/16 by default)
        misclassifies in every seed: mosaic and Grad-CAM.

Requires the dataset (for the images) and, for Grad-CAM, the seed-42 weights:
either the weights/ folder of the Zenodo record (<model>_seed42.pt) or the
training/<model>/seed_42/best.pt files written by train.py.

Usage:
    python src/error_analysis.py --dataset path/to/AddSubLab-8
        [--results results] [--weights weights] [--output analysis] [--no-gradcam]

Outputs:
    <output>/figures/        figNN_<name>.pdf / .png (IEEE Access format)
    <output>/error_tables/   CSV tables; for each image figure, a
                             <figure>_content.csv file lists the frame shown in
                             each panel (a), (b), ...
"""

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import figure_style as F

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_DIR / "AddSubLab-8"
RESULTS_DIR = REPO_DIR / "results"
WEIGHTS_DIR = REPO_DIR / "weights"
OUTPUT_DIR = REPO_DIR / "analysis"

# Classes analyzed in detail (the two with most errors).
FOCUS_CLASSES = ["mini_metal_lathe", "mini_metal_horizontal_milling_machine"]

N_MOSAIC = 12                 # most frequently misclassified frames (2 rows of 6)
MOSAIC_COLS = 6
N_REFERENCE = 7               # training images per row
N_GRADCAM_ERRORS = 4          # rows of misclassified frames in Grad-CAM
N_GRADCAM_CORRECT = 1         # rows of frames classified correctly by all runs
MIN_SEPARATION = 10           # minimum distance (in frames) between selected frames
                              # of a class, to avoid near-identical frames;
                              # 0 = no restriction
GRADCAM_SEED = 42
RUN_GRADCAM = True

# Part D: segment that one model always misclassifies.
SEGMENT_MODEL = "vit_base"
SEGMENT_CLASS = "mini_metal_lathe"
SEGMENT_TOLERANCE = 2         # isolated correct frames that do not break a segment
SEGMENT_N_FAILED = 12
SEGMENT_N_CORRECT = 6
SEGMENT_N_GRADCAM = 4
SEGMENT_GRADCAM_MODELS = ["vit_base", "xception", "maxvit_tiny", "swin_tiny"]
SEGMENT_GRADCAM_WIDTH_MM = 88

# Model key -> (display name, timm name, Grad-CAM target layer).
# In ViT the head only uses the [CLS] token; therefore the input of the last
# block's attention (norm1) is used, where the patch tokens still influence
# the output.
MODELS = {
    "resnet50":        ("ResNet-50",       "resnet50.a1_in1k",                     "layer4"),
    "densenet121":     ("DenseNet-121",    "densenet121.tv_in1k",                  "features"),
    "xception":        ("Xception",        "legacy_xception.tf_in1k",              "act4"),
    "efficientnet_b0": ("EfficientNet-B0", "efficientnet_b0.ra_in1k",              "bn2"),
    "convnext_tiny":   ("ConvNeXt-T",      "convnext_tiny.fb_in1k",                "stages.3"),
    "maxvit_tiny":     ("MaxViT-T",        "maxvit_tiny_tf_224.in1k",              "stages.3"),
    "swin_tiny":       ("Swin-T",          "swin_tiny_patch4_window7_224.ms_in1k", "layers.3"),
    "vit_base":        ("ViT-B/16",        "vit_base_patch16_224.augreg_in1k",     "blocks.11.norm1"),
}

ANNOT_PT = 6                  # font size of the annotations below each image
C_OK, C_ERR = "#1f5fbf", "#c0282d"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


# =============================================================================
# UTILITIES
# =============================================================================

def natural_key(text):
    """'img_9' before 'img_10' (numeric frame order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def load_classes():
    """{class_folder: long name}, {class_folder: short name}, folders in label order."""
    with open(RESULTS_DIR / "classes.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["class_id"]))
    names = {r["class_folder"]: r["class_name"] for r in rows}
    short = {r["class_folder"]: r["abbreviation"] for r in rows}
    return names, short, [r["class_folder"] for r in rows]


def thumbnail(path, height=200):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.resize((round(im.width * height / im.height), height), Image.BILINEAR)


def image_aspect(path):
    with Image.open(path) as im:
        return im.height / im.width


def table_dir():
    d = OUTPUT_DIR / "error_tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_listing(rows, key):
    pd.DataFrame(rows).to_csv(table_dir() / f"{key}_content.csv", index=False,
                              encoding="utf-8")


def grid(rows, cols, width_mm, aspect, text_mm=6.0, header_mm=0.0, gap_mm=1.2,
         margin_mm=0.8):
    """Image grid with a text band below each row.

    Returns (fig, axs) with axs[row][col]. If the height exceeds 216 mm, the
    cells are shrunk and the grid is centered, keeping the figure width.
    """
    w = (width_mm - 2 * margin_mm - (cols - 1) * gap_mm) / cols
    h = w * aspect
    height = 2 * margin_mm + header_mm + rows * (h + text_mm)
    if height > 216:
        h = (216 - 2 * margin_mm - header_mm - rows * text_mm) / rows
        w = h / aspect
        height = 216
    x0 = (width_mm - cols * w - (cols - 1) * gap_mm) / 2
    fig = plt.figure(figsize=(width_mm * F.MM, height * F.MM))
    axs = []
    for i in range(rows):
        y = margin_mm + header_mm + i * (h + text_mm)
        row = []
        for j in range(cols):
            x = x0 + j * (w + gap_mm)
            ax = fig.add_axes([x / width_mm, 1 - (y + h) / height, w / width_mm,
                               h / height])
            ax.set_axis_off()
            row.append(ax)
        axs.append(row)
    return fig, axs


def text_below(ax, lines, colors=None, fonts=None, sizes=None):
    """Writes one or more centered lines below an image."""
    y = -0.015
    for k, t in enumerate(lines):
        if not t:
            continue
        c = (colors or [None] * len(lines))[k] or F.C_TXT
        fam = (fonts or [None] * len(lines))[k] or F.FONT
        size = (sizes or [None] * len(lines))[k] or ANNOT_PT
        ax.text(0.5, y, t, transform=ax.transAxes, ha="center", va="top",
                fontsize=size, color=c, fontfamily=fam, linespacing=0.95)
        y -= (size * (1 + t.count("\n")) * 1.15 / 72) / (ax.get_position().height *
                                                        ax.figure.get_figheight())


def panel(k, text=""):
    return f"({LETTERS[k]}) {text}".strip() if k < len(LETTERS) else text


def wrap(text, width=15):
    return "\n".join(textwrap.wrap(text, width)) if len(text) > width else text


# =============================================================================
# PART A: PER-IMAGE ERRORS
# =============================================================================

def load_predictions():
    rows = []
    for path in sorted((RESULTS_DIR / "training").glob("*/seed_*/test_predictions.csv")):
        model = path.parent.parent.name
        seed = int(path.parent.name.split("_")[1])
        p = pd.read_csv(path, usecols=["file", "true", "predicted", "correct"])
        p["model"], p["seed"] = model, seed
        rows.append(p)
    if not rows:
        raise SystemExit(f"No predictions found in {RESULTS_DIR / 'training'}.")
    return pd.concat(rows, ignore_index=True)


def errors_per_image(pred, names):
    err = pred[pred.correct == 0]
    table = pred.groupby(["file", "true"]).correct.agg(
        runs="size", hits="sum").reset_index()
    table["errors"] = table.runs - table.hits
    by_model = (err.groupby(["file", "model"]).size().unstack(fill_value=0)
                .reindex(columns=[m for m in MODELS if m in set(pred.model)],
                         fill_value=0))
    by_model.columns = [f"errors_{MODELS[m][0]}" for m in by_model.columns]
    freq = (err.groupby("file").predicted
            .agg(lambda s: s.value_counts().index[0]).rename("most_frequent_prediction"))
    table = (table.merge(by_model, left_on="file", right_index=True, how="left")
             .merge(freq, left_on="file", right_index=True, how="left"))
    table = table.fillna({c: 0 for c in by_model.columns})
    table["true_class"] = table.true.map(names)
    # Sorting: most errors first; ties in temporal (natural file-name) order.
    errors, files = table.errors.to_numpy(), table.file.to_numpy()
    order = sorted(range(len(table)), key=lambda i: (-errors[i], natural_key(files[i])))
    return table.iloc[order].reset_index(drop=True)


def summary_per_class(table, n_runs, names):
    rows = []
    for c, g in table.groupby("true"):
        rows.append({
            "class": names[c], "images": len(g),
            "images_with_some_error": int((g.errors > 0).sum()),
            "errors_in_half_or_more_runs": int((g.errors >= n_runs / 2).sum()),
            "errors_in_all_runs": int((g.errors == n_runs).sum()),
            "error_rate_%": round(100 * g.errors.sum() / g.runs.sum(), 2),
        })
    return pd.DataFrame(rows).sort_values("error_rate_%", ascending=False)


def fig_error_map(pred, cls, names):
    """Row = model, column = test frame (temporal order), color = failing seeds."""
    p = pred[pred.true == cls]
    models = [m for m in MODELS if m in set(p.model)]
    files = sorted(p.file.unique(), key=natural_key)
    mat = (p.assign(fail=1 - p.correct).groupby(["model", "file"]).fail.sum()
           .unstack(fill_value=0).reindex(index=models, columns=files))
    n_seeds = int(p.groupby("model").seed.nunique().max())
    palette = ["#f0efec"] + F.SEQUENTIAL[1:]      # light gray = no seed fails
    cmap = matplotlib.colors.ListedColormap(
        [palette[round(i * (len(palette) - 1) / n_seeds)] for i in range(n_seeds + 1)])
    fig = plt.figure(figsize=(F.PAGE_WIDTH, (14 + 4.2 * len(models)) * F.MM))
    ax = fig.add_axes([0.115, 0.2, 0.775, 0.77])
    cax = fig.add_axes([0.9, 0.2, 0.012, 0.77])
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=n_seeds + 0.5,
                   interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(len(models)), [MODELS[m][0] for m in models])
    ax.tick_params(axis="y", length=0)
    step = max(1, round(len(files) / 10, -1)) if len(files) > 20 else 1
    ax.set_xticks(np.arange(0, len(files), step))
    ax.set_xlabel(f"Test frames of {names[cls]} in temporal order (frame index)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, ticks=range(n_seeds + 1))
    cb.set_label("Seeds that\nmisclassify the frame", labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    F.save(fig, f"error_map_{cls}")


# =============================================================================
# PART B: MOSAICS
# =============================================================================

def frame_positions(table, cls):
    """Position of each frame of the class in natural (temporal) order."""
    files = sorted(table[table.true == cls].file, key=natural_key)
    return {a: i for i, a in enumerate(files)}


def pick_separated(t, n, pos, sep=None):
    """Takes rows of t in their order, skipping those closer than `sep` frames
    to an already selected one. If fewer than n, completes without the rule."""
    sep = MIN_SEPARATION if sep is None else sep
    chosen = []
    for a in t.file:
        if len(chosen) == n:
            break
        if all(abs(pos[a] - pos[b]) >= sep for b in chosen):
            chosen.append(a)
    for a in t.file:
        if len(chosen) == n:
            break
        if a not in chosen:
            chosen.append(a)
    chosen = set(chosen)
    return t[t.file.isin(chosen)]               # keeps the original order of t


def fig_error_mosaic(table, cls, names, short, n_runs):
    t = table[(table.true == cls) & (table.errors > 0)]
    if t.empty:
        return
    pos = frame_positions(table, cls)
    t = pick_separated(t, N_MOSAIC, pos)
    rows = math.ceil(len(t) / MOSAIC_COLS)
    asp = image_aspect(DATASET_DIR / t.file.iloc[0])
    fig, axs = grid(rows, MOSAIC_COLS, 181, asp, text_mm=9.5)
    listing = []
    for k, (_, f) in enumerate(t.iterrows()):
        ax = axs[k // MOSAIC_COLS][k % MOSAIC_COLS]
        ax.imshow(thumbnail(DATASET_DIR / f.file, 400))
        text_below(ax, [panel(k, f"{int(f.errors)}/{n_runs} wrong"),
                        f"→ {short.get(f.most_frequent_prediction, '')}"],
                   colors=[None, C_ERR], fonts=[F.SUBFIG_FONT, None], sizes=[8, ANNOT_PT])
        listing.append({"panel": f"({LETTERS[k]})", "file": f.file,
                        "frame_index": pos[f.file], "errors": int(f.errors),
                        "runs": n_runs,
                        "most_frequent_prediction": names.get(f.most_frequent_prediction)})
    F.save(fig, f"error_mosaic_{cls}")
    save_listing(listing, f"error_mosaic_{cls}")


def fig_training_reference(table, cls, names):
    """Training examples of the true class and of the classes it is confused with."""
    conf = (table[(table.true == cls) & (table.errors > 0)].most_frequent_prediction
            .value_counts().head(3).index.tolist())
    groups = [cls] + [c for c in conf if c != cls]
    rng = np.random.default_rng(0)
    first = next((DATASET_DIR / "train" / cls).iterdir())
    fig, axs = grid(len(groups), N_REFERENCE, 181, image_aspect(first), text_mm=5.0)
    listing = []
    for i, c in enumerate(groups):
        imgs = sorted((DATASET_DIR / "train" / c).iterdir(), key=lambda p: natural_key(p.name))
        sel = [imgs[j] for j in np.sort(rng.choice(len(imgs), min(N_REFERENCE, len(imgs)),
                                                   replace=False))]
        for ax, path in zip(axs[i], sel):
            ax.imshow(thumbnail(path, 400))
            listing.append({"panel": f"({LETTERS[i]})", "class": names[c],
                            "file": path.relative_to(DATASET_DIR).as_posix()})
        role = "true class" if c == cls else "confused class"
        p = axs[i][0].get_position()
        fig.text(0.5, p.y0 - 0.006, panel(i, f"Training images of {names[c]} ({role})"),
                 ha="center", va="top", fontsize=8, fontfamily=F.SUBFIG_FONT)
    F.save(fig, f"training_reference_{cls}")
    save_listing(listing, f"training_reference_{cls}")


# =============================================================================
# PART C: GRAD-CAM
# =============================================================================

def checkpoint_path(model, seed):
    for p in (WEIGHTS_DIR / f"{model}_seed{seed}.pt",
              RESULTS_DIR / "training" / model / f"seed_{seed}" / "best.pt"):
        if p.exists():
            return p
    return None


def compute_gradcam(imgs, models, folders, seed=None):
    """Returns {model: [(rgb_image, map, predicted_class), ...]} for each image."""
    import torch
    import timm

    seed = GRADCAM_SEED if seed is None else seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(RESULTS_DIR / "model_configs.json", encoding="utf-8") as f:
        cfgs = json.load(f)
    interp = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}
    out = {}
    for m in models:
        ck_path = checkpoint_path(m, seed)
        if ck_path is None:
            print(f"    {m}: no seed-{seed} checkpoint found; skipped")
            continue
        disp_name, timm_name, layer = MODELS[m]
        cfg = cfgs[m]
        model = timm.create_model(timm_name, pretrained=False, num_classes=len(folders))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        prefix = getattr(model, "num_prefix_tokens", 0)
        capt = {}

        def hook(_, __, output):
            output.register_hook(lambda g: capt.__setitem__("g", g))
            capt["a"] = output
        h = model.get_submodule(layer).register_forward_hook(hook)
        _, height, width = cfg["input_size"]
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        res = []
        for a in imgs:
            with Image.open(DATASET_DIR / a) as im:
                vis = np.asarray(im.convert("RGB").resize(
                    (width, height), interp.get(cfg["interpolation"], Image.BICUBIC)))
            x = torch.from_numpy(vis.copy()).permute(2, 0, 1).float().div(255)
            x = ((x.unsqueeze(0) - mean) / std).to(device).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            pred = int(logits.argmax(1))
            logits[0, pred].backward()
            A, G = capt["a"].detach(), capt["g"].detach()
            if A.dim() == 3:                                    # ViT: (B, tokens, C)
                A, G = A[:, prefix:], G[:, prefix:]
                side = int(math.isqrt(A.shape[1]))
                A = A.reshape(1, side, side, -1).permute(0, 3, 1, 2)
                G = G.reshape(1, side, side, -1).permute(0, 3, 1, 2)
            elif A.shape[-1] != A.shape[-2] and A.shape[1] == A.shape[2]:   # Swin: NHWC
                A, G = A.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
            cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(height, width),
                                                  mode="bilinear", align_corners=False)
            cam = cam[0, 0].cpu().numpy()
            cam = cam / cam.max() if cam.max() > 0 else cam
            res.append((vis, cam, folders[pred]))
        h.remove()
        out[m] = res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"    {disp_name} done")
    return out


def draw_gradcam(imgs, labels, cls, maps, short, width_mm, key, extra=None):
    """Rows = images; columns = input image + one map per model."""
    models = list(maps)
    asp = image_aspect(DATASET_DIR / imgs[0])
    fig, axs = grid(len(imgs), len(models) + 1, width_mm, asp, text_mm=7.5, header_mm=6.0)
    narrow = (width_mm / (len(models) + 1)) < 19
    for j, title in enumerate(["Input"] + [MODELS[m][0] for m in models]):
        axs[0][j].text(0.5, 1.02, wrap(title, 9) if narrow else title,
                       transform=axs[0][j].transAxes, ha="center", va="bottom",
                       fontsize=ANNOT_PT + 0.5, fontweight="bold")
    listing = []
    for i, (a, lab) in enumerate(zip(imgs, labels)):
        ax = axs[i][0]
        ax.imshow(thumbnail(DATASET_DIR / a, 400))
        text_below(ax, [panel(i), lab], fonts=[F.SUBFIG_FONT, None], sizes=[8, ANNOT_PT])
        row = {"panel": f"({LETTERS[i]})", "file": a, "annotation": lab}
        for j, m in enumerate(models, start=1):
            vis, cam, pred = maps[m][i]
            ax = axs[i][j]
            ax.imshow(vis, extent=(0, 1, asp, 0))
            ax.imshow(cam, cmap="jet", alpha=0.45, extent=(0, 1, asp, 0), vmin=0, vmax=1)
            ax.set_aspect("auto")
            ok = pred == cls
            text_below(ax, ["Correct" if ok else wrap(short[pred], 12 if narrow else 16)],
                       colors=[C_OK if ok else C_ERR])
            row[f"prediction_{MODELS[m][0]}"] = short[pred]
        if extra:
            row.update(extra[i])
        listing.append(row)
    F.save(fig, key)
    save_listing(listing, key)


def gradcam_figures(table, classes, folders, short, names):
    n_runs = int(table.runs.max())
    for cls in classes:
        t = table[table.true == cls]
        pos = frame_positions(table, cls)
        err = pick_separated(t[t.errors > 0], N_GRADCAM_ERRORS, pos)
        # Contrast: frames classified correctly by all runs; if there are none,
        # those with the fewest errors.
        rest = t[~t.file.isin(err.file)]
        minimum = rest.errors.min() if len(rest) else 0
        if minimum > 0:
            print(f"    {names[cls]}: no frame is classified correctly by all runs; "
                  f"the contrast uses frames with {int(minimum)} errors")
        cand = rest[rest.errors == minimum].sample(frac=1, random_state=0)
        ok = pick_separated(cand, N_GRADCAM_CORRECT, pos) if len(cand) else cand
        imgs = list(err.file) + list(ok.file)
        if not imgs:
            continue
        labels = [f"{int(t[t.file == a].errors.iloc[0])}/{n_runs} wrong" for a in imgs]
        maps = compute_gradcam(imgs, list(MODELS), folders)
        if maps:
            draw_gradcam(imgs, labels, cls, maps, short, 181, f"gradcam_{cls}")


# =============================================================================
# PART D: SEGMENT THAT ONE MODEL ALWAYS MISCLASSIFIES
# =============================================================================

def longest_segment(mask, tolerance):
    """[start, end] indices of the longest run of True, allowing short gaps."""
    best, start, last, gap = (0, -1, -1), None, None, 0
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if gap > tolerance:
                if last - start + 1 > best[0]:
                    best = (last - start + 1, start, last)
                start, gap = None, 0
    if start is not None and last - start + 1 > best[0]:
        best = (last - start + 1, start, last)
    return best[1], best[2]


def segment_analysis(pred, folders, short, run_gradcam):
    model, cls = SEGMENT_MODEL, SEGMENT_CLASS
    p = pred[pred.true == cls]
    files = sorted(p.file.unique(), key=natural_key)
    pos = {a: i for i, a in enumerate(files)}
    n_seeds = p[p.model == model].seed.nunique()
    fails = (p[p.model == model].assign(f=1 - p.correct).groupby("file").f.sum()
             .reindex(files))
    disp = MODELS[model][0]

    i0, i1 = longest_segment((fails == n_seeds).to_numpy(), SEGMENT_TOLERANCE)
    if i0 is None or i0 < 0:
        print(f"  {disp} does not misclassify any segment of {cls} in every seed.")
        return
    j0, j1 = longest_segment((fails == 0).to_numpy(), SEGMENT_TOLERANCE)
    segment = files[i0:i1 + 1]
    print(f"  Segment misclassified by {disp} in all {n_seeds} seeds: {len(segment)} "
          f"frames, positions {i0}-{i1} of {len(files)}")

    ps = p[(p.model == model) & p.file.isin(segment)]
    dist = (ps.predicted.map(short).value_counts(normalize=True) * 100).round(1)
    print(f"  Predictions of {disp} in the segment (%):\n{dist.to_string()}")

    p = p.assign(in_segment=p.file.isin(segment))
    t = (p.assign(error=100 * (1 - p.correct))
         .groupby(["model", "in_segment"]).error.mean().unstack()
         .rename(columns={True: "error_in_segment_%", False: "error_rest_of_class_%"}))
    t = t.reindex([m for m in MODELS if m in t.index])
    t.index = [MODELS[m][0] for m in t.index]
    t = t[["error_in_segment_%", "error_rest_of_class_%"]].round(1)
    print(f"  Error of each model in the segment and in the rest of the class:\n"
          f"{t.to_string()}")
    base = f"segment_{model}_{cls}"
    t.to_csv(table_dir() / f"{base}.csv", encoding="utf-8")
    dist.rename("percent").to_csv(table_dir() / f"{base}_predictions.csv", encoding="utf-8")

    # Mosaic: misclassified segment (top rows) and a correct segment (last row).
    sel_f = [segment[k] for k in np.linspace(0, len(segment) - 1,
                                             SEGMENT_N_FAILED).astype(int)]
    sel_c = []
    if j0 is not None and j0 >= 0:
        ok = files[j0:j1 + 1]
        sel_c = [ok[k] for k in np.linspace(0, len(ok) - 1,
                                            min(SEGMENT_N_CORRECT, len(ok))).astype(int)]
    cols = MOSAIC_COLS
    rows_f = int(np.ceil(len(sel_f) / cols))
    rows = rows_f + (int(np.ceil(len(sel_c) / cols)) if sel_c else 0)
    fig, axs = grid(rows, cols, 181, image_aspect(DATASET_DIR / sel_f[0]), text_mm=9.5)
    pm = p[p.model == model]
    listing = []
    for k, a in enumerate(sel_f + sel_c):
        correct = k >= len(sel_f)
        kk = k if not correct else rows_f * cols + (k - len(sel_f))
        ax = axs[kk // cols][kk % cols]
        ax.imshow(thumbnail(DATASET_DIR / a, 400))
        pr = pm[pm.file == a].predicted.map(short).value_counts().index[0]
        text_below(ax, [panel(k, f"Frame {pos[a]}"),
                        "Correct" if correct else f"→ {pr}"],
                   colors=[None, C_OK if correct else C_ERR],
                   fonts=[F.SUBFIG_FONT, None], sizes=[8, ANNOT_PT])
        listing.append({"panel": f"({LETTERS[k]})", "file": a, "frame_index": pos[a],
                        "group": "always correct" if correct else "always wrong",
                        f"prediction_{disp}": pr})
    F.save(fig, f"{base}_mosaic")
    save_listing(listing, f"{base}_mosaic")

    if not run_gradcam:
        return
    sel_g = [segment[k] for k in np.linspace(0, len(segment) - 1,
                                             SEGMENT_N_GRADCAM).astype(int)]
    if sel_c:
        sel_g.append(sel_c[len(sel_c) // 2])
    labels = [f"Frame {pos[a]}" for a in sel_g]
    extra = [{"group": "always wrong" if a in segment else "always correct"}
             for a in sel_g]
    maps = compute_gradcam(sel_g, SEGMENT_GRADCAM_MODELS, folders)
    if maps:
        draw_gradcam(sel_g, labels, cls, maps, short, SEGMENT_GRADCAM_WIDTH_MM,
                     f"{base}_gradcam", extra)


# =============================================================================
# MAIN
# =============================================================================

def main(run_gradcam=True):
    F.style()
    plt.rcParams["axes.grid"] = False
    print(f"Figures in: {F.FIGURE_DIR}\nTables in: {OUTPUT_DIR / 'error_tables'}")
    names, short, folders = load_classes()
    pred = load_predictions()
    n_runs = int(pred.groupby("file").size().max())
    table = errors_per_image(pred, names)
    table.to_csv(table_dir() / "errors_per_image.csv", index=False, encoding="utf-8")
    summ = summary_per_class(table, n_runs, names)
    summ.to_csv(table_dir() / "errors_per_class.csv", index=False, encoding="utf-8")
    print(summ.to_string(index=False))

    classes = FOCUS_CLASSES or list(table[table.errors > 0].groupby("true").errors
                                    .sum().sort_values(ascending=False).index)
    print("Part A: error maps")
    for c in classes:
        fig_error_map(pred, c, names)
    print("Part B: mosaics")
    for c in classes:
        fig_error_mosaic(table, c, names, short, n_runs)
        fig_training_reference(table, c, names)
    if run_gradcam and RUN_GRADCAM:
        print("Part C: Grad-CAM")
        gradcam_figures(table, classes, folders, short, names)
    print("Part D: segment analysis")
    segment_analysis(pred, folders, short, run_gradcam and RUN_GRADCAM)
    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Error analysis and Grad-CAM figures.")
    ap.add_argument("--dataset", default=None, help=f"dataset folder (default: {DATASET_DIR})")
    ap.add_argument("--results", default=None, help=f"results folder (default: {RESULTS_DIR})")
    ap.add_argument("--weights", default=None,
                    help=f"folder with <model>_seed42.pt files (default: {WEIGHTS_DIR})")
    ap.add_argument("--output", default=None, help=f"output folder (default: {OUTPUT_DIR})")
    ap.add_argument("--no-gradcam", action="store_true", help="skip the Grad-CAM figures")
    a = ap.parse_args()
    if a.dataset:
        DATASET_DIR = Path(a.dataset).resolve()
    if a.results:
        RESULTS_DIR = Path(a.results).resolve()
    if a.weights:
        WEIGHTS_DIR = Path(a.weights).resolve()
    if a.output:
        OUTPUT_DIR = Path(a.output).resolve()
    F.FIGURE_DIR = OUTPUT_DIR / "figures"
    main(run_gradcam=not a.no_gradcam)

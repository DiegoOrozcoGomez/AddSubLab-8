# -*- coding: utf-8 -*-
"""
AddSubLab-8 - background / foreground experiments requested in the review.

Do the models recognize the machine, or the place where it stands? The machine
of every image is segmented with IS-Net (isnet-general-use, ONNX; U2-Net can be
chosen with --segmenter u2net, but it misses most of the open-frame printers), and:

  1. evaluate   The models of the main study (10 seeds x 8 models, weights
                training/<model>/seed_<s>/best.pt) are evaluated on four versions of the test set:
                  original     the test images (sanity check: reproduces Table IV)
                  machine_only the background is replaced by uniform gray
                  background   the machine (plus a 5% margin) is replaced by uniform
                               gray (only the place is visible; chance level = 12.5%)
                  background_wide  the same with a 10% margin (sensitivity to the
                               precision of the mask borders)
                  swapped      the machine is pasted on the background of an
                               image of ANOTHER class (whose machine was removed);
                               we count whether the prediction follows the
                               machine or the background.
  2. background A "background-only" classifier is trained on images whose machine
                was removed (ResNet-50 and MaxViT-T, 3 seeds, same protocol and
                hyperparameters as the main study) and tested on background-only
                test images. Accuracy far above 12.5% would mean that the place
                alone identifies the class even in the new environment.

The masks are computed once (step "masks") and saved; check the quality
sheets in results/masking/qc/ before trusting the results.

Requirements (once, in the cnn_transformer environment):
    pip install onnxruntime
and the file isnet-general-use.onnx in C:\\Users\\<user>\\.u2net\\ (download it with the
browser from https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx).
Use --model to give another path.

Usage (Anaconda Prompt):
    conda activate cnn_transformer
    cd C:\\proyectos\\AddSubLab-8
    python src\\masking_experiments.py all --dataset "C:\\proyectos\\34_Add_Sub_Man_Mac\\dataset_publico\\AddSubLab-8"
Steps can also be run separately: masks | evaluate | background | status.
Each step resumes if interrupted. Results: results/masking/isnet/
Estimated time: masks ~1-2 h (CPU), evaluate ~20 min, background ~2 h (GPU).
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

log = T.log
GRAY = 128
TRAIN_STEP = 4          # every 4th training image (frames are ~3 apart; 6,400 images)
VAL_STEP = 4            # every 4th validation image (1,600 images)
BG_MODELS = ["resnet50", "maxvit_tiny"]
BG_SEEDS = [42, 1234, 2025]
EVAL_SEEDS = [42, 1234, 2025, 7, 31415, 123, 777, 2718, 99, 4242]   # models evaluated
UNSHIFTED = ["delta_3d_printer", "resin_3d_printer", "bench_drill"]
MASK_SIDE = 320          # resolution at which the masks are stored
SEGMENTER = "isnet"
SEGMENTERS = {           # file, input side, mean, std  (same preprocessing as rembg)
    "isnet": ("isnet-general-use.onnx", 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "u2net": ("u2net.onnx", 320, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def out_dir():
    return T.OUTPUT_DIR / "masking" / SEGMENTER


# =============================================================================
# 1. MASKS
# =============================================================================

def seg_session(path):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is not installed. Run:  pip install onnxruntime")
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path.name} not found at {path}. Download it (see the header of "
                         f"this script) or give its location with --model <path>.")
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    s = ort.InferenceSession(str(path), providers=prov)
    log.info(f"  {SEGMENTER} loaded ({s.get_providers()[0]})")
    return s


def seg_prob(sess, img):
    """Foreground map in [0, 1], resized to MASK_SIDE (same preprocessing as rembg)."""
    _, side, mean, std = SEGMENTERS[SEGMENTER]
    x = np.asarray(img.convert("RGB").resize((side, side), Image.LANCZOS), dtype=np.float32)
    x = x / max(np.max(x), 1e-6)
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    if out.shape[0] != MASK_SIDE:
        out = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize(
            (MASK_SIDE, MASK_SIDE), Image.BILINEAR), dtype=np.float32) / 255.0
    return out


def clean(prob):
    """Binary mask: threshold 0.5, keep the large components, fill holes."""
    m = prob > 0.5
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.where(sizes >= 0.15 * sizes.max())[0] + 1
        m = np.isin(lab, keep)
    return ndimage.binary_fill_holes(m)


def split_indices(split, n):
    step = {"train": TRAIN_STEP, "validation": VAL_STEP}.get(split, 1)
    return np.arange(0, n, step)


def step_masks(model_path):
    T.title(f"MASKING - machine segmentation with {SEGMENTER}")
    data, classes, _ = T.list_dataset()
    folder = out_dir() / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    sess = None
    for split in ("test", "validation", "train"):
        files, labels = data[split]
        idx = split_indices(split, len(files))
        dest = folder / f"{split}_masks.npy"
        if dest.exists():
            log.info(f"  {split}: masks already computed")
            continue
        sess = sess or seg_session(model_path)
        part = folder / f"{split}_partial.npy"
        masks = np.load(part) if part.exists() else np.full((len(idx), MASK_SIDE, MASK_SIDE), 255, np.uint8)
        done_path = folder / f"{split}_partial_done.txt"
        start = int(done_path.read_text()) if done_path.exists() else 0
        log.info(f"  {split}: {len(idx)} images (from {start})")
        t0 = time.time()
        for k in range(start, len(idx)):
            with Image.open(T.DATASET_DIR / files[idx[k]]) as im:
                masks[k] = clean(seg_prob(sess, im)).astype(np.uint8)
            if (k + 1) % 500 == 0 or k + 1 == len(idx):
                np.save(part, masks)
                done_path.write_text(str(k + 1))
                rate = (k + 1 - start) / (time.time() - t0)
                log.info(f"    {k + 1}/{len(idx)}  ({rate:.1f} img/s)")
        np.save(dest, masks)
        np.save(folder / f"{split}_index.npy", idx)
        part.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
    quality_report(data, classes)


def quality_report(data, classes):
    folder = out_dir() / "masks"
    qc = out_dir() / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "validation", "test"):
        m = np.load(folder / f"{split}_masks.npy")
        idx = np.load(folder / f"{split}_index.npy")
        labels = data[split][1][idx]
        area = m.reshape(len(m), -1).mean(1)
        for c in range(len(classes)):
            a = area[labels == c]
            rows.append(dict(split=split, class_folder=classes[c], n=len(a),
                             area_median=round(float(np.median(a)), 3),
                             area_p05=round(float(np.percentile(a, 5)), 3),
                             area_p95=round(float(np.percentile(a, 95)), 3),
                             empty_or_tiny=int((a < 0.03).sum()), almost_full=int((a > 0.85).sum())))
        if split == "test":   # contact sheet: 12 images per class with the mask outline
            files = data[split][0]
            for c in range(len(classes)):
                sel = np.where(labels == c)[0]
                sel = sel[np.linspace(0, len(sel) - 1, 12).astype(int)]
                tiles = []
                for s in sel:
                    with Image.open(T.DATASET_DIR / files[idx[s]]) as im:
                        a = np.asarray(im.convert("RGB").resize((180, 320)), dtype=np.float32)
                    mk = np.asarray(Image.fromarray(m[s] * 255).resize((180, 320))) > 127
                    a[~mk] *= 0.3                       # background darkened
                    tiles.append(a.astype(np.uint8))
                sheet = np.concatenate(tiles, axis=1)
                Image.fromarray(sheet).save(qc / f"test_{c + 1}_{classes[c]}.jpg", quality=85)
    T.save_csv(out_dir() / "mask_quality.csv", rows)
    log.info(f"  Quality sheets in {qc} (machine bright, background dark).")


# =============================================================================
# Image versions
# =============================================================================

def resized_masks(split, side, dilate_frac):
    m = np.load(out_dir() / "masks" / f"{split}_masks.npy")
    out = np.empty((len(m), side, side), bool)
    r = max(1, int(round(dilate_frac * side)))
    st = ndimage.generate_binary_structure(2, 1)
    for i in range(len(m)):
        x = np.asarray(Image.fromarray(m[i] * 255).resize((side, side), Image.BILINEAR)) > 127
        out[i] = ndimage.binary_dilation(x, st, iterations=r) if dilate_frac > 0 else x
    return out


def versions(imgs, split, side, rng):
    """Returns dict of uint8 arrays (N, side, side, 3) for the subset of the split."""
    tight = resized_masks(split, side, 0.01)      # machine with a thin margin
    wide = resized_masks(split, side, 0.05)       # machine plus 5% margin (removed)
    v = {}
    v["machine_only"] = np.where(tight[..., None], imgs, np.uint8(GRAY))
    v["background"] = np.where(wide[..., None], np.uint8(GRAY), imgs)
    if split == "test":                           # sensitivity: 10% margin
        wider = resized_masks(split, side, 0.10)
        v["background_wide"] = np.where(wider[..., None], np.uint8(GRAY), imgs)
    return v, tight


def swap_partners(labels, rng):
    n = len(labels)
    partner = np.empty(n, int)
    for i in range(n):
        cand = np.where(labels != labels[i])[0]
        partner[i] = rng.choice(cand)
    return partner


# =============================================================================
# 2. EVALUATION OF THE TRAINED MODELS
# =============================================================================

def f1_parts(y, pred, classes):
    f = T.f1_score(y, pred, average=None, labels=list(range(len(classes))), zero_division=0)
    sh = [i for i, c in enumerate(classes) if c not in UNSHIFTED]
    un = [i for i, c in enumerate(classes) if c in UNSHIFTED]
    return float(f.mean()), float(f[sh].mean()), float(f[un].mean()), f


WEIGHTS_DIR = None   # folder that contains training/<model>/seed_<s>/best.pt


def step_evaluate():
    T.title("MASKING - evaluation of the trained models on modified test images")
    device = T.get_device()
    T.setup_torch()
    data, classes, _ = T.list_dataset()
    configs = T.read_json(T.OUTPUT_DIR / "model_configs.json")
    res_path = out_dir() / "test_conditions.csv"
    rows = list(csv.DictReader(open(res_path, encoding="utf-8"))) if res_path.exists() else []
    done = {(r["model"], r["seed"]) for r in rows}
    labels_all = data["test"][1]
    partner = swap_partners(labels_all, np.random.default_rng(0))
    np.save(out_dir() / "swap_partners.npy", partner)
    per_image = out_dir() / "predictions"
    per_image.mkdir(parents=True, exist_ok=True)
    for key in T.MODELS:
        if key not in configs:
            continue
        cfg = configs[key]
        side = cfg["input_size"][1]
        imgs, y, files = T.CACHE.get(cfg, "test")
        v, tight = versions(imgs, "test", side, None)
        v["original"] = imgs
        v["swapped"] = np.where(tight[..., None], imgs, v["background"][partner])
        mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        for seed in EVAL_SEEDS:
            if (key, str(seed)) in done:
                continue
            root = WEIGHTS_DIR or T.OUTPUT_DIR
            options = [root / "training" / key / f"seed_{seed}" / "best.pt",
                       root / "entrenamientos" / key / f"semilla_{seed}" / "mejor.pt"]
            ck_path = next((c for c in options if c.exists()), None)
            if ck_path is None:
                log.info(f"  WARNING: weights of {key} seed {seed} not found in {root} - skipped")
                continue
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            state = ck
            if isinstance(ck, dict):
                for k in ("model", "modelo", "state_dict", "model_state_dict"):
                    if k in ck and isinstance(ck[k], dict):
                        state = ck[k]
                        break
            model = T.create_model(key, device, pretrained=False)
            model.load_state_dict(state)
            expected = None
            mpath = T.OUTPUT_DIR / "training" / key / f"seed_{seed}" / "metrics.json"
            if mpath.exists():
                expected = T.read_json(mpath)["test_metrics"]["f1_macro"]
            preds = {}
            for cond in ("original", "machine_only", "background", "background_wide", "swapped"):
                r = T.evaluate(model, v[cond], y, mean, std, device)
                preds[cond] = r["pred"]
                f, fsh, fun, _ = f1_parts(y, r["pred"], classes)
                row = dict(model=key, seed=str(seed), condition=cond,
                           accuracy=round(r["acc"], 5), f1_macro=round(f, 5),
                           f1_shifted=round(fsh, 5), f1_unshifted=round(fun, 5))
                if cond == "swapped":
                    row["follows_machine"] = round(float((r["pred"] == y).mean()), 5)
                    row["follows_background"] = round(float((r["pred"] == y[partner]).mean()), 5)
                rows.append(row)
                if cond == "original" and expected is not None and abs(f - expected) > 0.002:
                    log.info(f"  WARNING: {key} seed {seed}: F1 on the original test images "
                             f"({f:.4f}) differs from the one reported ({expected:.4f}). "
                             f"These weights may not belong to the runs of the article.")
                log.info(f"  {key} seed {seed} {cond:13s} acc {r['acc']:.4f} f1 {f:.4f}"
                         + (f" | follows machine {row['follows_machine']:.3f}, background "
                            f"{row['follows_background']:.3f}" if cond == "swapped" else ""))
            np.savez(per_image / f"{key}_seed_{seed}.npz", **preds)
            cols = ["model", "seed", "condition", "accuracy", "f1_macro", "f1_shifted",
                    "f1_unshifted", "follows_machine", "follows_background"]
            with open(res_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # example images of each version (first model of each resolution only)
        ex = out_dir() / "qc" / f"versions_{side}.jpg"
        if not ex.exists():
            pick = np.linspace(0, len(y) - 1, 8).astype(int)
            grid = np.concatenate([np.concatenate([v[c][i] for i in pick], 1)
                                   for c in ("original", "machine_only", "background", "swapped")], 0)
            Image.fromarray(grid).save(ex, quality=85)
        del v, tight
        gc.collect()


# =============================================================================
# 3. BACKGROUND-ONLY CLASSIFIER
# =============================================================================

def step_background():
    T.title("MASKING - background-only classifier")
    device = T.get_device()
    T.setup_torch()
    data, classes, _ = T.list_dataset()
    configs = T.read_json(T.OUTPUT_DIR / "model_configs.json")
    original_get = T.CACHE.get
    base = out_dir() / "background_classifier"
    for key in BG_MODELS:
        cfg = configs[key]
        side = cfg["input_size"][1]
        sel = T.read_json(T.OUTPUT_DIR / "hyperparameter_search" / key / "best_hyperparameters.json")
        hp = {k: sel[k] for k in ("lr", "weight_decay", "label_smoothing")}
        masked = {}
        for split in ("train", "validation", "test"):
            imgs, y, files = original_get(cfg, split)
            idx = np.load(out_dir() / "masks" / f"{split}_index.npy")
            v, _ = versions(imgs[idx], split, side, None)
            masked[split] = (np.ascontiguousarray(v["background"]), y[idx], [files[i] for i in idx])
            if split == "test":
                masked["test_wide"] = (np.ascontiguousarray(v["background_wide"]), y[idx],
                                       [files[i] for i in idx])
            del v
        T.CACHE.get = lambda c, s: masked[s]
        try:
            for seed in BG_SEEDS:
                folder = base / key / f"seed_{seed}"
                if (folder / "metrics.json").exists():
                    continue
                label = f"[Background-only | {key} | seed {seed}]"
                res = T.train_run(key=key, cfg=cfg, hp=hp, seed=seed, max_epochs=T.MAX_EPOCHS,
                                  patience=T.PATIENCE, folder=folder, label=label,
                                  device=device, classes=classes)
                model = T.create_model(key, device, pretrained=False)
                model.load_state_dict(torch.load(folder / "best.pt", map_location=device,
                                                 weights_only=False)["model"])
                mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
                std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
                out = {}
                for split in ("validation", "test", "test_wide"):
                    imgs, y, files = masked[split]
                    r = T.evaluate(model, imgs, y, mean, std, device)
                    f, fsh, fun, fc = f1_parts(y, r["pred"], classes)
                    out[split] = dict(accuracy=r["acc"], f1_macro=f, f1_shifted=fsh,
                                      f1_unshifted=fun,
                                      f1_per_class=dict(zip(classes, map(float, fc))))
                    T.save_predictions(folder / f"{split}_predictions.csv", files, y,
                                       r["probs"], classes)
                T.save_json(folder / "metrics.json", dict(model=key, seed=seed, hyperparameters=hp,
                                                         best_epoch=res["best_epoch"],
                                                         epochs_trained=len(res["history"]),
                                                         train_images=len(masked["train"][1]),
                                                         **{f"{k}_metrics": v for k, v in out.items()}))
                (folder / "best.pt").unlink(missing_ok=True)   # not needed afterwards
                (folder / "last.pt").unlink(missing_ok=True)
                log.info(f"{label} TEST background-only acc {out['test']['accuracy']:.4f} "
                         f"f1 {out['test']['f1_macro']:.4f} (shifted {out['test']['f1_shifted']:.4f}, "
                         f"unshifted {out['test']['f1_unshifted']:.4f}); with 10% margin: acc "
                         f"{out['test_wide']['accuracy']:.4f}; chance = 0.125")
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            T.CACHE.get = original_get
        del masked
        gc.collect()


def step_status():
    o = out_dir()
    print(f"Results: {o}")
    for s in ("test", "validation", "train"):
        print(f"  masks {s:10s}: {'done' if (o / 'masks' / f'{s}_masks.npy').exists() else 'pending'}")
    n = 0
    if (o / "test_conditions.csv").exists():
        n = len({(r['model'], r['seed']) for r in csv.DictReader(open(o / 'test_conditions.csv'))})
    print(f"  evaluate        : {n}/{len(T.MODELS) * len(EVAL_SEEDS)} models")
    nb = sum((o / "background_classifier" / k / f"seed_{s}" / "metrics.json").exists()
             for k in BG_MODELS for s in BG_SEEDS)
    print(f"  background      : {nb}/{len(BG_MODELS) * len(BG_SEEDS)} runs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["masks", "evaluate", "background", "all", "status"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--weights", default=None,
                   help="results folder that contains training/<model>/seed_<s>/best.pt "
                        "(default: the --output folder)")
    p.add_argument("--segmenter", default="isnet", choices=list(SEGMENTERS))
    p.add_argument("--model", default=None,
                   help="path of the .onnx file (default: C:\\Users\\<user>\\.u2net\\<file>)")
    a = p.parse_args()
    T.set_paths(a.dataset, a.output or str(T.REPO_DIR / "results"))
    global WEIGHTS_DIR, SEGMENTER
    SEGMENTER = a.segmenter
    model_path = a.model or str(Path.home() / ".u2net" / SEGMENTERS[SEGMENTER][0])
    WEIGHTS_DIR = Path(a.weights).resolve() if a.weights else None
    if a.step == "status":
        return step_status()
    T.setup_logging()
    out_dir().mkdir(parents=True, exist_ok=True)
    log.info(f"Dataset: {T.DATASET_DIR}\nResults: {out_dir()}")
    blocked = T.block_sleep(True)
    try:
        if a.step in ("masks", "all"):
            step_masks(model_path)
        if a.step in ("evaluate", "all"):
            step_evaluate()
        if a.step in ("background", "all"):
            step_background()
        if a.step == "all":
            T.title("MASKING EXPERIMENTS FINISHED")
    except KeyboardInterrupt:
        log.info("\nInterrupted. Run the same command again to continue.")
    finally:
        if blocked:
            T.block_sleep(False)


if __name__ == "__main__":
    main()

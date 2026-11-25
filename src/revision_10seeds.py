# -*- coding: utf-8 -*-
"""
AddSubLab-8 - all analyses with the same ten seeds (IEEE Access revision).

All eight models are trained again with the ten seeds under ONE protocol, so
that every analysis of the article uses the same 80 runs:

  - 15 epochs, the validation and test sets evaluated after every epoch;
  - early stopping (patience 5 on val_loss, min_delta 1e-4) applied post hoc
    to the validation history, exactly as in revision_experiments.py;
  - the weights of the epoch selected by early stopping are kept
    (training/<model>/seed_<s>/best.pt), so the masked-image evaluation and
    Grad-CAM can use the same models as the tables.

Stages (run in this order by "all"; each one resumes if interrupted):

  main      80 runs (8 models x 10 seeds) with the hyperparameters already
            found (results/hyperparameter_search). The previous results/training
            folder is first renamed to results/training_previous_protocol
            (nothing is deleted).                                  ~21-23 h
  masking   The 80 models on the masked / swapped test images (the IS-Net
            masks already computed in results/masking/isnet/masks are reused;
            the previous evaluation is renamed *_previous_protocol).  ~40 min
  errors    error_analysis.py on the 80 runs (Section VI, Appendix B), with
            Grad-CAM of the new seed-42 models. Output: results/analysis_unified
                                                                    < 1 h
  rotation  The five missing seeds (123, 777, 2718, 99, 4242) of the
            validation <-> test rotation, with its hyperparameters already
            found (results/rotation/hyperparameter_search) and the same
            protocol as its first five seeds (no weights kept).     ~10-11 h
  pack      results_unified.zip (next to the results folder) with what is
            needed for the article: no weights, no per-epoch probabilities.
  all       The five stages, in this order.                       ~33-36 h
  status    Progress of each stage.

Not repeated (independent of these models): hyperparameter searches, ViT-B/16
with ImageNet-21k (already 10 seeds with this protocol), IS-Net masks,
background-only classifiers, frame-to-video mapping.

Place this file in src/ next to train.py, revision_experiments.py,
masking_experiments.py and error_analysis.py.

Usage (Anaconda Prompt):
    conda activate cnn_transformer
    cd C:\\proyectos\\AddSubLab-8
    python src\\revision_10seeds.py all --dataset "C:\\proyectos\\34_Add_Sub_Man_Mac\\dataset_publico\\AddSubLab-8"
    python src\\revision_10seeds.py status
If it stops (power cut, Ctrl+C, restart), run the same command again.
"""

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as R  # noqa: E402  (patches train.evaluate)
T = R.T
import masking_experiments as M  # noqa: E402

log = T.log
SEEDS = R.ALL_SEEDS                      # 42, 1234, 2025, 7, 31415, 123, 777, 2718, 99, 4242
MODELS = R.BASE_MODELS
MARKER = "unified_protocol.json"


# =============================================================================
# Keeping the weights of the epoch selected by the post hoc early stopping
# =============================================================================

class ES:
    """Early-stopping state replayed epoch by epoch (same rule as the article)."""
    keep_weights = False
    stopped = False
    best = float("inf")
    without = 0

    @classmethod
    def reset(cls):
        cls.stopped, cls.best, cls.without = False, float("inf"), 0

    @classmethod
    def update(cls, row):
        if cls.stopped:
            return
        if row["val_loss"] < cls.best - T.MIN_DELTA:
            cls.best, cls.without = row["val_loss"], 0
        else:
            cls.without += 1
            if cls.without >= T.PATIENCE:
                cls.stopped = True


_train_run = T.train_run
_save_torch = T.save_torch


def _train_run_keep_best(**kw):
    """train.train_run (called by revision_experiments.run_full) with best.pt
    written only up to the epoch at which early stopping would have stopped."""
    if not ES.keep_weights:
        return _train_run(**kw)
    ES.reset()
    inner = kw.get("on_epoch_end")

    def on_epoch_end(row, report_only):
        ES.update(row)                      # also replayed when a run resumes
        return inner(row, report_only) if inner else False

    kw["on_epoch_end"] = on_epoch_end
    kw["save_best"] = True
    return _train_run(**kw)


def _save_torch_guard(path, obj):
    # train_run saves best.pt when val_loss improves; after the (simulated)
    # stopping epoch the article's protocol would no longer be training.
    if ES.keep_weights and Path(path).name == "best.pt" and ES.stopped:
        return
    _save_torch(path, obj)


T.train_run = _train_run_keep_best
T.save_torch = _save_torch_guard


def check_weights(folder):
    m = T.read_json(folder / "metrics.json")
    ck = folder / "best.pt"
    if not ck.exists():
        raise SystemExit(f"ERROR: {ck} is missing. Delete {folder} and run again.")
    epoch = torch.load(ck, map_location="cpu", weights_only=False).get("epoch")
    if epoch != m["best_epoch"]:
        raise SystemExit(f"ERROR: {ck} is from epoch {epoch}, but the selected epoch is "
                         f"{m['best_epoch']}. Delete {folder} and run again.")


# =============================================================================
# Stages
# =============================================================================

def archive(path):
    """Renames path -> path_previous_protocol (once)."""
    if path.exists():
        dest = path.with_name(path.stem + "_previous_protocol" + path.suffix)
        if dest.exists():
            raise SystemExit(f"ERROR: both {path} and {dest} exist. Move one of them "
                             f"out of the results folder and run again.")
        path.rename(dest)
        log.info(f"  {path.name} renamed to {dest.name} (kept, not deleted)")


def stage_main():
    T.title("UNIFIED PROTOCOL - 8 models x 10 seeds (weights of the selected epoch kept)")
    train_dir = T.OUTPUT_DIR / "training"
    if not (train_dir / MARKER).exists():
        archive(train_dir)
        archive(T.OUTPUT_DIR / "test_summary.csv")
        train_dir.mkdir(parents=True)
        T.save_json(train_dir / MARKER, {
            "seeds": SEEDS, "max_epochs": T.MAX_EPOCHS, "patience": T.PATIENCE,
            "note": "all runs trained for the full 15 epochs with the test set evaluated "
                    "after every epoch; early stopping applied post hoc; best.pt = "
                    "weights of the selected epoch"})
    device, classes, configs = R.prepare(MODELS)
    ES.keep_weights = True
    try:
        runs = [(s, m) for s in SEEDS for m in MODELS]
        t0 = time.time()
        for i, (s, m) in enumerate(runs, start=1):
            folder = train_dir / m / f"seed_{s}"
            label = f"[Main | run {i}/{len(runs)} | seed {s} | {m}]"
            if not (folder / "metrics.json").exists():
                R.run_full(m, configs[m], R.best_hp(m), s, "training", classes, device, label)
                log.info(f"{label} elapsed in this session: "
                         f"{T.format_duration(time.time() - t0)}")
            check_weights(folder)
    finally:
        ES.keep_weights = False
    T.update_test_summary()


def stage_masking():
    T.title("UNIFIED PROTOCOL - masked-image evaluation of the 80 models")
    M.SEGMENTER = "isnet"
    M.WEIGHTS_DIR = T.OUTPUT_DIR
    o = M.out_dir()
    if not (o / "masks" / "test_masks.npy").exists():
        raise SystemExit(f"ERROR: the masks are not in {o / 'masks'}. Run "
                         f"masking_experiments.py masks first.")
    if not (o / MARKER).exists():
        archive(o / "test_conditions.csv")
        archive(o / "predictions")
        T.save_json(o / MARKER, {"models": "results/training (unified protocol, 10 seeds)"})
    vit21k = T.MODELS.pop(R.VIT21K_KEY, None)      # only the eight models of the study
    try:
        M.step_evaluate()
    finally:
        if vit21k is not None:
            T.MODELS[R.VIT21K_KEY] = vit21k
    n = len({(r["model"], r["seed"]) for r in
             csv.DictReader(open(o / "test_conditions.csv", encoding="utf-8"))})
    if n != len(MODELS) * len(SEEDS):
        raise SystemExit(f"ERROR: only {n} models were evaluated (see the warnings above).")


def stage_errors():
    T.title("UNIFIED PROTOCOL - error analysis (Section VI, Appendix B)")
    out = T.OUTPUT_DIR / "analysis_unified"
    if (out / "error_tables" / "errors_per_image.csv").exists() and \
            any((out / "figures").glob("*gradcam*")):
        log.info("  already done.")
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    script = Path(__file__).resolve().parent / "error_analysis.py"
    cmd = [sys.executable, str(script), "--dataset", str(T.DATASET_DIR),
           "--results", str(T.OUTPUT_DIR), "--weights", str(T.OUTPUT_DIR),
           "--output", str(out)]
    log.info("  " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("ERROR: error_analysis.py failed (see the messages above).")


def stage_rotation():
    T.title("UNIFIED PROTOCOL - rotation, five missing seeds")
    main_output = T.OUTPUT_DIR
    rot = main_output / "rotation"
    hp = {}
    for m in MODELS:
        f = rot / "hyperparameter_search" / m / "best_hyperparameters.json"
        if not f.exists():
            raise SystemExit(f"ERROR: {f} is missing (the rotation search must exist).")
        sel = T.read_json(f)
        hp[m] = {k: sel[k] for k in ("lr", "weight_decay", "label_smoothing")}
    device, classes, configs = R.prepare(MODELS)
    T.OUTPUT_DIR = rot
    original_get = T.CACHE.get
    swap = {"validation": "test", "test": "validation"}
    T.CACHE.get = lambda cfg, split: original_get(cfg, swap.get(split, split))
    ES.keep_weights = False          # the rotation does not need weights
    try:
        R.seeds_loop(MODELS, R.EXTRA_SEEDS, configs, classes, device, "training",
                     "Rotation", lambda m: hp[m])
        T.update_test_summary()
    finally:
        T.CACHE.get = original_get
        T.OUTPUT_DIR = main_output


def stage_pack():
    """results_unified.zip with everything needed for the article, without the
    weights (*.pt) and the per-epoch probability files (epochs/)."""
    import zipfile
    o = T.OUTPUT_DIR
    dest = o.parent / "results_unified.zip"
    parts = [o / "training", o / "rotation" / "training", o / "masking" / "isnet",
             o / "analysis_unified"]
    files = [o / n for n in ("test_summary.csv", "classes.csv", "model_configs.json")
             if (o / n).exists()]
    for p in parts:
        if p.exists():
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix != ".pt"
                      and "epochs" not in f.relative_to(p).parts
                      and "masks" not in f.relative_to(p).parts
                      and "previous_protocol" not in str(f.relative_to(o))]
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("results") / f.relative_to(o))
    log.info(f"  {dest} ({dest.stat().st_size / 1e6:.0f} MB, {len(files)} files): send this file.")


def stage_all():
    t0 = time.time()
    for name, fn in (("main", stage_main), ("masking", stage_masking),
                     ("errors", stage_errors), ("rotation", stage_rotation),
                     ("pack", stage_pack)):
        fn()
        log.info(f"\n=== Stage '{name}' finished | total elapsed: "
                 f"{T.format_duration(time.time() - t0)} ===\n")
    T.title("ALL STAGES FINISHED - send the file results_unified.zip")


def stage_status():
    o = T.OUTPUT_DIR
    tr = o / "training"
    new = (tr / MARKER).exists()
    n = sum((tr / m / f"seed_{s}" / "metrics.json").exists() for m in MODELS for s in SEEDS) if new else 0
    w = sum((tr / m / f"seed_{s}" / "best.pt").exists() for m in MODELS for s in SEEDS) if new else 0
    print(f"Results folder: {o}")
    print(f"  main     : {n}/{len(MODELS) * len(SEEDS)} runs ({w} with weights)"
          + ("" if new else "  [not started]"))
    mo = o / "masking" / "isnet"
    k = 0
    if (mo / MARKER).exists() and (mo / "test_conditions.csv").exists():
        k = len({(r["model"], r["seed"]) for r in csv.DictReader(open(mo / "test_conditions.csv"))})
    print(f"  masking  : {k}/{len(MODELS) * len(SEEDS)} models")
    e = (o / "analysis_unified" / "error_tables" / "errors_per_image.csv").exists()
    print(f"  errors   : {'done' if e else 'pending'}")
    r = sum((o / "rotation" / "training" / m / f"seed_{s}" / "metrics.json").exists()
            for m in MODELS for s in SEEDS)
    print(f"  rotation : {r}/{len(MODELS) * len(SEEDS)} runs")


def main():
    p = argparse.ArgumentParser(description="All analyses with the same ten seeds.")
    p.add_argument("stage", choices=["main", "masking", "errors", "rotation", "pack", "all",
                                     "status"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default=None,
                   help="results folder of train.py (default: <repo>/results)")
    a = p.parse_args()
    T.set_paths(a.dataset, a.output or str(T.REPO_DIR / "results"))
    if a.stage == "status":
        return stage_status()
    T.setup_logging()
    log.info(f"Dataset: {T.DATASET_DIR}\nResults: {T.OUTPUT_DIR}")
    blocked = T.block_sleep(True)
    try:
        {"main": stage_main, "masking": stage_masking, "errors": stage_errors,
         "rotation": stage_rotation, "pack": stage_pack, "all": stage_all}[a.stage]()
    except KeyboardInterrupt:
        log.info("\nInterrupted. Run the same command again to continue.")
    finally:
        if blocked:
            T.block_sleep(False)


if __name__ == "__main__":
    main()

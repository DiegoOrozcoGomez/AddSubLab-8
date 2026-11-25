# -*- coding: utf-8 -*-
"""
AddSubLab-8 - additional experiments requested in the IEEE Access review.

Place this file in the same folder as train.py (src/) and run it with the same
conda environment. It imports train.py and reuses its functions, so the
training protocol is the same as in the manuscript (same augmentation,
optimizer, scheduler, batch, precision, hyperparameters, early stopping).

Experiments (run one at a time; each can be interrupted with Ctrl+C and
resumed by running the same command again):

  seeds     Five additional seeds (10 in total) for the eight models, with the
            best hyperparameters already found. Each run is trained for the
            full 15 epochs and the test set is evaluated after EVERY epoch.
            Early stopping (patience 5 on val_loss, min_delta 1e-4) is then
            applied post hoc to the validation history: because the
            cosine schedule depends only on the 15-epoch maximum, the
            trajectory up to the stopping epoch is the same as with on-line
            early stopping, so test_predictions.csv / metrics.json are
            equivalent to those of the original protocol, and analyze.py
            reads them together with the first five seeds.
            The test F1 at every epoch (per_epoch_test.csv) answers the
            question about the sensitivity of the ranking to the stopping
            rule and to fixed training budgets.
            Results: results/training/<model>/seed_<s>/
            Estimated time: ~9-10 h (RTX-class GPU used in the article).

  vit21k    ViT-B/16 initialized from ImageNet-21k (augreg_in21k_ft_in1k):
            hyperparameter search (12 trials, same space) + 10 seeds.
            Results: results/hyperparameter_search/vit_base_in21k/ and
                     results/training_vit21k/vit_base_in21k/seed_<s>/
            Estimated time: ~5-6 h.

  rotation  Validation <-> test rotation: the test environment is used for
            hyperparameter search and early stopping, and the original
            validation environment is used as the held-out test. Same 8
            models, new hyperparameter search, 5 seeds.
            Results: results/rotation/ (same structure as results/)
            Estimated time: ~25 h (can be run over several days; it resumes).

  all       Runs seeds, vit21k and rotation one after the other (~40 h).
            If it is interrupted, run "all" again: finished runs are skipped.

  status    Shows how many runs of each experiment are finished.

Usage (Anaconda Prompt):
    conda activate cnn_transformer
    cd C:\\proyectos\\AddSubLab-8
    python src\\revision_experiments.py seeds
    python src\\revision_experiments.py vit21k
    python src\\revision_experiments.py rotation
    python src\\revision_experiments.py all
    python src\\revision_experiments.py status

Options: --dataset and --output, as in train.py (default output: results).
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

EXTRA_SEEDS = [123, 777, 2718, 99, 4242]
ALL_SEEDS = T.SEEDS + EXTRA_SEEDS
BASE_MODELS = list(T.MODELS.keys())
VIT21K_KEY = "vit_base_in21k"
T.MODELS[VIT21K_KEY] = ("vit_base_patch16_224.augreg_in21k_ft_in1k", "Pure Transformer")
UNSHIFTED = ["delta_3d_printer", "resin_3d_printer", "bench_drill"]

log = T.log


# =============================================================================
# Per-epoch evaluation of the test set (hook on train.evaluate)
# =============================================================================

class Hook:
    active = False
    val_imgs = None
    test = None          # (imgs, labels)
    pending = None


_ORIGINAL_EVALUATE = T.evaluate


def _evaluate_with_test(model, imgs, labels, mean, std, device):
    res = _ORIGINAL_EVALUATE(model, imgs, labels, mean, std, device)
    if Hook.active and imgs is Hook.val_imgs:
        t0 = time.perf_counter()
        te = _ORIGINAL_EVALUATE(model, Hook.test[0], Hook.test[1], mean, std, device)
        Hook.pending = {"val_probs": res["probs"], "test_probs": te["probs"],
                        "test_loss": te["loss"],
                        "test_time_s": time.perf_counter() - t0}
    return res


T.evaluate = _evaluate_with_test


def simulate_early_stopping(history, patience):
    best, best_epoch, without = float("inf"), 0, 0
    for row in history:
        if row["val_loss"] < best - T.MIN_DELTA:
            best, best_epoch, without = row["val_loss"], row["epoch"], 0
        else:
            without += 1
        if without >= patience:
            return best_epoch, row["epoch"], (f"early stopping ({patience} epochs "
                                              f"without val_loss improvement)")
    return best_epoch, history[-1]["epoch"], "maximum epochs"


def run_full(key, cfg, hp, seed, base_folder, classes, device, label):
    """Trains 15 epochs evaluating the test set at every epoch, then applies
    the early-stopping rule of the article post hoc."""
    folder = T.OUTPUT_DIR / base_folder / key / f"seed_{seed}"
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        return T.read_json(metrics_path)
    epochs_dir = folder / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)

    imgs_va, y_va, files_va = T.CACHE.get(cfg, "validation")
    imgs_te, y_te, files_te = T.CACHE.get(cfg, "test")
    Hook.val_imgs, Hook.test, Hook.pending, Hook.active = imgs_va, (imgs_te, y_te), None, True
    test_times = {}

    def on_epoch_end(row, report_only):
        if Hook.pending is not None:
            p = Hook.pending
            np.savez(epochs_dir / f"epoch_{row['epoch']:02d}.npz",
                     val_probs=p["val_probs"].astype(np.float32),
                     test_probs=p["test_probs"].astype(np.float32),
                     test_loss=p["test_loss"], test_time_s=p["test_time_s"])
            Hook.pending = None
        return False

    try:
        res = T.train_run(key=key, cfg=cfg, hp=hp, seed=seed,
                          max_epochs=T.MAX_EPOCHS, patience=T.MAX_EPOCHS,
                          folder=folder, label=label, device=device, classes=classes,
                          on_epoch_end=on_epoch_end, save_best=False)
    finally:
        Hook.active = False

    history = res["history"]
    missing = [r["epoch"] for r in history
               if not (epochs_dir / f"epoch_{r['epoch']:02d}.npz").exists()]
    if missing:
        raise SystemExit(f"{label} ERROR: per-epoch test results missing for epochs "
                         f"{missing}. Delete {folder} and run again.")

    # Per-epoch table (validation and test).
    shifted_idx = [i for i, c in enumerate(classes) if c not in UNSHIFTED]
    unshifted_idx = [i for i, c in enumerate(classes) if c in UNSHIFTED]
    rows, clean_hist = [], []
    for r in history:
        z = np.load(epochs_dir / f"epoch_{r['epoch']:02d}.npz")
        pred = z["test_probs"].argmax(1)
        f_cls = T.f1_score(y_te, pred, average=None, zero_division=0,
                           labels=list(range(len(classes))))
        tt = float(z["test_time_s"])
        rows.append({
            "epoch": r["epoch"], "val_loss": r["val_loss"], "val_f1_macro": r["val_f1_macro"],
            "test_loss": float(z["test_loss"]),
            "test_acc": float(T.accuracy_score(y_te, pred)),
            "test_f1_macro": float(f_cls.mean()),
            "test_f1_shifted": float(f_cls[shifted_idx].mean()),
            "test_f1_unshifted": float(f_cls[unshifted_idx].mean()),
            **{f"test_f1_{c}": float(f_cls[i]) for i, c in enumerate(classes)},
        })
        # Times without the extra test evaluation, comparable with the article.
        h = dict(r)
        h["val_time_s"] = max(0.0, r["val_time_s"] - tt)
        h["epoch_time_s"] = r["train_time_s"] + h["val_time_s"]
        clean_hist.append(h)
    T.save_csv(folder / "per_epoch_test.csv", rows)
    T.save_csv(folder / "history_full.csv", clean_hist)

    best_epoch, stop_epoch, stop_reason = simulate_early_stopping(clean_hist, T.PATIENCE)
    T.save_csv(folder / "history.csv", [h for h in clean_hist if h["epoch"] <= stop_epoch])
    z = np.load(epochs_dir / f"epoch_{best_epoch:02d}.npz")
    p_te, p_va = z["test_probs"], z["val_probs"]
    met_test = T.full_metrics(p_te, y_te, classes)
    met_val = T.full_metrics(p_va, y_va, classes)
    T.save_predictions(folder / "test_predictions.csv", files_te, y_te, p_te, classes)
    T.save_predictions(folder / "validation_predictions.csv", files_va, y_va, p_va, classes)
    np.save(folder / "test_probabilities.npy", p_te)

    final = {
        "model": key, "timm": T.MODELS[key][0], "category": T.MODELS[key][1],
        "seed": seed, "max_epochs": T.MAX_EPOCHS, "patience": T.PATIENCE,
        "hyperparameters": hp, "input_size": cfg["input_size"],
        "interpolation": cfg["interpolation"], "micro_batch": res["micro_batch"],
        "gradient_accumulation": res["accumulation"],
        "best_epoch": best_epoch, "epochs_trained": stop_epoch, "stop_reason": stop_reason,
        "training_time_s": float(sum(h["epoch_time_s"] for h in clean_hist
                                     if h["epoch"] <= stop_epoch)),
        "test_evaluation_time_s": float(z["test_time_s"]),
        "test_metrics": met_test, "validation_metrics": met_val,
        "protocol_note": (f"trained for the full {T.MAX_EPOCHS} epochs with the test set "
                          f"evaluated after every epoch; early stopping (patience "
                          f"{T.PATIENCE}, min_delta {T.MIN_DELTA}) applied post hoc"),
        "end_date": T.datetime.now().isoformat(timespec="seconds"),
    }
    T.save_json(metrics_path, final)
    (folder / "last.pt").unlink(missing_ok=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mt = met_test
    log.info(f"{label} simulated early stop at epoch {stop_epoch}, best epoch "
             f"{best_epoch} | TEST acc {mt['accuracy']:.4f}  f1 {mt['f1_macro']:.4f} | "
             f"test F1 at epoch 15: {rows[-1]['test_f1_macro']:.4f}")
    return final


# =============================================================================
# Preparation (dataset, configurations, cache)
# =============================================================================

def prepare(keys):
    device = T.get_device()
    T.setup_torch()
    if T.block_sleep(True):
        log.info("  Windows sleep blocked while training.")
    data, classes, problems = T.list_dataset()
    for p in problems:
        log.info(f"  WARNING: {p}")
    saved = T.OUTPUT_DIR / "model_configs.json"
    configs = T.read_json(saved) if saved.exists() else {}
    need = [k for k in keys if k not in configs]
    if need:
        backup = dict(T.MODELS)
        T.MODELS.clear()
        T.MODELS.update({k: backup[k] for k in need})
        configs.update(T.check_and_download_models())
        T.MODELS.clear()
        T.MODELS.update(backup)
    configs = {k: configs[k] for k in keys}
    done = set()
    for cfg in configs.values():
        if T.cache_key(cfg) not in done:
            T.build_cache(cfg, data)        # skipped when the cache already exists
            done.add(T.cache_key(cfg))
    return device, classes, configs


def best_hp(key, base=None):
    base = base or T.OUTPUT_DIR
    sel = T.read_json(base / "hyperparameter_search" / key / "best_hyperparameters.json")
    return {k: sel[k] for k in ("lr", "weight_decay", "label_smoothing")}


def seeds_loop(models, seeds, configs, classes, device, base_folder, prefix, hp_of):
    runs = [(s, m) for s in seeds for m in models]
    t_start = time.time()
    for i, (s, m) in enumerate(runs, start=1):
        label = f"[{prefix} | run {i}/{len(runs)} | seed {s} | {m}]"
        if (T.OUTPUT_DIR / base_folder / m / f"seed_{s}" / "metrics.json").exists():
            continue
        run_full(m, configs[m], hp_of(m), s, base_folder, classes, device, label)
        log.info(f"{label} elapsed in this session: "
                 f"{T.format_duration(time.time() - t_start)}")


# =============================================================================
# Experiments
# =============================================================================

def exp_seeds():
    T.title("REVISION - five additional seeds with per-epoch test evaluation")
    device, classes, configs = prepare(BASE_MODELS)
    seeds_loop(BASE_MODELS, EXTRA_SEEDS, configs, classes, device, "training",
               "Extra seeds", best_hp)
    T.update_test_summary()
    log.info("Done. Run analyze.py again: it now reads 10 seeds per model.")


def exp_vit21k():
    T.title("REVISION - ViT-B/16 pretrained on ImageNet-21k")
    device, classes, configs = prepare([VIT21K_KEY])
    sel = T.hyperparameter_search(VIT21K_KEY, configs[VIT21K_KEY], device, classes)
    hp = {k: sel[k] for k in ("lr", "weight_decay", "label_smoothing")}
    seeds_loop([VIT21K_KEY], ALL_SEEDS, configs, classes, device, "training_vit21k",
               "ViT-21k", lambda m: hp)
    T.update_test_summary("training_vit21k", "test_summary_vit21k.csv")


def exp_rotation():
    T.title("REVISION - validation <-> test rotation")
    main_output = T.OUTPUT_DIR
    device, classes, configs = prepare(BASE_MODELS)
    # Results go to results/rotation; the image cache of results/ is reused, with
    # the roles of the validation and test splits exchanged.
    T.OUTPUT_DIR = main_output / "rotation"
    T.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_get = T.CACHE.get
    swap = {"validation": "test", "test": "validation"}
    T.CACHE.get = lambda cfg, split: original_get(cfg, swap.get(split, split))
    T.save_json(T.OUTPUT_DIR / "rotation_note.json", {
        "validation_role": "original test environment (test/ folder)",
        "test_role": "original validation environment (validation/ folder)",
        "note": "In the prediction files of this folder, 'test' rows are images of "
                "the validation/ folder and vice versa."})
    for name in ("classes.csv", "model_configs.json", "efficiency.csv"):
        if (main_output / name).exists():
            (T.OUTPUT_DIR / name).write_bytes((main_output / name).read_bytes())
    best = {}
    for i, key in enumerate(BASE_MODELS, start=1):
        log.info(f"\n--- Rotation search, model {i}/{len(BASE_MODELS)}: {key} ---")
        best[key] = T.hyperparameter_search(key, configs[key], device, classes)
    T.save_json(T.OUTPUT_DIR / "hyperparameter_search" / "best_hyperparameters_all.json", best)
    seeds_loop(BASE_MODELS, T.SEEDS, configs, classes, device, "training", "Rotation",
               lambda m: {k: best[m][k] for k in ("lr", "weight_decay", "label_smoothing")})
    T.update_test_summary()
    log.info(f"Done. Analyze with: python src\\analyze.py --results {T.OUTPUT_DIR} "
             f"--output {T.OUTPUT_DIR / 'analysis'}")


def exp_all():
    t0 = time.time()
    for name, fn in (("seeds", exp_seeds), ("vit21k", exp_vit21k), ("rotation", exp_rotation)):
        fn()
        log.info(f"\n=== Experiment '{name}' finished | total elapsed: "
                 f"{T.format_duration(time.time() - t0)} ===\n")
    T.title("ALL REVISION EXPERIMENTS FINISHED")


def exp_status():
    out = T.OUTPUT_DIR

    def count(base, models, seeds):
        return sum((out / base / m / f"seed_{s}" / "metrics.json").exists()
                   for m in models for s in seeds)
    print(f"Results folder: {out}")
    print(f"  seeds    : {count('training', BASE_MODELS, EXTRA_SEEDS)}/"
          f"{len(BASE_MODELS) * len(EXTRA_SEEDS)} additional runs")
    hs = (out / "hyperparameter_search" / VIT21K_KEY / "best_hyperparameters.json").exists()
    print(f"  vit21k   : search {'done' if hs else 'pending'}; "
          f"{count('training_vit21k', [VIT21K_KEY], ALL_SEEDS)}/{len(ALL_SEEDS)} runs")
    rot = out / "rotation"
    n_hs = sum((rot / "hyperparameter_search" / m / "best_hyperparameters.json").exists()
               for m in BASE_MODELS)
    n_tr = sum((rot / "training" / m / f"seed_{s}" / "metrics.json").exists()
               for m in BASE_MODELS for s in T.SEEDS)
    print(f"  rotation : search {n_hs}/{len(BASE_MODELS)} models; "
          f"{n_tr}/{len(BASE_MODELS) * len(T.SEEDS)} runs")


def main():
    p = argparse.ArgumentParser(description="Additional experiments for the revision.")
    p.add_argument("experiment", choices=["seeds", "vit21k", "rotation", "all", "status"])
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default=None,
                   help="results folder of train.py (default: <repo>/results)")
    a = p.parse_args()
    T.set_paths(a.dataset, a.output or str(T.REPO_DIR / "results"))
    if a.experiment == "status":
        return exp_status()
    T.setup_logging()
    log.info(f"Dataset: {T.DATASET_DIR}\nResults: {T.OUTPUT_DIR}")
    try:
        {"seeds": exp_seeds, "vit21k": exp_vit21k, "rotation": exp_rotation,
         "all": exp_all}[a.experiment]()
    except KeyboardInterrupt:
        log.info("\nInterrupted. Run the same command again to continue.")
    finally:
        T.block_sleep(False)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
AddSubLab-8: training pipeline for the comparison of CNN, hybrid, and
Transformer architectures on eight additive and subtractive manufacturing
machines.

Automatic and resumable pipeline:
    Phase 0. Environment and dataset checks; download of pretrained weights.
    Phase 1. Hyperparameter search (Optuna) for each architecture.
    Phase 2. Final training with 5 seeds, in interleaved order:
             seed 1 -> the 8 models, seed 2 -> the 8 models, ...
    Phase 3. Computational efficiency (parameters, FLOPs, latency).
    Phase 4. Duration ablation (30 epochs, no early stopping) for 5 models.

If the process is interrupted (power cut, closed window, Ctrl+C), run the
same command again: it resumes from the last saved epoch.

Usage:
    python src/train.py --dataset path/to/AddSubLab-8 --output runs

Requirements: see requirements.txt (PyTorch with CUDA, timm, optuna,
scikit-learn, numpy, pillow, psutil).
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import optuna
from optuna.trial import TrialState
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)


# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_DIR = Path(__file__).resolve().parent.parent

# Default paths (can be changed with --dataset and --output).
DATASET_DIR = REPO_DIR / "AddSubLab-8"
OUTPUT_DIR = REPO_DIR / "runs"
CACHE_DIR = OUTPUT_DIR / "cache"
LOG_DIR = OUTPUT_DIR / "logs"

# Dataset subfolders.
SPLITS = {"train": "train", "validation": "validation", "test": "test"}

# Expected images per class (only used to check that the dataset is complete).
EXPECTED_IMAGES = {"train": 3200, "validation": 800, "test": 500}
NUM_CLASSES = 8
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Architectures: short key -> (timm model name, category).
# All of them with ImageNet-1k pretrained weights.
MODELS = {
    "resnet50":        ("resnet50.a1_in1k",                    "Classical CNN"),
    "densenet121":     ("densenet121.tv_in1k",                 "Classical CNN"),
    "xception":        ("legacy_xception.tf_in1k",             "Classical CNN (separable conv.)"),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k",             "Efficient CNN"),
    "convnext_tiny":   ("convnext_tiny.fb_in1k",               "Modernized CNN"),
    "maxvit_tiny":     ("maxvit_tiny_tf_224.in1k",             "Hybrid CNN + Transformer"),
    "swin_tiny":       ("swin_tiny_patch4_window7_224.ms_in1k", "Hierarchical Transformer"),
    "vit_base":        ("vit_base_patch16_224.augreg_in1k",    "Pure Transformer"),
}

# Seeds of the final training (interleaved order).
SEEDS = [42, 1234, 2025, 7, 31415]

# Hyperparameter search.
SEARCH_SEED = 0
TRIALS_PER_MODEL = 12
SEARCH_MAX_EPOCHS = 15
SEARCH_PATIENCE = 5
PRUNING_START_EPOCH = 5          # Optuna pruning acts from this epoch on
INITIAL_RANDOM_TRIALS = 5        # random trials before TPE takes over
STABILITY_EPOCHS = 5             # last epochs averaged by the selection criterion
SEARCH_SPACE = {
    "lr": (1e-5, 1e-3),                  # log scale
    "weight_decay": [0.01, 0.05, 0.1],
    "label_smoothing": [0.0, 0.1],
}

# Final training: same protocol as the search (epochs, patience, and
# learning-rate schedule), so that the selected hyperparameters remain valid.
MAX_EPOCHS = SEARCH_MAX_EPOCHS
PATIENCE = SEARCH_PATIENCE
MIN_DELTA = 1e-4                 # minimum val_loss improvement that resets patience

# Duration ablation (Phase 4): do the models improve with more epochs?
# The final training of one representative model per paradigm is repeated
# with twice the epochs and the full cosine schedule, without early stopping
# (the best validation epoch is still kept). The same seeds as in Phase 2 are
# used, for a paired comparison.
ABLATION_MODELS = ["resnet50", "swin_tiny", "densenet121", "xception", "vit_base"]
ABLATION_EPOCHS = 30
ABLATION_PATIENCE = ABLATION_EPOCHS      # no early stopping
ABLATION_SEEDS = SEEDS[:2]

# Common to search and training.
WARMUP_EPOCHS = 3
EFFECTIVE_BATCH = 32
MIN_MICRO_BATCH = 16             # a warning is printed below this (BatchNorm)
EVAL_BATCH = 64
GRAD_CLIP = 1.0

# Data augmentation (applied on the GPU, after the squash resize).
AUGMENTATION = {
    "horizontal_flip": True,
    "max_zoom": 1.25,            # 1.25 = crops down to 64 % of the area
    "rotation_degrees": 10.0,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
}

# Efficiency measurement.
GPU_LATENCY_REPS = 100
CPU_LATENCY_REPS = 30

# Performance / reproducibility.
DETERMINISTIC = False            # True = bitwise reproducible, but slower
CACHE_WORKERS = max(1, min(16, (os.cpu_count() or 2) - 2))

# Testing only: allows running without a GPU (very slow).
ALLOW_CPU = False


def set_paths(dataset=None, output=None):
    global DATASET_DIR, OUTPUT_DIR, CACHE_DIR, LOG_DIR
    if dataset:
        DATASET_DIR = Path(dataset).resolve()
    if output:
        OUTPUT_DIR = Path(output).resolve()
    CACHE_DIR = OUTPUT_DIR / "cache"
    LOG_DIR = OUTPUT_DIR / "logs"


# =============================================================================
# LOGGING (console + file)
# =============================================================================

log = logging.getLogger("train")


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logfile = logging.FileHandler(
        LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    logfile.setFormatter(logging.Formatter("%(asctime)s | %(message)s",
                                           "%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)
    log.addHandler(logfile)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def title(text):
    line = "=" * 100
    log.info("")
    log.info(line)
    log.info(f"  {text}")
    log.info(line)


def format_duration(seconds):
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d} d {h:02d} h {m:02d} min"
    return f"{h:02d} h {m:02d} min {s:02d} s"


# =============================================================================
# FILE UTILITIES (atomic writes)
# =============================================================================

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_to_json)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _to_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not serializable: {type(o)}")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if columns is None:
        columns = []
        for row in rows:
            for k in row.keys():
                if k not in columns:
                    columns.append(k)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def save_npy(path, array):
    path = Path(path)
    tmp = path.with_name(path.stem + "_tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


# =============================================================================
# PHASE 0: CHECKS
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if ALLOW_CPU:
        log.info("WARNING: no GPU detected; running on CPU (testing only).")
        return torch.device("cpu")
    raise SystemExit(
        "ERROR: PyTorch does not detect a GPU (torch.cuda.is_available() = False).\n"
        "Install PyTorch with CUDA support.")


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def block_sleep(enable):
    """Asks Windows not to sleep or hibernate while training (no-op elsewhere)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        state = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(state))
    except Exception:
        return False


def setup_torch():
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def class_order(folders):
    """Class order (= label index): class_id in metadata/classes.csv if present,
    otherwise natural order of the folder names."""
    classes_csv = DATASET_DIR / "metadata" / "classes.csv"
    if classes_csv.exists():
        with open(classes_csv, encoding="utf-8") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["class_id"]))
        order = [r["class_folder"] for r in rows]
        if sorted(order) != sorted(folders):
            raise SystemExit(f"ERROR: {classes_csv} does not match the class folders.\n"
                             f"{order}\n{folders}")
        return order
    return sorted(folders, key=natural_key)


def list_dataset():
    """Returns {split: (file_list, labels)}, the class list, and warnings."""
    if not DATASET_DIR.exists():
        raise SystemExit(f"ERROR: dataset folder not found: {DATASET_DIR}")

    ref_classes = None
    data = {}
    problems = []
    for split, folder in SPLITS.items():
        path = DATASET_DIR / folder
        if not path.exists():
            raise SystemExit(f"ERROR: missing folder {path}")
        classes = class_order([d.name for d in path.iterdir() if d.is_dir()])
        if ref_classes is None:
            ref_classes = classes
        elif classes != ref_classes:
            raise SystemExit(f"ERROR: the classes of '{folder}' differ from those of "
                             f"'{SPLITS['train']}'.\n{classes}\n{ref_classes}")
        files, labels = [], []
        for idx, cls in enumerate(classes):
            imgs = sorted([p for p in (path / cls).iterdir()
                           if p.is_file() and p.suffix.lower() in EXTENSIONS])
            expected = EXPECTED_IMAGES.get(split)
            if expected is not None and len(imgs) != expected:
                problems.append(f"{folder}/{cls}: {len(imgs)} images "
                                f"({expected} expected)")
            files += [p.relative_to(DATASET_DIR).as_posix() for p in imgs]
            labels += [idx] * len(imgs)
        data[split] = (files, np.array(labels, dtype=np.int64))

    if len(ref_classes) != NUM_CLASSES:
        raise SystemExit(f"ERROR: found {len(ref_classes)} classes, expected "
                         f"{NUM_CLASSES}: {ref_classes}")
    return data, ref_classes, problems


def write_classes_file(classes):
    """Copies metadata/classes.csv (class names used by the analysis scripts)."""
    src = DATASET_DIR / "metadata" / "classes.csv"
    dst = OUTPUT_DIR / "classes.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dst.write_bytes(src.read_bytes())
    else:
        save_csv(dst, [{"class_id": i, "class_folder": c, "class_name": c,
                        "abbreviation": c} for i, c in enumerate(classes, start=1)])


def environment_info():
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "system": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "timm": timm.__version__,
        "optuna": optuna.__version__,
        "numpy": np.__version__,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_memory_gb"] = round(p.total_memory / 1024**3, 1)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            info["nvidia_driver"] = out.stdout.strip()
        except Exception:
            pass
    return info


def current_config():
    return {
        "models": {k: {"timm": v[0], "category": v[1]} for k, v in MODELS.items()},
        "seeds": SEEDS,
        "search": {
            "seed": SEARCH_SEED,
            "trials_per_model": TRIALS_PER_MODEL,
            "max_epochs": SEARCH_MAX_EPOCHS,
            "patience": SEARCH_PATIENCE,
            "pruning_start_epoch": PRUNING_START_EPOCH,
            "initial_random_trials": INITIAL_RANDOM_TRIALS,
            "space": SEARCH_SPACE,
            "selection_metric": "mean validation macro F1 over the last "
                                f"{STABILITY_EPOCHS} epochs (stability); tie-break: "
                                "F1 at the best epoch; then lowest val_loss",
        },
        "final_training": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                           "min_delta": MIN_DELTA,
                           "stopping_criterion": "val_loss (cross-entropy without "
                                                 "label smoothing)"},
        "duration_ablation": {"models": ABLATION_MODELS, "max_epochs": ABLATION_EPOCHS,
                              "patience": ABLATION_PATIENCE, "seeds": ABLATION_SEEDS},
        "common": {
            "preprocessing": "direct resize (squash) to the native input size of each "
                             "model, with its interpolation and normalization",
            "optimizer": "AdamW (no weight decay on biases and normalization layers)",
            "scheduler": "linear warmup + cosine, per step",
            "warmup_epochs": WARMUP_EPOCHS,
            "effective_batch": EFFECTIVE_BATCH,
            "precision": "mixed bfloat16",
            "grad_clip": GRAD_CLIP,
            "augmentation": AUGMENTATION,
            "deterministic": DETERMINISTIC,
        },
    }


def check_and_download_models():
    """Checks the model names and downloads the pretrained weights once."""
    configs = {}
    for key, (name, _) in MODELS.items():
        log.info(f"  - {key:<16} {name}")
        try:
            model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
        except Exception as e:
            raise SystemExit(f"ERROR loading '{name}': {e}\n"
                             f"Check the internet connection (weights are downloaded "
                             f"the first time) or the timm version.")
        cfg = timm.data.resolve_model_data_config(model)
        configs[key] = {
            "input_size": list(cfg["input_size"]),
            "interpolation": cfg.get("interpolation", "bicubic"),
            "mean": list(cfg["mean"]),
            "std": list(cfg["std"]),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model
        gc.collect()
    return configs


# =============================================================================
# CACHE OF RESIZED IMAGES (squash)
# =============================================================================

INTERPOLATIONS = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
}


def _load_and_resize(args):
    path, width, height, interp = args
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((width, height), INTERPOLATIONS.get(interp, Image.BICUBIC))
        return np.asarray(img, dtype=np.uint8)


def cache_key(cfg):
    _, height, width = cfg["input_size"]
    return f"{height}x{width}_{cfg['interpolation']}"


def build_cache(cfg, data):
    """Creates the cache of each split (once per resolution)."""
    key = cache_key(cfg)
    _, height, width = cfg["input_size"]
    folder = CACHE_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    for split, (files, labels) in data.items():
        marker = folder / f"{split}_complete.json"
        if marker.exists():
            info = read_json(marker)
            if info.get("files") == files:
                continue
            log.info(f"  Cache {key}/{split} does not match the dataset; rebuilding.")
        log.info(f"  Building cache {key} / {split}: {len(files)} images...")
        t0 = time.time()
        dest = np.empty((len(files), height, width, 3), dtype=np.uint8)
        tasks = [(str(DATASET_DIR / a), width, height, cfg["interpolation"])
                 for a in files]
        with ProcessPoolExecutor(max_workers=CACHE_WORKERS) as ex:
            for i, arr in enumerate(ex.map(_load_and_resize, tasks, chunksize=32)):
                dest[i] = arr
                if (i + 1) % 2000 == 0:
                    log.info(f"    {i + 1}/{len(files)}")
        save_npy(folder / f"{split}_images.npy", dest)
        save_npy(folder / f"{split}_labels.npy", labels)
        save_json(marker, {"files": files, "shape": list(dest.shape),
                           "created": datetime.now().isoformat(timespec="seconds")})
        log.info(f"  Done in {format_duration(time.time() - t0)}")
        del dest
        gc.collect()


class MemoryCache:
    """Keeps in RAM the splits of one resolution (the last one used)."""

    def __init__(self):
        self.key = None
        self.splits = {}

    def get(self, cfg, split):
        key = cache_key(cfg)
        if key != self.key:
            self.splits = {}
            gc.collect()
            self.key = key
        if split not in self.splits:
            folder = CACHE_DIR / key
            imgs = np.load(folder / f"{split}_images.npy")
            labels = np.load(folder / f"{split}_labels.npy")
            files = read_json(folder / f"{split}_complete.json")["files"]
            self.splits[split] = (imgs, labels, files)
        return self.splits[split]


CACHE = MemoryCache()


# =============================================================================
# DATA ON THE GPU: batches, normalization, and augmentation
# =============================================================================

def to_float(x_uint8):
    """(N, H, W, 3) uint8 on the device -> (N, 3, H, W) float in [0, 1]."""
    return x_uint8.permute(0, 3, 1, 2).float().div_(255.0)


class BatchLoader:
    """
    Copies image batches from RAM to the GPU without blocking the CPU.

    Uses two alternating pinned-memory buffers, so that the copy of one batch
    overlaps with the computation of the previous one. It does not change the
    data, only how it is transferred.
    """

    def __init__(self, imgs, device, max_batch):
        self.imgs = imgs
        self.device = device
        self.pinned = device.type == "cuda"
        if self.pinned:
            shape = (max_batch,) + tuple(imgs.shape[1:])
            self.bufs = [torch.empty(shape, dtype=torch.uint8).pin_memory()
                         for _ in range(2)]
            self.views = [b.numpy() for b in self.bufs]
            self.events = [None, None]
            self.k = 0

    def load(self, indices):
        if not self.pinned:
            return torch.from_numpy(self.imgs[indices]).to(self.device)
        k = self.k
        self.k ^= 1
        if self.events[k] is not None:
            self.events[k].synchronize()   # the buffer finished its previous copy
        n = len(indices)
        np.take(self.imgs, indices, axis=0, out=self.views[k][:n], mode="clip")
        x = self.bufs[k][:n].to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.events[k] = event
        return x

    def load_range(self, start, end):
        return self.load(np.arange(start, end))


def normalize(x, mean, std):
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


def augment(x, gen):
    """Per-sample augmentation on the GPU (images are already square)."""
    b = x.shape[0]
    device = x.device

    def rnd(*shape):
        return torch.rand(*shape, generator=gen, device=device)

    # Geometric transform: zoom, rotation, translation, and flip.
    zoom = 1.0 + (AUGMENTATION["max_zoom"] - 1.0) * rnd(b)
    scale = 1.0 / zoom
    ang = (rnd(b) * 2 - 1) * math.radians(AUGMENTATION["rotation_degrees"])
    if AUGMENTATION["horizontal_flip"]:
        flip = torch.where(rnd(b) < 0.5, -1.0, 1.0)
    else:
        flip = torch.ones(b, device=device)
    tx = (rnd(b) * 2 - 1) * (1 - scale)
    ty = (rnd(b) * 2 - 1) * (1 - scale)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([scale * cos * flip, -scale * sin, tx], dim=1),
        torch.stack([scale * sin * flip, scale * cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)

    # Color: brightness, contrast, and saturation.
    def factor(amp):
        return (1.0 + (rnd(b) * 2 - 1) * amp).view(b, 1, 1, 1)

    x = x * factor(AUGMENTATION["brightness"])
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor(AUGMENTATION["contrast"]) + mean
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = (x - gray) * factor(AUGMENTATION["saturation"]) + gray
    return x.clamp_(0.0, 1.0)


# =============================================================================
# MODEL, OPTIMIZER, AND LEARNING RATE
# =============================================================================

def create_model(key, device, pretrained=True):
    name = MODELS[key][0]
    model = timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)
    return model.to(device).to(memory_format=torch.channels_last)


def create_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": lr, "betas": (0.9, 0.999)}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)


def lr_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def find_micro_batch(model, cfg, device, forced=None):
    """Tries the largest micro-batch (32, 16, 8, ...) that fits in the GPU."""
    if device.type != "cuda":
        return EFFECTIVE_BATCH
    _, height, width = cfg["input_size"]
    n_params = sum(p.numel() for p in model.parameters())
    m = forced or EFFECTIVE_BATCH
    # Buffers (BatchNorm statistics) and the random state are saved so that
    # the test does not alter the model or the reproducibility.
    buffers_copy = [b.detach().clone() for b in model.buffers()]
    with torch.random.fork_rng(devices=[device]):
        m = _try_micro_batch(model, height, width, n_params, m, device)
    with torch.no_grad():
        for b, c in zip(model.buffers(), buffers_copy):
            b.copy_(c)
    return m


def _try_micro_batch(model, height, width, n_params, m, device):
    model.train()
    while m >= 1:
        reserve = x = out = None
        fits = False
        try:
            # Simulates the memory of the AdamW state (2 copies of the weights).
            reserve = torch.empty(n_params * 2, dtype=torch.float32, device=device)
            x = torch.randn(m, 3, height, width, device=device).contiguous(
                memory_format=torch.channels_last)
            with autocast(device):
                out = model(x)
            out.float().mean().backward()
            fits = True
        except torch.cuda.OutOfMemoryError:
            pass
        reserve = x = out = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        if fits:
            return m
        m //= 2
    raise RuntimeError("The model does not fit in the GPU even with micro-batch 1.")


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, imgs, labels, mean, std, device):
    model.eval()
    loader = BatchLoader(imgs, device, EVAL_BATCH)
    y_all = torch.from_numpy(labels).to(device)
    probs = []
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(imgs), EVAL_BATCH):
        end = min(i + EVAL_BATCH, len(imgs))
        x = normalize(to_float(loader.load_range(i, end)), mean, std)
        with autocast(device):
            logits = model(x)
        logits = logits.float()
        total_loss += F.cross_entropy(logits, y_all[i:end], reduction="sum").double()
        probs.append(torch.softmax(logits, dim=1))
    probs = torch.cat(probs).cpu().numpy()     # a single synchronization
    pred = probs.argmax(1)
    return {
        "loss": total_loss.item() / len(imgs),
        "acc": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "probs": probs,
        "pred": pred,
    }


def calibration_error(probs, labels, bins=15):
    confidence = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (confidence > a) & (confidence <= b)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def full_metrics(probs, labels, classes):
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(labels, pred, zero_division=0,
                                                 labels=list(range(len(classes))))
    pm, rm, fm, _ = precision_recall_fscore_support(labels, pred, average="macro",
                                                    zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(labels, pred,
                                                    average="weighted", zero_division=0)
    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "precision_weighted": float(pw), "recall_weighted": float(rw),
        "f1_weighted": float(fw),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "kappa": float(cohen_kappa_score(labels, pred)),
        "roc_auc_ovr_macro": auc,
        "ece": calibration_error(probs, labels),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(
            labels, pred, labels=list(range(len(classes)))).tolist(),
    }


def save_predictions(path, files, labels, probs, classes):
    rows = []
    pred = probs.argmax(1)
    for i, a in enumerate(files):
        row = {"file": a, "true": classes[labels[i]], "predicted": classes[pred[i]],
               "correct": int(pred[i] == labels[i])}
        for j, c in enumerate(classes):
            row[f"prob_{c}"] = f"{probs[i, j]:.6f}"
        rows.append(row)
    save_csv(path, rows)


# =============================================================================
# TRAINING LOOP (shared by the search and the final training)
# =============================================================================

PRUNED_REASON = "pruned by Optuna"


def train_run(*, key, cfg, hp, seed, max_epochs, patience, folder, label, device,
              classes, on_epoch_end=None, save_best=True):
    """
    Trains a model with a checkpoint per epoch and automatic resumption.
    Returns a dict with the history and the summary of the run.
    """
    folder.mkdir(parents=True, exist_ok=True)
    last_path = folder / "last.pt"
    best_path = folder / "best.pt"
    micro_path = folder / "forced_micro_batch.json"

    imgs_tr, y_tr, _ = CACHE.get(cfg, "train")
    imgs_va, y_va, files_va = CACHE.get(cfg, "validation")
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    set_seed(seed)
    resume = last_path.exists()
    model = create_model(key, device, pretrained=not resume)
    optim = create_optimizer(model, hp["lr"], hp["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])

    state = {"epoch": 0, "history": [], "best_val_loss": float("inf"),
             "best_epoch": 0, "no_improvement": 0, "finished": False,
             "stop_reason": None, "elapsed_time": 0.0, "micro_batch": None,
             "max_epochs": max_epochs}

    if resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        state = ck["state"]
        if state.get("max_epochs") != max_epochs:
            raise SystemExit(
                f"{label} ERROR: this run was started with a different maximum "
                f"number of epochs ({state.get('max_epochs')}; now {max_epochs}). "
                f"Delete the folder {folder} and run again.")
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        # Random states must be restored from the CPU, even if the checkpoint
        # was loaded on the GPU.
        torch.set_rng_state(ck["rng_torch"].cpu())
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        log.info(f"{label} Resuming from epoch {state['epoch']} "
                 f"(best epoch so far: {state['best_epoch']})")
        if on_epoch_end is not None:
            for row in state["history"]:
                on_epoch_end(row, report_only=True)

    forced = read_json(micro_path)["micro_batch"] if micro_path.exists() else None
    micro = find_micro_batch(model, cfg, device, forced)
    if state["micro_batch"] not in (None, micro) and not forced:
        micro = state["micro_batch"]
    state["micro_batch"] = micro
    accumulation = math.ceil(EFFECTIVE_BATCH / micro)
    if micro < MIN_MICRO_BATCH:
        log.info(f"{label} WARNING: micro-batch {micro} < {MIN_MICRO_BATCH}; "
                 f"BatchNorm statistics may be noisy.")
    if not resume:
        log.info(f"{label} lr={hp['lr']:.2e}  wd={hp['weight_decay']}  "
                 f"ls={hp['label_smoothing']}  seed={seed}  "
                 f"input={cfg['input_size'][1]}x{cfg['input_size'][2]}  "
                 f"micro-batch={micro} x {accumulation} = {micro * accumulation}")

    n = len(imgs_tr)
    steps_per_epoch = math.ceil(n / EFFECTIVE_BATCH)
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    gen = torch.Generator(device=device)
    loader = BatchLoader(imgs_tr, device, micro)

    try:
        while not state["finished"] and state["epoch"] < max_epochs:
            epoch = state["epoch"] + 1
            model.train()
            order = np.random.default_rng(seed * 1000 + epoch).permutation(n)
            gen.manual_seed(seed * 1000 + epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Sums are accumulated on the GPU and read once per epoch, so that
            # the CPU does not wait for the GPU at every batch.
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            hits = torch.zeros((), dtype=torch.int64, device=device)
            seen = 0
            batches = [np.sort(order[p * EFFECTIVE_BATCH:(p + 1) * EFFECTIVE_BATCH])
                       for p in range(steps_per_epoch)]
            y_epoch = torch.from_numpy(y_tr[np.concatenate(batches)]).to(device)

            for step in range(steps_per_epoch):
                global_step = (epoch - 1) * steps_per_epoch + step
                lr = hp["lr"] * lr_factor(global_step, total_steps, warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = lr
                idx = batches[step]
                batch_size = len(idx)
                for k in range(0, batch_size, micro):
                    sub = idx[k:k + micro]
                    pos = step * EFFECTIVE_BATCH + k
                    x = augment(to_float(loader.load(sub)), gen)
                    x = normalize(x, mean, std)
                    y = y_epoch[pos:pos + len(sub)]
                    with autocast(device):
                        logits = model(x)
                    loss = criterion(logits.float(), y)
                    (loss * len(sub) / batch_size).backward()
                    loss_sum += loss.detach().double() * len(sub)
                    hits += (logits.argmax(1) == y).sum()
                    seen += len(sub)
                if GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_train = time.perf_counter() - t0
            t1 = time.perf_counter()
            val = evaluate(model, imgs_va, y_va, mean, std, device)
            t_val = time.perf_counter() - t1

            mem = (torch.cuda.max_memory_allocated() / 1024**3
                   if device.type == "cuda" else 0.0)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum.item() / seen,
                "train_acc": hits.item() / seen,
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_f1_macro": val["f1_macro"],
                "final_lr": lr,
                "train_time_s": t_train,
                "val_time_s": t_val,
                "epoch_time_s": t_train + t_val,
                "gpu_memory_gb": mem,
            }
            state["history"].append(row)
            state["elapsed_time"] += t_train + t_val
            state["epoch"] = epoch

            improved = val["loss"] < state["best_val_loss"] - MIN_DELTA
            if improved:
                state["best_val_loss"] = val["loss"]
                state["best_epoch"] = epoch
                state["no_improvement"] = 0
                if save_best:
                    save_torch(best_path, {"model": model.state_dict(), "epoch": epoch})
            else:
                state["no_improvement"] += 1

            log.info(f"{label} Epoch {epoch:02d}/{max_epochs} | "
                     f"loss {row['train_loss']:.4f}  acc {row['train_acc']:.4f} | "
                     f"val_loss {row['val_loss']:.4f}  val_acc {row['val_acc']:.4f}  "
                     f"val_f1 {row['val_f1_macro']:.4f} | lr {lr:.2e} | "
                     f"{row['epoch_time_s']:.1f} s"
                     f"{'  * best' if improved else ''}")

            if state["no_improvement"] >= patience:
                state["finished"] = True
                state["stop_reason"] = (f"early stopping ({patience} epochs without "
                                        f"val_loss improvement)")
            elif epoch >= max_epochs:
                state["finished"] = True
                state["stop_reason"] = "maximum epochs"

            prune = False
            if on_epoch_end is not None and not state["finished"]:
                prune = on_epoch_end(row, report_only=False)
                if prune:
                    state["finished"] = True
                    state["stop_reason"] = PRUNED_REASON
            elif on_epoch_end is not None:
                on_epoch_end(row, report_only=True)

            save_csv(folder / "history.csv", state["history"])
            save_torch(last_path, {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "state": state,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if device.type == "cuda" else None),
            })

        if state["finished"]:
            log.info(f"{label} End: {state['stop_reason']}. Best epoch "
                     f"{state['best_epoch']} (val_loss {state['best_val_loss']:.4f}). "
                     f"Time {format_duration(state['elapsed_time'])}")

    except torch.cuda.OutOfMemoryError:
        new = max(1, micro // 2)
        save_json(micro_path, {"micro_batch": new})
        raise SystemExit(f"{label} ERROR: not enough GPU memory with micro-batch "
                         f"{micro}. {new} will be used when resuming. Run the script "
                         f"again.")

    best = next((r for r in state["history"] if r["epoch"] == state["best_epoch"]), None)
    result = {
        "history": state["history"],
        "best_epoch": state["best_epoch"],
        "best_row": best,
        "epochs_trained": state["epoch"],
        "stop_reason": state["stop_reason"],
        "total_time_s": state["elapsed_time"],
        "micro_batch": micro,
        "accumulation": accumulation,
        "pruned": state["stop_reason"] == PRUNED_REASON,
    }
    del model, optim
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# =============================================================================
# PHASE 1: HYPERPARAMETER SEARCH
# =============================================================================

def params_hash(params):
    text = json.dumps(params, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def hyperparameter_search(key, cfg, device, classes):
    base = OUTPUT_DIR / "hyperparameter_search" / key
    base.mkdir(parents=True, exist_ok=True)
    final_path = base / "best_hyperparameters.json"
    previous = read_json(final_path) if final_path.exists() else None

    storage = f"sqlite:///{(OUTPUT_DIR / 'hyperparameter_search' / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"search_{key}", storage=storage, direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=SEARCH_SEED, n_startup_trials=INITIAL_RANDOM_TRIALS, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=PRUNING_START_EPOCH, interval_steps=1))

    # Trials left unfinished by an interruption are re-queued with the same
    # parameters, to continue from their checkpoint.
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)
        study.enqueue_trial(t.params)

    # When resuming, the sampler seed is shifted by the number of trials
    # already recorded, so that the same random suggestions are not repeated.
    study.sampler = optuna.samplers.TPESampler(
        seed=SEARCH_SEED + len(study.get_trials(deepcopy=False)),
        n_startup_trials=INITIAL_RANDOM_TRIALS, multivariate=True)

    def finished():
        return len(study.get_trials(deepcopy=False,
                                    states=(TrialState.COMPLETE, TrialState.PRUNED)))

    times = []
    while finished() < TRIALS_PER_MODEL:
        n_trial = finished() + 1
        trial = study.ask()
        hp = {
            "lr": trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", SEARCH_SPACE["weight_decay"]),
            "label_smoothing": trial.suggest_categorical(
                "label_smoothing", SEARCH_SPACE["label_smoothing"]),
        }
        folder = base / f"trial_{params_hash(hp)}"
        result_path = folder / "result.json"
        label = f"[Search | {key} | trial {n_trial}/{TRIALS_PER_MODEL}]"

        if result_path.exists():
            res = read_json(result_path)
            for row in res["history"]:
                trial.report(row["val_f1_macro"], row["epoch"])
            if res["pruned"]:
                study.tell(trial, state=TrialState.PRUNED)
            else:
                study.tell(trial, res["objective_value"])
            continue

        if times:
            remaining = (TRIALS_PER_MODEL - n_trial + 1) * np.mean(times)
            log.info(f"{label} Estimated remaining time for {key}: "
                     f"{format_duration(remaining)}")

        def on_epoch_end(row, report_only):
            trial.report(row["val_f1_macro"], row["epoch"])
            if report_only:
                return False
            return trial.should_prune()

        res = train_run(
            key=key, cfg=cfg, hp=hp, seed=SEARCH_SEED,
            max_epochs=SEARCH_MAX_EPOCHS, patience=SEARCH_PATIENCE,
            folder=folder, label=label, device=device, classes=classes,
            on_epoch_end=on_epoch_end, save_best=False)

        best = res["best_row"]
        res["hyperparameters"] = hp
        res["optuna_trial_number"] = trial.number
        res["objective_value"] = best["val_f1_macro"]
        res["val_loss_best_epoch"] = best["val_loss"]
        save_json(result_path, res)
        if res["pruned"]:
            study.tell(trial, state=TrialState.PRUNED)
            log.info(f"{label} Trial pruned by Optuna.")
        else:
            study.tell(trial, res["objective_value"])
        times.append(res["total_time_s"])
        for name in ("last.pt", "best.pt"):
            (folder / name).unlink(missing_ok=True)

    # Summary of all trials and selection of the best one.
    results = []
    for sub in sorted(base.glob("trial_*")):
        if (sub / "result.json").exists():
            r = read_json(sub / "result.json")
            last = [f["val_f1_macro"] for f in r["history"][-STABILITY_EPOCHS:]]
            results.append({
                "trial": sub.name,
                "optuna_trial": r["optuna_trial_number"],
                **r["hyperparameters"],
                "status": "pruned" if r["pruned"] else "complete",
                "val_f1_macro": r["objective_value"],
                "val_loss": r["val_loss_best_epoch"],
                "val_f1_last_epochs": float(np.mean(last)),
                "best_epoch": r["best_epoch"],
                "epochs_trained": r["epochs_trained"],
                "stop_reason": r["stop_reason"],
                "total_time_s": round(r["total_time_s"], 1),
            })
    results.sort(key=lambda r: r["optuna_trial"])
    save_csv(base / "trials_summary.csv", results)

    complete = [r for r in results if r["status"] == "complete"] or results
    # Selection criterion:
    #   1) highest mean validation macro F1 over the last epochs (stability).
    #      An average of several epochs is a less noisy estimator than the F1
    #      of a single epoch, which can be an isolated peak;
    #   2) tie-break: highest validation macro F1 at the best epoch;
    #   3) if still tied: lowest val_loss.
    ranking = sorted(complete, key=lambda r: (-r["val_f1_last_epochs"],
                                              -r["val_f1_macro"],
                                              r["val_loss"]))
    best = ranking[0]
    save_csv(base / "selection_ranking.csv",
             [{"rank": i + 1, **r} for i, r in enumerate(ranking)])
    selection = {
        "model": key,
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "label_smoothing": best["label_smoothing"],
        "val_f1_macro": best["val_f1_macro"],
        "val_loss": best["val_loss"],
        "val_f1_last_epochs": best["val_f1_last_epochs"],
        "trial": best["trial"],
        "criterion": f"mean F1 over the last {STABILITY_EPOCHS} epochs > "
                     "F1 at the best epoch > val_loss",
    }
    save_json(final_path, selection)
    if previous and previous.get("trial") != selection["trial"]:
        log.info(f"[Search | {key}] The selection changed (before: "
                 f"{previous.get('trial')}).")
    log.info(f"[Search | {key}] Best hyperparameters: lr={selection['lr']:.2e}  "
             f"wd={selection['weight_decay']}  ls={selection['label_smoothing']}  "
             f"(val_f1={selection['val_f1_macro']:.4f})")
    return selection


# =============================================================================
# PHASE 2: FINAL TRAINING (interleaved order by seed)
# =============================================================================

def final_training(key, cfg, hp, seed, n_seed, n_model, device, classes,
                   base_folder="training", max_epochs=None, patience=None,
                   prefix="Training", n_seeds=None, n_models=None):
    max_epochs = max_epochs or MAX_EPOCHS
    patience = patience or PATIENCE
    folder = OUTPUT_DIR / base_folder / key / f"seed_{seed}"
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)

    label = (f"[{prefix} | seed {n_seed}/{n_seeds or len(SEEDS)} "
             f"({seed}) | model {n_model}/{n_models or len(MODELS)} {key}]")
    res = train_run(
        key=key, cfg=cfg, hp=hp, seed=seed, max_epochs=max_epochs,
        patience=patience, folder=folder, label=label, device=device, classes=classes)

    # Test evaluation with the weights of the best epoch.
    log.info(f"{label} Evaluating on test with the best epoch ({res['best_epoch']})...")
    model = create_model(key, device, pretrained=False)
    ck = torch.load(folder / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    imgs_te, y_te, files_te = CACHE.get(cfg, "test")
    t0 = time.perf_counter()
    test = evaluate(model, imgs_te, y_te, mean, std, device)
    t_test = time.perf_counter() - t0
    met_test = full_metrics(test["probs"], y_te, classes)
    save_predictions(folder / "test_predictions.csv", files_te, y_te,
                     test["probs"], classes)
    np.save(folder / "test_probabilities.npy", test["probs"])

    imgs_va, y_va, files_va = CACHE.get(cfg, "validation")
    val = evaluate(model, imgs_va, y_va, mean, std, device)
    met_val = full_metrics(val["probs"], y_va, classes)
    save_predictions(folder / "validation_predictions.csv", files_va, y_va,
                     val["probs"], classes)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final = {
        "model": key,
        "timm": MODELS[key][0],
        "category": MODELS[key][1],
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "hyperparameters": hp,
        "input_size": cfg["input_size"],
        "interpolation": cfg["interpolation"],
        "micro_batch": res["micro_batch"],
        "gradient_accumulation": res["accumulation"],
        "best_epoch": res["best_epoch"],
        "epochs_trained": res["epochs_trained"],
        "stop_reason": res["stop_reason"],
        "training_time_s": res["total_time_s"],
        "test_evaluation_time_s": t_test,
        "test_metrics": met_test,
        "validation_metrics": met_val,
        "end_date": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(metrics_path, final)
    (folder / "last.pt").unlink(missing_ok=True)

    mt = met_test
    log.info(f"{label} TEST | acc {mt['accuracy']:.4f}  f1_macro {mt['f1_macro']:.4f}  "
             f"mcc {mt['mcc']:.4f}  kappa {mt['kappa']:.4f}  ece {mt['ece']:.4f}")
    return final


def update_test_summary(base_folder="training", filename="test_summary.csv"):
    rows = []
    for path in sorted((OUTPUT_DIR / base_folder).glob("*/seed_*/metrics.json")):
        r = read_json(path)
        mt = r["test_metrics"]
        rows.append({
            "model": r["model"], "category": r["category"], "seed": r["seed"],
            "test_accuracy": mt["accuracy"], "test_precision_macro": mt["precision_macro"],
            "test_recall_macro": mt["recall_macro"], "test_f1_macro": mt["f1_macro"],
            "test_mcc": mt["mcc"], "test_kappa": mt["kappa"],
            "test_roc_auc": mt["roc_auc_ovr_macro"], "test_ece": mt["ece"],
            "val_f1_macro": r["validation_metrics"]["f1_macro"],
            "best_epoch": r["best_epoch"], "epochs_trained": r["epochs_trained"],
            "training_time_s": round(r["training_time_s"], 1),
            "lr": r["hyperparameters"]["lr"],
            "weight_decay": r["hyperparameters"]["weight_decay"],
            "label_smoothing": r["hyperparameters"]["label_smoothing"],
            "micro_batch": r["micro_batch"],
        })
    save_csv(OUTPUT_DIR / filename, rows)
    return rows


def compare_ablation():
    """Paired table (same model and seed): MAX_EPOCHS vs ABLATION_EPOCHS."""
    rows = []
    for key in ABLATION_MODELS:
        for seed in ABLATION_SEEDS:
            r15 = OUTPUT_DIR / "training" / key / f"seed_{seed}" / "metrics.json"
            r30 = (OUTPUT_DIR / f"ablation_{ABLATION_EPOCHS}_epochs" / key /
                   f"seed_{seed}" / "metrics.json")
            if not (r15.exists() and r30.exists()):
                continue
            a, b = read_json(r15), read_json(r30)
            row = {"model": key, "seed": seed}
            for m in ("accuracy", "f1_macro", "mcc", "ece"):
                row[f"{m}_{MAX_EPOCHS}ep"] = a["test_metrics"][m]
                row[f"{m}_{ABLATION_EPOCHS}ep"] = b["test_metrics"][m]
                row[f"diff_{m}"] = b["test_metrics"][m] - a["test_metrics"][m]
            row[f"best_epoch_{MAX_EPOCHS}ep"] = a["best_epoch"]
            row[f"best_epoch_{ABLATION_EPOCHS}ep"] = b["best_epoch"]
            row[f"time_s_{MAX_EPOCHS}ep"] = round(a["training_time_s"], 1)
            row[f"time_s_{ABLATION_EPOCHS}ep"] = round(b["training_time_s"], 1)
            rows.append(row)
    save_csv(OUTPUT_DIR / f"ablation_{ABLATION_EPOCHS}_epochs_comparison.csv", rows)
    return rows


def estimate_remaining(pending, durations):
    if not durations:
        return None
    global_mean = np.mean([d for v in durations.values() for d in v])
    return sum(np.mean(durations[c]) if c in durations else global_mean
               for c in pending)


# =============================================================================
# PHASE 3: EFFICIENCY
# =============================================================================

def measure_efficiency(key, cfg, device):
    path = OUTPUT_DIR / "efficiency" / f"{key}.json"
    if path.exists():
        return read_json(path)
    _, height, width = cfg["input_size"]
    model = timm.create_model(MODELS[key][0], pretrained=False,
                              num_classes=NUM_CLASSES).eval()
    res = {"model": key, "input_size": cfg["input_size"],
           "parameters": sum(p.numel() for p in model.parameters())}

    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(torch.randn(1, 3, height, width))
        flops = counter.get_total_flops()
        res["gflops"] = flops / 1e9
        res["gmacs"] = flops / 2e9
    except Exception as e:
        res["gflops"] = None
        log.info(f"  Could not count the FLOPs of {key}: {e}")

    def timeit(fn, reps, warmup, sync):
        for _ in range(warmup):
            fn()
        sync()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(times)), float(np.std(times))

    with torch.no_grad():
        # CPU, batch 1, fp32
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        x = torch.randn(1, 3, height, width)
        res["cpu_latency_ms"], res["cpu_latency_std_ms"] = timeit(
            lambda: model(x), CPU_LATENCY_REPS, 5, lambda: None)

        if device.type == "cuda":
            m = model.to(device).to(memory_format=torch.channels_last)
            sync = torch.cuda.synchronize
            x1 = torch.randn(1, 3, height, width, device=device).contiguous(
                memory_format=torch.channels_last)
            res["gpu_fp32_latency_ms"], res["gpu_fp32_latency_std_ms"] = timeit(
                lambda: m(x1), GPU_LATENCY_REPS, 20, sync)

            def bf16_1():
                with autocast(device):
                    m(x1)
            res["gpu_bf16_latency_ms"], res["gpu_bf16_latency_std_ms"] = timeit(
                bf16_1, GPU_LATENCY_REPS, 20, sync)

            x32 = torch.randn(EFFECTIVE_BATCH, 3, height, width, device=device
                              ).contiguous(memory_format=torch.channels_last)
            torch.cuda.reset_peak_memory_stats()

            def bf16_32():
                with autocast(device):
                    m(x32)
            t, _ = timeit(bf16_32, 30, 5, sync)
            res["gpu_bf16_throughput_img_s"] = EFFECTIVE_BATCH / (t / 1000)
            res["inference_memory_batch32_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            del m, x1, x32
            torch.cuda.empty_cache()

    save_json(path, res)
    model = None
    gc.collect()
    return res


# =============================================================================
# MAIN
# =============================================================================

def main():
    setup_logging()
    start = time.time()
    title("ADDSUBLAB-8: CNN / HYBRID / TRANSFORMER COMPARISON")
    log.info(f"Dataset: {DATASET_DIR}")
    log.info(f"Output:  {OUTPUT_DIR}")

    # ------------------------------------------------------------- Phase 0
    title("PHASE 0 - Environment and dataset checks")
    device = get_device()
    setup_torch()
    if block_sleep(True):
        log.info("  Windows sleep and hibernation blocked while training.")
    env = environment_info()
    for k, v in env.items():
        log.info(f"  {k:<16}: {v}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "environment.json", env)
    save_json(OUTPUT_DIR / "config.json", current_config())

    data, classes, problems = list_dataset()
    write_classes_file(classes)
    log.info(f"  Classes: {', '.join(classes)}")
    for split, (files, _) in data.items():
        log.info(f"  {split:<11}: {len(files)} images")
    if problems:
        log.info("  WARNING: differences with the expected number of images:")
        for p in problems:
            log.info(f"    - {p}")

    log.info("  Checking models and pretrained weights:")
    configs = check_and_download_models()
    save_json(OUTPUT_DIR / "model_configs.json", configs)
    for key, c in configs.items():
        log.info(f"    {key:<16} input {c['input_size'][1]}x{c['input_size'][2]}  "
                 f"interp. {c['interpolation']:<8}  {c['parameters'] / 1e6:.1f} M parameters")

    title("Cache of resized images (squash)")
    done = set()
    for key, cfg in configs.items():
        k = cache_key(cfg)
        if k not in done:
            build_cache(cfg, data)
            done.add(k)
    log.info("  Cache ready.")

    # ------------------------------------------------------------- Phase 1
    title("PHASE 1 - Hyperparameter search")
    best = {}
    for i, (key, cfg) in enumerate(configs.items(), start=1):
        log.info(f"\n--- Model {i}/{len(configs)}: {key} ---")
        best[key] = hyperparameter_search(key, cfg, device, classes)
    save_json(OUTPUT_DIR / "hyperparameter_search" / "best_hyperparameters_all.json", best)
    log.info("\nSearch summary:")
    for key, hp in best.items():
        log.info(f"  {key:<16} lr={hp['lr']:.2e}  wd={hp['weight_decay']}  "
                 f"ls={hp['label_smoothing']}  val_f1={hp['val_f1_macro']:.4f}")

    # ------------------------------------------------------------- Phase 2
    title("PHASE 2 - Final training (interleaved order by seed)")
    durations = {}
    runs = [(s, n_s, c, n_m)
            for n_s, s in enumerate(SEEDS, start=1)
            for n_m, c in enumerate(configs.keys(), start=1)]

    def done_path(c, s):
        return OUTPUT_DIR / "training" / c / f"seed_{s}" / "metrics.json"

    for s, n_s, c, n_m in runs:
        if done_path(c, s).exists():
            durations.setdefault(c, []).append(read_json(done_path(c, s))["training_time_s"])
    for s, n_s, c, n_m in runs:
        if done_path(c, s).exists():
            continue
        pending = [cc for ss, _, cc, _ in runs if not done_path(cc, ss).exists()]
        eta = estimate_remaining(pending, durations)
        n_done = len(runs) - len(pending)
        log.info(f"\n--- Run {n_done + 1}/{len(runs)}: seed {s}, {c}"
                 + (f" | estimated remaining time: {format_duration(eta)}" if eta else "")
                 + " ---")
        hp = {k: best[c][k] for k in ("lr", "weight_decay", "label_smoothing")}
        final = final_training(c, configs[c], hp, s, n_s, n_m, device, classes)
        durations.setdefault(c, []).append(final["training_time_s"])
        update_test_summary()
    update_test_summary()

    # ------------------------------------------------------------- Phase 3
    title("PHASE 3 - Computational efficiency")
    rows = []
    for key, cfg in configs.items():
        log.info(f"  Measuring {key}...")
        r = measure_efficiency(key, cfg, device)
        rows.append(r)
        log.info(f"    {r['parameters'] / 1e6:.1f} M parameters | "
                 f"{(r.get('gflops') or 0):.2f} GFLOPs | CPU {r['cpu_latency_ms']:.1f} ms"
                 + (f" | GPU {r['gpu_fp32_latency_ms']:.2f} ms | "
                    f"{r['gpu_bf16_throughput_img_s']:.0f} img/s"
                    if "gpu_fp32_latency_ms" in r else ""))
    save_csv(OUTPUT_DIR / "efficiency.csv",
             [{k: v for k, v in f.items() if k != "input_size"} for f in rows])

    # ------------------------------------------------------------- Phase 4
    abl_models = [c for c in ABLATION_MODELS if c in configs]
    if abl_models and ABLATION_SEEDS:
        title(f"PHASE 4 - Duration ablation ({ABLATION_EPOCHS} epochs, "
              f"no early stopping)")
        abl_folder = f"ablation_{ABLATION_EPOCHS}_epochs"
        for n_s, s in enumerate(ABLATION_SEEDS, start=1):
            for n_m, c in enumerate(abl_models, start=1):
                hp = {k: best[c][k] for k in ("lr", "weight_decay", "label_smoothing")}
                final_training(
                    c, configs[c], hp, s, n_s, n_m, device, classes,
                    base_folder=abl_folder, max_epochs=ABLATION_EPOCHS,
                    patience=ABLATION_PATIENCE, prefix="Ablation",
                    n_seeds=len(ABLATION_SEEDS), n_models=len(abl_models))
                update_test_summary(abl_folder, f"test_summary_{abl_folder}.csv")
        log.info(f"\nComparison {MAX_EPOCHS} vs {ABLATION_EPOCHS} epochs (test):")
        for f in compare_ablation():
            log.info(f"  {f['model']:<12} seed {f['seed']:<6} "
                     f"acc {f[f'accuracy_{MAX_EPOCHS}ep']:.4f} -> "
                     f"{f[f'accuracy_{ABLATION_EPOCHS}ep']:.4f}  "
                     f"f1 {f[f'f1_macro_{MAX_EPOCHS}ep']:.4f} -> "
                     f"{f[f'f1_macro_{ABLATION_EPOCHS}ep']:.4f}")

    title("PIPELINE COMPLETE")
    log.info(f"Time of this execution: {format_duration(time.time() - start)}")
    log.info(f"Results in: {OUTPUT_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help=f"dataset folder (default: {DATASET_DIR})")
    p.add_argument("--output", default=None,
                   help=f"output folder for checkpoints and results "
                        f"(default: {OUTPUT_DIR})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_paths(args.dataset, args.output)
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nInterrupted by the user. Run the script again to continue from "
                 "the last saved point.")
    finally:
        block_sleep(False)

# -*- coding: utf-8 -*-
"""
AddSubLab-8: example images of the dataset (Fig. 1 of the article).

One image per class (columns) and split (rows: training, validation, test),
which shows the change of background, supporting surface, and viewpoint
between the splits.

Usage:
    python src/dataset_overview.py --dataset path/to/AddSubLab-8 [--output analysis]

By default, the image at relative position FRACTION of each class folder (in
frame order) is used. To choose a specific frame, add it to SELECTION, e.g.:
    SELECTION = {("test", "bench_drill"): "bench_drill_test_0250.jpg"}

Outputs:
    <output>/figures/fig01_dataset_examples.pdf / .png
    <output>/error_tables/dataset_examples_content.csv   (file shown in each cell)
"""

import argparse
import csv
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import figure_style as F

REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_DIR / "AddSubLab-8"
RESULTS_DIR = REPO_DIR / "results"
OUTPUT_DIR = REPO_DIR / "analysis"

SPLITS = [("train", "Training"), ("validation", "Validation"), ("test", "Test")]
FRACTION = 0.5
# Frames chosen manually so that the whole machine is visible.
SELECTION = {
    ("train", "fdm_3d_printer_2"): "fdm_3d_printer_2_train_0895.jpg",
    ("validation", "fdm_3d_printer_2"): "fdm_3d_printer_2_validation_0532.jpg",
    ("validation", "bench_drill"): "bench_drill_validation_0735.jpg",
    ("train", "mini_cnc_router"): "mini_cnc_router_train_0010.jpg",
    ("train", "fdm_3d_printer_1"): "fdm_3d_printer_1_train_0346.jpg",
    ("validation", "fdm_3d_printer_1"): "fdm_3d_printer_1_validation_0216.jpg",
    ("test", "fdm_3d_printer_2"): "fdm_3d_printer_2_test_0184.jpg",
    ("test", "mini_cnc_router"): "mini_cnc_router_test_0213.jpg",
    ("validation", "mini_metal_horizontal_milling_machine"): "mini_metal_horizontal_milling_machine_validation_0075.jpg",
    ("train", "mini_metal_lathe"): "mini_metal_lathe_train_0115.jpg",
    ("validation", "mini_metal_lathe"): "mini_metal_lathe_validation_0141.jpg",
    ("test", "mini_metal_lathe"): "mini_metal_lathe_test_0140.jpg",
}


def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def load_classes():
    path = DATASET_DIR / "metadata" / "classes.csv"
    if not path.exists():
        path = RESULTS_DIR / "classes.csv"
    with open(path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["class_id"]))
    return [(r["class_folder"], r["abbreviation"]) for r in rows]


def pick(split, folder):
    d = DATASET_DIR / split / folder
    files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: natural_key(p.name))
    if (split, folder) in SELECTION:
        return d / SELECTION[(split, folder)]
    return files[min(len(files) - 1, int(FRACTION * len(files)))]


def main():
    F.style()
    classes = load_classes()
    cols, rows = len(classes), len(SPLITS)
    width, left, header, gap, margin = 181.0, 6.0, 7.0, 1.0, 0.6
    first = pick(SPLITS[0][0], classes[0][0])
    with Image.open(first) as im:
        aspect = im.height / im.width
    w = (width - left - 2 * margin - (cols - 1) * gap) / cols
    h = w * aspect
    height = 2 * margin + header + rows * h + (rows - 1) * gap
    fig = plt.figure(figsize=(width * F.MM, height * F.MM))
    listing = []
    for i, (split, split_name) in enumerate(SPLITS):
        y = margin + header + i * (h + gap)
        for j, (folder, short) in enumerate(classes):
            x = left + margin + j * (w + gap)
            ax = fig.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])
            ax.set_axis_off()
            path = pick(split, folder)
            with Image.open(path) as im:
                im = im.convert("RGB")
                ax.imshow(im.resize((round(im.width * 500 / im.height), 500), Image.BILINEAR))
            if i == 0:
                ax.text(0.5, 1.03, "\n".join(textwrap.wrap(short, 13)), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7, linespacing=0.95)
            listing.append({"split": split, "class_folder": folder,
                            "file": path.relative_to(DATASET_DIR).as_posix()})
        fig.text((margin + left / 2) / width, 1 - (y + h / 2) / height, split_name,
                 rotation=90, ha="center", va="center", fontsize=8)
    F.save(fig, "dataset_examples")
    out = OUTPUT_DIR / "error_tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(listing).to_csv(out / "dataset_examples_content.csv", index=False)
    print(f"Done. Files used: {out / 'dataset_examples_content.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Example images of the dataset.")
    ap.add_argument("--dataset", default=None, help=f"dataset folder (default: {DATASET_DIR})")
    ap.add_argument("--output", default=None, help=f"output folder (default: {OUTPUT_DIR})")
    a = ap.parse_args()
    if a.dataset:
        DATASET_DIR = Path(a.dataset).resolve()
    if a.output:
        OUTPUT_DIR = Path(a.output).resolve()
    F.FIGURE_DIR = OUTPUT_DIR / "figures"
    main()

# -*- coding: utf-8 -*-
"""
Figure style shared by analyze.py and error_analysis.py (IEEE Access format).

  - No titles inside the figures (the explanation goes in the caption).
  - Arial (or Helvetica) 8 pt; fonts embedded in the PDF.
  - Single-column (88 mm) or double-column (181 mm) width; at most 216 mm high.
  - Subfigure labels (a), (b), ... below each part, in Times New Roman 8 pt.
  - Vector PDF and 600-dpi PNG.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = Path("figures")       # set by the calling script
FORMATS = ["pdf", "png"]
DPI = 600

# Figure number in the article. Figures not listed are numbered from 100.
FIGURE_ORDER = {
    "dataset_examples": 1,
    "hyperparameter_search": 2,
    "loss_curves": 3,
    "accuracy_curves": 4,
    "test_results": 5,
    "critical_difference": 6,
    "efficiency_vs_performance": 7,
    "calibration_ece": 8,
    "validation_vs_test": 9,
    "ablation_30_epochs": 10,
    "confusion_matrices": 11,
    "f1_per_class": 12,
    "error_map_mini_metal_lathe": 13,
    "error_mosaic_mini_metal_lathe": 14,
    "training_reference_mini_metal_lathe": 15,
    "gradcam_mini_metal_lathe": 16,
    "segment_vit_base_mini_metal_lathe_mosaic": 17,
    "segment_vit_base_mini_metal_lathe_gradcam": 18,
    # Appendix
    "protocol_50_epochs": 19,
    "error_map_mini_metal_horizontal_milling_machine": 20,
    "error_mosaic_mini_metal_horizontal_milling_machine": 21,
    "training_reference_mini_metal_horizontal_milling_machine": 22,
    "gradcam_mini_metal_horizontal_milling_machine": 23,
}

MM = 1 / 25.4
COLUMN_WIDTH = 88 * MM
PAGE_WIDTH = 181 * MM
MAX_HEIGHT = 216 * MM

FONT = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
SUBFIG_FONT = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]

C_BG, C_TXT, C_TXT2, C_GRID = "#ffffff", "#000000", "#404040", "#d9d9d9"
C_1, C_2, C_3 = "#2a78d6", "#eb6834", "#1baf7a"     # blue, orange, teal
SEQUENTIAL = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "figure.facecolor": C_BG, "axes.facecolor": C_BG, "savefig.facecolor": C_BG,
        "axes.edgecolor": C_TXT2, "axes.labelcolor": C_TXT, "text.color": C_TXT,
        "xtick.color": C_TXT, "ytick.color": C_TXT, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
    })


_INDEX = {}


def save(fig, key):
    """Saves the figure as figNN_<key>.pdf/.png and updates the index."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    number = FIGURE_ORDER.get(key)
    if number is None:
        number = 100 + len([k for k in _INDEX if k not in FIGURE_ORDER])
    base = f"fig{number:02d}_{key}"
    width_mm, height_mm = (v * 25.4 for v in fig.get_size_inches())
    for ext in FORMATS:
        fig.savefig(FIGURE_DIR / f"{base}.{ext}", dpi=DPI)
    plt.close(fig)
    _INDEX[key] = {"figure": f"Fig. {number}", "file": base, "content": key,
                   "width_mm": round(width_mm), "height_mm": round(height_mm)}
    warning = "  (HEIGHT > 216 mm)" if height_mm > 216.5 else ""
    print(f"  Fig. {number:>3}: {base}  [{width_mm:.0f} x {height_mm:.0f} mm]{warning}")
    write_index()


def write_index():
    path = FIGURE_DIR / "figure_index.csv"
    rows = dict(_INDEX)
    if path.exists():                       # keep entries written by other scripts
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["content"], r)
    order = sorted(rows.values(), key=lambda r: int(r["figure"].split()[-1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "file", "content",
                                          "width_mm", "height_mm"])
        w.writeheader()
        w.writerows(order)


def subfig_label(ax, index, text="", dy=-0.18):
    """(a), (b), ... label centered below the axes, in Times New Roman 8 pt."""
    letter = "abcdefghijklmnopqrstuvwxyz"[index]
    ax.text(0.5, dy, f"({letter}) {text}".strip(), transform=ax.transAxes,
            ha="center", va="top", fontsize=8, fontfamily=SUBFIG_FONT)


def subfig_xlabel(ax, index, text=""):
    """(a), (b), ... label used as the x-axis label (below the tick labels)."""
    letter = "abcdefghijklmnopqrstuvwxyz"[index]
    ax.set_xlabel(f"({letter}) {text}".strip(), fontsize=8, fontfamily=SUBFIG_FONT,
                  labelpad=3)

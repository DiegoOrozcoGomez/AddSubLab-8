# AddSubLab-8

Code and results for the article:

> \*\*Image-Based Recognition of Additive and Subtractive Manufacturing Machines in a Teaching Laboratory: 

The AddSubLab-8 Dataset and a Comparison of CNN, Hybrid, and Transformer Classifiers\*\*
> Diego Orozco-Gomez, Sergio Ruiz-Obando

The AddSubLab-8 dataset contains 36,000 images of eight additive and
subtractive manufacturing machines of a teaching laboratory, extracted from
247 smartphone videos. Training, validation, and test images come from
different videos recorded with the machines placed at different locations, so
the test set measures generalization to an unseen background and viewpoint.

This repository compares eight ImageNet-pretrained architectures (five CNNs,
one hybrid CNN + Transformer, and two Transformers) on the dataset, with
hyperparameter search, five training seeds per model, statistical comparison,
calibration, efficiency measurements, and an error analysis with Grad-CAM.

* **Dataset and trained weights:** Zenodo, https://doi.org/10.5281/zenodo.23485468 
* **Code:** this repository (MIT License)

## Dataset

|ID|Class folder|Class|Type|Train|Validation|Test|
|-|-|-|-|-|-|-|
|1|`delta\_3d\_printer`|Delta 3D Printer|Additive|3200|800|500|
|2|`fdm\_3d\_printer\_1`|Fused Deposition Modeling 3D Printer 1|Additive|3200|800|500|
|3|`fdm\_3d\_printer\_2`|Fused Deposition Modeling 3D Printer 2|Additive|3200|800|500|
|4|`resin\_3d\_printer`|Resin 3D Printer|Additive|3200|800|500|
|5|`bench\_drill`|Bench Drill|Subtractive|3200|800|500|
|6|`mini\_cnc\_router`|Mini CNC Router|Subtractive|3200|800|500|
|7|`mini\_metal\_horizontal\_milling\_machine`|Mini Metal Horizontal Milling Machine|Subtractive|3200|800|500|
|8|`mini\_metal\_lathe`|Mini Metal Lathe|Subtractive|3200|800|500|

Download the dataset from Zenodo and unzip it, for example, next to this
repository:

```
AddSubLab-8/
  train/<class\_folder>/<class\_folder>\_train\_0001.jpg ...
  validation/<class\_folder>/...
  test/<class\_folder>/...
  metadata/classes.csv      class IDs, names, and image counts
  metadata/manifest.csv     one row per image (split, class, frame order, SHA-256)
```

Images are 1080 × 1920 pixels (portrait). File numbers follow the frame order
of the source videos within each split and class. Bystanders appear
incidentally in 78 frames of the training and validation sets; their faces
were blurred in the public release (column `faces\_blurred` of
`manifest.csv`). The models reported in the article were trained on the
original frames, so the public images differ from the experimental data only
in those regions. No test image was modified.

The dataset is released under the
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
license.

## Repository structure

```
src/train.py            training pipeline (search, final training, efficiency, ablation)
src/analyze.py          statistics, tables, and figures 2-12 and 19 of the article
src/error\_analysis.py   error analysis and Grad-CAM (figures 13-18 and 20-23)
src/dataset\_overview.py example images of the dataset (figure 1)
src/figure\_style.py     IEEE Access figure format
results/                outputs of train.py used in the article (see below)
requirements.txt
```

`results/` contains everything except the model weights:

|Path|Content|
|-|-|
|`config.json`, `model\_configs.json`|protocol and per-model input configuration|
|`classes.csv`|class IDs and names|
|`hyperparameter\_search/<model>/`|Optuna trials (history and result of each trial), ranking, selected hyperparameters|
|`training/<model>/seed\_<s>/`|main protocol (15 epochs, 5 seeds): metrics, training history, per-image validation and test predictions, test probabilities|
|`ablation\_30\_epochs/`|duration ablation: 30 epochs without early stopping (5 models, 2 seeds)|
|`protocol\_50\_epochs/`|supplementary runs with a 50-epoch schedule (see the note below)|
|`efficiency/`, `efficiency.csv`|parameters, FLOPs, latency, and throughput|
|`test\_summary\*.csv`, `ablation\_30\_epochs\_comparison.csv`|summary tables|

## Installation

Python 3.10 and a CUDA GPU (for training). The article used:

|Component|Version|
|-|-|
|Python|3.10.21|
|PyTorch|2.14.1 (CUDA 13.2)|
|timm|1.0.30|
|Optuna|5.0.0|
|GPU|NVIDIA RTX A5000 (24 GB)|
|CPU|Intel Xeon w7-2475X, 256 GB RAM|

```
pip install torch --index-url https://download.pytorch.org/whl/<your CUDA version>
pip install -r requirements.txt
```

## Reproducing the article

**1. Statistics, tables, and figures (no GPU, about 1 min).** Uses the
predictions in `results/` and reproduces the numbers of the article exactly:

```
python src/analyze.py
```

Outputs go to `analysis/tables/` (including `statistical\_report.txt`) and
`analysis/figures/`.

**2. Error analysis and Grad-CAM.** Needs the dataset and, for Grad-CAM, the
seed-42 weights from Zenodo (`weights/<model>\_seed42.pt`):

```
python src/error\_analysis.py --dataset ../AddSubLab-8 --weights weights
```

Use `--no-gradcam` to skip Grad-CAM (then the weights are not needed).
The example images of Fig. 1 are generated with

```
python src/dataset\_overview.py --dataset ../AddSubLab-8
```

**3. Full training (GPU).** Runs the five phases and writes everything to
`runs/` (the provided `results/` folder is not modified). It is resumable: if
interrupted, run the same command again.

```
python src/train.py --dataset ../AddSubLab-8 --output runs
python src/analyze.py --results runs --output analysis\_runs
```

On the hardware above, the hyperparameter search (96 trials) took about 16 h
of training time, the main training (40 runs) about 7 h, and the duration
ablation (10 runs) about 5 h.

### Notes on reproducibility

* Re-running the training gives close but not bitwise-identical results:
cuDNN autotuning is enabled for speed (`DETERMINISTIC = False` in
`train.py`), and the public images differ from the original ones in the 78
blurred frames. The exact numbers of the article are reproduced from the
saved predictions (step 1).
* `protocol\_50\_epochs/` contains 14 runs (seeds 42 and 1234) of an earlier
final training that used a 50-epoch cosine schedule and patience 10 with the
same selected hyperparameters, before the final protocol was aligned with
the 15-epoch search. They are reported as supplementary material. To repeat
them, set `MAX\_EPOCHS = 50` and `PATIENCE = 10` in `train.py` and use a
different `--output` folder.

## Citation

```
\[BibTeX entry to be added after publication]
```

## License

Code: MIT License (see `LICENSE`). Dataset and weights: CC BY-NC 4.0.


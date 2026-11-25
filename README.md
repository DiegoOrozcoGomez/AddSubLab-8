# AddSubLab-8

Code and results for the article:

> **Image-Based Recognition of Additive and Subtractive Manufacturing Machines in a Teaching Laboratory: The AddSubLab-8 Dataset and a Comparison of CNN, Hybrid, and Transformer Classifiers**
> Diego Orozco-Gomez, Sergio Ruiz-Obando

The AddSubLab-8 dataset contains 36,000 images of eight additive and
subtractive manufacturing machines of two teaching laboratories (mechatronics
and mechanics), extracted from 247 smartphone videos. Each class is a single
physical machine, so the task is instance recognition. Training, validation,
and test images come from different videos, and five of the eight machines were
relocated to a different supporting surface and background between training
and test; the bench drill and the delta and resin printers stayed on the same
surface. The test set therefore measures generalization to a new environment
rather than to new frames of the same scene.

This repository compares eight ImageNet-pretrained architectures (five CNNs,
one hybrid CNN + Transformer, and two Transformers) on the dataset, with
per-model hyperparameter search, ten training seeds per model, statistical
comparison over seeds and over test videos, calibration, efficiency
measurements, and an error analysis with Grad-CAM. It also contains the
additional experiments of the article: the test performance at every training
epoch, a validation-test rotation, ViT-B/16 with ImageNet-21k pretraining, and
an evaluation of background reliance with masked and background-swapped test
images.

* **Dataset and trained weights:** Zenodo, https://doi.org/10.5281/zenodo.24544128 (version 2, weights of the revised article)
* **Code:** this repository (MIT License)

## Dataset

|ID|Class folder|Class|Type|Train|Validation|Test|
|-|-|-|-|-|-|-|
|1|`delta_3d_printer`|Delta 3D Printer|Additive|3200|800|500|
|2|`fdm_3d_printer_1`|Fused Deposition Modeling 3D Printer 1|Additive|3200|800|500|
|3|`fdm_3d_printer_2`|Fused Deposition Modeling 3D Printer 2|Additive|3200|800|500|
|4|`resin_3d_printer`|Resin 3D Printer|Additive|3200|800|500|
|5|`bench_drill`|Bench Drill|Subtractive|3200|800|500|
|6|`mini_cnc_router`|Mini CNC Router|Subtractive|3200|800|500|
|7|`mini_metal_horizontal_milling_machine`|Mini Metal Horizontal Milling Machine|Subtractive|3200|800|500|
|8|`mini_metal_lathe`|Mini Metal Lathe|Subtractive|3200|800|500|

Download the dataset from Zenodo and unzip it, for example, next to this
repository:

```
AddSubLab-8/
  train/<class_folder>/<class_folder>_train_0001.jpg ...
  validation/<class_folder>/...
  test/<class_folder>/...
  metadata/classes.csv      class IDs, names, and image counts
  metadata/manifest.csv     one row per image (split, class, frame order, SHA-256)
```

Images are 1080 × 1920 pixels (portrait). File numbers follow the frame order
of the source videos within each split and class. The source video of every
image is given in `results/video_mapping/image_to_video.csv`. Bystanders
appear incidentally in 78 frames of the training and validation sets; their
faces were blurred in the public release (column `faces_blurred` of
`manifest.csv`). The models reported in the article were trained on the
original frames, so the public images differ from the experimental data only
in those regions. No test image was modified.

The dataset is released under the
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
license.

## Repository structure

```
src/train.py                  training pipeline (search, efficiency, duration ablation)
src/revision_10seeds.py       final training of the 80 runs (8 models x 10 seeds), masked-image
                              evaluation, error analysis, and validation-test rotation
src/revision_experiments.py   ViT-B/16 with ImageNet-21k and the rotation search
src/masking_experiments.py    machine masks (IS-Net), masked/swapped test images, background-only classifiers
src/map_frames_to_videos.py   assignment of every image to its source video
src/analyze.py                statistics, tables, and figures of the main comparison
src/error_analysis.py         error analysis and Grad-CAM
src/dataset_overview.py       example images of the dataset (Fig. 1)
src/revision_analysis/        video-level analysis, per-epoch figure, relocated-class F1 (see its README)
src/figure_style.py           IEEE Access figure format
results/                      outputs used in the article (see below)
analysis/                     tables and figures generated from results/
requirements.txt
```

`results/` contains everything except the model weights:

|Path|Content|
|-|-|
|`config.json`, `model_configs.json`|protocol and per-model input configuration|
|`classes.csv`|class IDs and names|
|`hyperparameter_search/<model>/`|Optuna trials (history and result of each trial), ranking, selected hyperparameters; includes `vit_base_in21k`|
|`training/<model>/seed_<s>/`|main protocol (15 epochs, 10 seeds): metrics, training history, per-image validation and test predictions, test probabilities, and the test performance after every epoch (`per_epoch_test.csv`)|
|`training_vit21k/vit_base_in21k/`|ViT-B/16 with ImageNet-21k pretraining (10 seeds)|
|`rotation/`|validation-test rotation: its own hyperparameter search and 10 seeds, selected on the original test environment and evaluated on the original validation environment|
|`masking/isnet/`|masked-image evaluation of the 80 models (`test_conditions.csv`, per-image predictions), background-only classifiers, and mask quality-control sheets|
|`video_mapping/`|source video of every image, frame counts, and per-video summary|
|`revision_analysis/`|video-level results and bootstrap tests, per-epoch results, relocated-class F1|
|`analysis_unified/error_tables/`|per-image and per-class error tables of the 80 runs|
|`ablation_30_epochs/`|duration ablation: 30 epochs without early stopping (5 models, 2 seeds)|
|`protocol_50_epochs/`|supplementary runs with a 50-epoch schedule (see the note below)|
|`efficiency/`, `efficiency.csv`|parameters, FLOPs, latency, and throughput|
|`test_summary*.csv`, `ablation_30_epochs_comparison.csv`|summary tables|

### Training protocol of the 80 runs

Every run was trained for the full 15 epochs, and the validation and test sets
were evaluated after every epoch. The early-stopping rule (patience 5 on the
validation loss) was then applied to the recorded validation history, and the
weights of the selected epoch were kept. Because the learning-rate schedule
depends only on the 15-epoch maximum, the selected checkpoints are those that
on-line early stopping would have produced. The test set played no role in the
hyperparameter search, the early stopping, or the choice of checkpoint; the
per-epoch test results are used only for the analysis of Fig. 11.

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

The masking experiments also need `onnxruntime` and the IS-Net model
`isnet-general-use.onnx` (see the header of `src/masking_experiments.py`).

## Reproducing the article

**1. Statistics, tables, and figures (no GPU).** These steps use the saved
predictions in `results/` and reproduce the numbers of the article exactly:

```
python src/analyze.py
python src/revision_analysis/video_level.py results
python src/revision_analysis/video_hboot.py results
python src/revision_analysis/video_perm.py results
python src/revision_analysis/per_epoch.py results
python src/revision_analysis/relocated_f1.py results
```

`analyze.py` writes to `analysis/tables/` (including `statistical_report.txt`)
and `analysis/figures/`; the other scripts write to `results/revision_analysis/`.
The results of the rotation, ImageNet-21k, and masking experiments are in the
corresponding folders of `results/`.

**2. Error analysis and Grad-CAM.** Needs the dataset and, for Grad-CAM, the
seed-42 weights from Zenodo (`weights/<model>_seed42.pt`):

```
python src/error_analysis.py --dataset ../AddSubLab-8 --weights weights
```

Use `--no-gradcam` to skip Grad-CAM (then the weights are not needed).
The example images of Fig. 1 are generated with

```
python src/dataset_overview.py --dataset ../AddSubLab-8
```

**3. Full training (GPU).** All steps are resumable: if interrupted, run the
same command again.

```
python src/train.py --dataset ../AddSubLab-8 --output runs
python src/revision_experiments.py vit21k --dataset ../AddSubLab-8 --output runs
python src/revision_experiments.py rotation --dataset ../AddSubLab-8 --output runs
python src/masking_experiments.py masks --dataset ../AddSubLab-8 --output runs
python src/masking_experiments.py background --dataset ../AddSubLab-8 --output runs
python src/revision_10seeds.py all --dataset ../AddSubLab-8 --output runs
python src/analyze.py --results runs --output analysis_runs
```

`train.py` runs the hyperparameter search, the efficiency measurements, and the
duration ablation; `revision_10seeds.py` then trains the 80 final runs with the
selected hyperparameters and runs the masked-image evaluation, the error
analysis, and the remaining seeds of the rotation. The image-to-video mapping
requires the original videos, which are not distributed; the mapping itself is
provided in `results/video_mapping/`.

On the hardware above, the hyperparameter search (96 trials) took about 16 h
of training time, the 80 final runs about 20 h (13.8 h up to the selected
epochs), and the duration ablation (10 runs) about 5 h.

### Figure numbering

The figure files keep the numbering of the scripts, which differs from the
article after Fig. 10: the per-epoch figure (Fig. 11 of the article) is
written to `results/revision_analysis/`, and two figures of an earlier version
were removed. `analysis/figures/figure_index.csv` gives the article figure of
every file.

### Notes on reproducibility

* Re-running the training gives close but not bitwise-identical results:
cuDNN autotuning is enabled for speed (`DETERMINISTIC = False` in
`train.py`), and the public images differ from the original ones in the 78
blurred frames. The exact numbers of the article are reproduced from the
saved predictions (step 1).
* `protocol_50_epochs/` contains 14 runs (seeds 42 and 1234) of an earlier
final training that used a 50-epoch cosine schedule and patience 10 with the
same selected hyperparameters, before the final protocol was aligned with
the 15-epoch search. They are reported as supplementary material. To repeat
them, set `MAX_EPOCHS = 50` and `PATIENCE = 10` in `train.py` and use a
different `--output` folder.

## License

Code: MIT License (see `LICENSE`). Dataset and weights: CC BY-NC 4.0.

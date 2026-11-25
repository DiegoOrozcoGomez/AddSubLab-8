# Revision analyses (AddSubLab-8, IEEE Access revision)

Scripts that produce the video-level, per-epoch and relocated-class analyses of the revised manuscript.
They read the outputs of `train.py`, `revision_experiments.py` and `map_frames_to_videos.py` and write
to `results/revision_analysis/`.

Run from the repository root (or pass the results folder as the only argument):

```
python video_level.py   results   # Table V: video-level majority vote, video-bootstrap 95% CI of macro F1
python video_hboot.py   results   # pairwise comparisons: two-level bootstrap (videos within class x seeds), Holm
python video_perm.py    results   # pairwise comparisons: sign-flip permutation test over videos, Holm (sensitivity)
python per_epoch.py     results   # Fig. 11: test macro F1 per epoch (five additional seeds)
python relocated_f1.py  results   # macro F1 over the five relocated machines (Table IV)
```

Expected inputs:

| Path | Produced by |
|---|---|
| `results/classes.csv` | `train.py` |
| `results/training/<model>/seed_<s>/test_predictions.csv`, `metrics.json`, `per_epoch_test.csv` | `train.py`, `revision_experiments.py seeds` |
| `results/training_vit21k/vit_base_in21k/seed_<s>/test_predictions.csv` | `revision_experiments.py vit21k` |
| `results/video_mapping/image_to_video.csv` | `map_frames_to_videos.py` |

The bootstraps use fixed random seeds; `video_level.py` and `video_hboot.py` take several minutes on a CPU.

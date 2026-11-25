# -*- coding: utf-8 -*-
"""
AddSubLab-8 - assigns every image of the dataset to the video it was taken from.

The frames were selected manually from the videos, so the mapping is recovered
by image content: every frame of every video is decoded with OpenCV (the same
library used to extract the frames), reduced to a 32x32 grayscale descriptor,
and each dataset image is matched to the most similar frame among the videos of
the same split and class. Only the videos of the same split and class are
compared, so a mistake can only confuse videos of the same machine.

It uses the CPU only and can run while the GPU is training.

Usage (Anaconda Prompt):
    conda activate cnn_transformer
    cd C:\\proyectos\\AddSubLab-8
    python src\\map_frames_to_videos.py ^
        --videos "C:\\proyectos\\34_Add_Sub_Man_Mac_videos" ^
        --dataset "C:\\proyectos\\34_Add_Sub_Man_Mac\\dataset_publico\\AddSubLab-8"

Output (results/video_mapping/):
    image_to_video.csv     one row per dataset image: video, frame number and
                           match quality
    video_summary.csv      one row per video: frames, duration, images assigned
    frame_counts.csv       frames of every video, as counted by OpenCV
    report.txt             checks (totals, ambiguous matches, unused videos)
Send the whole folder (it is small).
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SPLITS = ["train", "validation", "test"]
SIDE = 32
VIDEO_EXT = {".mp4", ".mov", ".avi", ".m4v", ".3gp", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def descriptor(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    v = cv2.resize(g, (SIDE, SIDE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)       # works with non-ASCII paths
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    desc = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        desc.append(descriptor(frame))
    cap.release()
    return (np.stack(desc) if desc else np.zeros((0, SIDE * SIDE), np.float32)), fps


def load_classes(dataset):
    rows = list(csv.DictReader(open(dataset / "metadata" / "classes.csv", encoding="utf-8")))
    return {int(r["class_id"]): r["class_folder"] for r in rows}


def rotations(img):
    return {0: img, 90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(img, cv2.ROTATE_180), 270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with train/validation/test/clase_N")
    ap.add_argument("--dataset", required=True, help="AddSubLab-8 folder (train/validation/test)")
    ap.add_argument("--output", default=None, help="default: <repo>/results/video_mapping")
    a = ap.parse_args()
    videos_root, dataset = Path(a.videos), Path(a.dataset)
    out = Path(a.output) if a.output else Path(__file__).resolve().parent.parent / "results" / "video_mapping"
    out.mkdir(parents=True, exist_ok=True)
    classes = load_classes(dataset)
    print(f"Videos : {videos_root}\nDataset: {dataset}\nOutput : {out}\n")

    img_rows, vid_rows, count_rows, report = [], [], [], []
    total_frames, t0 = 0, time.time()
    for split in SPLITS:
        for cid, cfolder in sorted(classes.items()):
            vdir = videos_root / split / f"clase_{cid}"
            idir = dataset / split / cfolder
            vids = sorted([p for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXT], key=natural_key) \
                if vdir.exists() else []
            imgs = sorted([p for p in idir.iterdir() if p.suffix.lower() in IMAGE_EXT], key=natural_key) \
                if idir.exists() else []
            tag = f"{split}/clase_{cid} ({cfolder})"
            if not vids or not imgs:
                msg = f"WARNING {tag}: {len(vids)} videos, {len(imgs)} images - skipped"
                print(msg); report.append(msg)
                continue
            print(f"{tag}: {len(vids)} videos ({vids[0].name} ... {vids[-1].name}), {len(imgs)} images")

            # Decode all frames of the videos of this split and class.
            F, owner, local = [], [], []
            for k, v in enumerate(vids):
                d, fps = decode_video(v)
                if d is None:
                    msg = f"ERROR could not open {v}"
                    print("  " + msg); report.append(msg)
                    d, fps = np.zeros((0, SIDE * SIDE), np.float32), 0.0
                n = len(d)
                total_frames += n
                count_rows.append(dict(split=split, class_id=cid, class_folder=cfolder, video=v.name,
                                       frames=n, fps=round(fps, 3),
                                       duration_s=round(n / fps, 2) if fps else "",
                                       first_global_frame=total_frames - n + 1))
                F.append(d); owner += [k] * n; local += list(range(1, n + 1))
            F = np.concatenate(F); owner = np.array(owner); local = np.array(local)
            print(f"  {len(F)} frames decoded | elapsed {time.time() - t0:.0f} s")

            # Orientation check on a sample of images (the frames are portrait).
            sample = imgs[:: max(1, len(imgs) // 20)]
            best_rot, best_med = 0, -2
            for rot in (0, 90, 180, 270):
                D = np.stack([descriptor(rotations(read_image(p))[rot]) for p in sample])
                med = float(np.median((D @ F.T).max(1)))
                if med > best_med:
                    best_rot, best_med = rot, med
            if best_rot:
                report.append(f"NOTE {tag}: images matched after rotating {best_rot} degrees")

            # Match every image.
            D = np.stack([descriptor(rotations(read_image(p))[best_rot]) for p in imgs])
            S = D @ F.T                                   # correlation, -1..1
            j = S.argmax(1)
            best = S[np.arange(len(imgs)), j]
            vid_best = owner[j]
            other = np.where(owner[None, :] == vid_best[:, None], -2.0, S).max(1)
            for i, p in enumerate(imgs):
                img_rows.append(dict(split=split, class_id=cid, class_folder=cfolder,
                                     file=f"{split}/{cfolder}/{p.name}", image_index=i,
                                     video=vids[vid_best[i]].name, frame_in_video=int(local[j[i]]),
                                     correlation=round(float(best[i]), 4),
                                     best_other_video=round(float(other[i]), 4),
                                     margin=round(float(best[i] - other[i]), 4)))
            used = np.bincount(vid_best, minlength=len(vids))
            for k, v in enumerate(vids):
                vid_rows.append(dict(split=split, class_id=cid, class_folder=cfolder, video=v.name,
                                     frames=int((owner == k).sum()), images=int(used[k])))
            unused = [vids[k].name for k in range(len(vids)) if used[k] == 0]
            weak = int((best < 0.95).sum()); amb = int((best - other < 0.02).sum())
            order = np.argsort(np.argsort(owner[j] * 10**7 + local[j]))
            mono = float(np.corrcoef(order, np.arange(len(imgs)))[0, 1])
            msg = (f"{tag}: {len(vids)} videos, {len(imgs)} images | median correlation "
                   f"{np.median(best):.4f} | correlation<0.95: {weak} | ambiguous (margin<0.02): {amb} "
                   f"| videos without images: {len(unused)} {unused if unused else ''} "
                   f"| order agreement image-number vs video/frame: {mono:.3f}")
            print("  " + msg); report.append(msg)
            del F, S

    def write(name, rows):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    write("image_to_video.csv", img_rows)
    write("video_summary.csv", vid_rows)
    write("frame_counts.csv", count_rows)
    report.insert(0, f"Total frames decoded: {total_frames} (article: 179,317) | "
                     f"videos: {len(count_rows)} (article: 247) | images: {len(img_rows)} (article: 36,000)")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + report[0])
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Send the folder {out}")


if __name__ == "__main__":
    sys.exit(main())

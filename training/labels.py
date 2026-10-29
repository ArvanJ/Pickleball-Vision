"""Step 2: turn teacher candidates into clean ball labels and a YOLO dataset.

Ball candidates are linked into tracks across frames; only tracks that persist and move like a
ball are kept (logos, shoes and heads don't), and short gaps inside a track are filled, so the
dataset also has balls the teacher missed. Train/val are split by time to avoid leakage.

Usage: python training/labels.py training/data/ppa [--val-frac 0.2 --stride 2]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pbv.tracking import DEVICE

CLASSES = {0: "person", 1: "pickleball", 2: "paddle"}  # student classes (one model for the whole pipeline)
TEACHER_TO_STUDENT = {0: 0, 38: 2}                       # COCO person -> 0, tennis racket -> 2


def link_tracks(frames, max_step=28.0, max_gap=3):
    """Greedy tracklets: each candidate joins the track whose last point is closest (within reach)."""
    tracks, live = [], []
    for i, fr in enumerate(frames):
        if not fr["main"]:
            live = []  # camera cut: tracks can't continue across it
            continue
        live = [t for t in live if i - t[-1][0] <= max_gap + 1]
        used = set()
        for x, y, w, h, c in sorted(fr["c"], key=lambda c: -c[4]):
            best, bd = None, None
            for k, t in enumerate(live):
                if k in used:
                    continue
                j, px, py = t[-1][:3]
                d = np.hypot(x - px, y - py)
                if d <= max_step * (i - j) and (bd is None or d < bd):
                    best, bd = k, d
            if best is None:
                live.append([(i, x, y, w, h, c)])
                tracks.append(live[-1])
                used.add(len(live) - 1)
            else:
                live[best].append((i, x, y, w, h, c))
                used.add(best)
    return tracks


def ball_tracks(tracks, min_len=5, min_move=15.0, min_conf=0.12):
    """Keep tracks that persist, move, and are confident on average."""
    keep = []
    for t in tracks:
        a = np.array(t)
        span = np.ptp(a[:, 1:3], axis=0).max() if len(a) > 1 else 0
        if len(t) >= min_len and span >= min_move and a[:, 5].mean() >= min_conf:
            keep.append(t)
    return keep


def labels_from_tracks(tracks):
    """frame -> (x, y, size, conf, interpolated); gaps inside a track are filled linearly."""
    out = {}
    for t in tracks:
        a = np.array(t)
        size = float(np.clip(np.median(a[:, 3:5]), 4, 14))
        score = float(a[:, 5].mean())
        for (i0, x0, y0, *_), (i1, x1, y1, *_) in zip(t, t[1:]):
            for i in range(int(i0), int(i1)):
                u = (i - i0) / (i1 - i0)
                cand = (x0 + u * (x1 - x0), y0 + u * (y1 - y0), size, score, i != i0)
                if i not in out or out[i][3] < score:  # overlapping tracks: keep the more confident
                    out[i] = cand
        i, x, y = int(t[-1][0]), t[-1][1], t[-1][2]
        if i not in out or out[i][3] < score:
            out[i] = (x, y, size, score, False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth labelled frame (neighbours are near-duplicates)")
    ap.add_argument("--teacher", default="yolov8m.pt", help="labels people and paddles on the exported frames")
    args = ap.parse_args()

    cand = json.loads((args.data / "candidates.json").read_text())
    meta, frames = cand["meta"], cand["frames"]
    tracks = link_tracks(frames)
    kept = ball_tracks(tracks)
    labels = labels_from_tracks(kept)
    n_main = sum(f["main"] for f in frames)
    n_det = sum(1 for i, f in enumerate(frames) if f["main"] and f["c"])
    print(f"main-camera frames {n_main}; teacher had candidates in {n_det}; "
          f"{len(tracks)} tracks -> {len(kept)} ball tracks -> {len(labels)} labelled frames "
          f"({sum(v[4] for v in labels.values())} filled by interpolation)")
    (args.data / "labels.json").write_text(json.dumps({"meta": meta, "labels": {str(k): v for k, v in labels.items()}}))

    # YOLO dataset: images + one "pickleball" box per labelled frame
    split_at = int(len(frames) * (1 - args.val_frac))
    W, H = meta["width"], meta["height"]
    root = args.data / "yolo"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    keep = sorted(labels)[::args.stride]
    # A few main-camera frames with no candidates nearby (ball out of play) as negatives.
    labelled = set(labels)
    negs = [i for i, f in enumerate(frames) if f["main"] and not any(j in labelled for j in range(i - 25, i + 26))]
    random.Random(0).shuffle(negs)
    want = {i: labels[i] for i in keep} | {i: None for i in negs[: len(keep) // 10]}
    cap = cv2.VideoCapture(meta["video"])
    teacher = YOLO(args.teacher)
    counts = {"train": 0, "val": 0}
    for i in range(max(want) + 1):  # one sequential pass: far faster than seeking per frame
        ok, img = cap.read()
        if not ok:
            break
        if i not in want:
            continue
        split = "val" if i >= split_at else "train"
        name = f"{args.data.name}_{i:06d}"
        cv2.imwrite(str(root / f"images/{split}/{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = want[i]
        rows = [] if lab is None else [f"1 {lab[0] / W:.6f} {lab[1] / H:.6f} {lab[2] / W:.6f} {lab[2] / H:.6f}"]
        r = teacher.predict(img, classes=list(TEACHER_TO_STUDENT), conf=0.35, imgsz=640, device=DEVICE, verbose=False)[0]
        for k, (x, y, w, h) in zip(r.boxes.cls.int().tolist(), r.boxes.xywhn.tolist()):
            rows.append(f"{TEACHER_TO_STUDENT[k]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        (root / f"labels/{split}/{name}.txt").write_text("".join(row + "\n" for row in rows))
        counts[split] += 1
    names = "".join(f"  {k}: {v}\n" for k, v in CLASSES.items())
    (root / "data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    print(f"dataset: {counts['train']} train / {counts['val']} val images -> {root / 'data.yaml'}")


if __name__ == "__main__":
    main()

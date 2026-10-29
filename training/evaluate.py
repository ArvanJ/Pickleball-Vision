"""Step 4: compare ball detectors on the held-out (later-in-time) main-camera frames.

Against the pseudo-labels: recall = labelled frames where the model's best ball is within
--radius px of the label; false positives = balls reported on frames with no ball nearby in time.
Pseudo-labels are imperfect, so this is backed by a visual sheet (--sheet) checked by eye.

Usage: python training/evaluate.py training/data/ppa training/runs/ppa-v1/weights/best.pt [--sheet out.jpg]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pbv.tracking import DEVICE


def best_ball(model, img, cls, imgsz, conf=0.15):
    r = model.predict(img, classes=[cls], conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    if not len(r.boxes):
        return None
    return r.boxes.xywh[int(r.boxes.conf.argmax())][:2].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("student", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=400, help="frames to test of each kind")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--sheet", type=Path)
    args = ap.parse_args()

    cand = json.loads((args.data / "candidates.json").read_text())
    lab = {int(k): v for k, v in json.loads((args.data / "labels.json").read_text())["labels"].items()}
    frames, meta = cand["frames"], cand["meta"]
    split_at = int(len(frames) * (1 - args.val_frac))
    pos = [i for i in lab if i >= split_at]
    neg = [i for i in range(split_at, len(frames)) if frames[i]["main"] and not any(j in lab for j in range(i - 25, i + 26))]
    rnd = random.Random(1)
    pos, neg = sorted(rnd.sample(pos, min(args.n, len(pos)))), sorted(rnd.sample(neg, min(args.n // 2, len(neg))))

    models = {"COCO yolov8n @640 (current web)": (YOLO("yolov8n.pt"), 32, 640),
              "COCO yolov8m @1280 (teacher)": (YOLO("yolov8m.pt"), 32, 1280),
              "trained yolov8n @640": (YOLO(str(args.student)), 1, 640),
              "trained yolov8n @1280": (YOLO(str(args.student)), 1, 1280)}
    cap = cv2.VideoCapture(meta["video"])
    imgs = {}
    for i in range(max(pos + neg) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if i in pos or i in neg:
            imgs[i] = img
    print(f"held-out: {len(pos)} frames with a ball label, {len(neg)} frames with no ball nearby")
    preds = {}
    for name, (m, cls, sz) in models.items():
        p = {i: best_ball(m, imgs[i], cls, sz) for i in pos + neg}
        preds[name] = p
        hit = sum(1 for i in pos if p[i] and np.hypot(p[i][0] - lab[i][0], p[i][1] - lab[i][1]) <= args.radius)
        fp = sum(1 for i in neg if p[i])
        print(f"{name:34s} recall {100 * hit / len(pos):5.1f}%   false alarms on empty frames {100 * fp / len(neg):5.1f}%")

    if args.sheet:  # random held-out frames, crop around the label: green = label, yellow = trained, red = current web
        tiles = []
        for i in rnd.sample(pos, min(24, len(pos))):
            img = imgs[i].copy()
            x, y = lab[i][:2]
            cv2.circle(img, (int(x), int(y)), 9, (0, 255, 0), 1)
            for name, col in (("trained yolov8n @640", (0, 255, 255)), ("COCO yolov8n @640 (current web)", (0, 0, 255))):
                q = preds[name][i]
                if q:
                    cv2.drawMarker(img, (int(q[0]), int(q[1])), col, cv2.MARKER_CROSS, 14, 1)
            x0, y0 = int(np.clip(x - 80, 0, img.shape[1] - 160)), int(np.clip(y - 45, 0, img.shape[0] - 90))
            tiles.append(cv2.resize(img[y0:y0 + 90, x0:x0 + 160], (320, 180), interpolation=cv2.INTER_NEAREST))
        rows = [np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles) - len(tiles) % 6, 6)]
        cv2.imwrite(str(args.sheet), np.vstack(rows))


if __name__ == "__main__":
    main()

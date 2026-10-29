"""Step 1 of ball-detector training: scan a broadcast video with a big "teacher" model.

For every frame: is it the main (fixed, high behind-baseline) camera, and which ball candidates
(COCO "sports ball") does the teacher see? Writes training/data/<video>/candidates.json.

Usage: python training/scan.py PICKLEBALLPOV/ppa.mp4 [--teacher yolov8m.pt --imgsz 1280]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from pbv.tracking import DEVICE


def thumb(frame):
    """Small blurred greyscale of the picture below the scoreboard, for comparing camera views."""
    h = frame.shape[0]
    g = cv2.cvtColor(cv2.resize(frame[h // 4:], (80, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0).astype("float32")


def is_main_camera(frame, ref, max_diff=18.0):
    """Main camera = a fixed view, so it matches a reference main-camera frame (players aside)."""
    return float(abs(thumb(frame) - ref).mean()) < max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--teacher", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--ref", type=float, default=186.0, help="time (s) of a main-camera frame to match against")
    args = ap.parse_args()

    out = Path("training/data") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.teacher)
    cap = cv2.VideoCapture(str(args.video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.ref * fps))
    ref = thumb(cap.read()[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames, t0 = [], time.time()
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        main_cam = is_main_camera(f, ref)
        cands = []
        if main_cam:
            r = model.predict(f, classes=[32], conf=args.conf, imgsz=args.imgsz, device=DEVICE, verbose=False)[0]
            cands = [[round(v, 1) for v in b] + [round(float(c), 3)]
                     for b, c in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist())][:5]
        frames.append({"main": main_cam, "c": cands})
        if i % 2500 == 0:
            print(f"{i}/{n} frames, {time.time() - t0:.0f}s", flush=True)
    meta = {"video": str(args.video), "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
            "teacher": args.teacher, "imgsz": args.imgsz, "conf": args.conf}
    (out / "candidates.json").write_text(json.dumps({"meta": meta, "frames": frames}))
    print(f"done: {sum(f['main'] for f in frames)} main-camera frames, "
          f"{sum(bool(f['c']) for f in frames)} with candidates -> {out / 'candidates.json'}")


if __name__ == "__main__":
    main()

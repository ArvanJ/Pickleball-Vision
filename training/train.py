"""Step 3: fine-tune YOLOv8n on the pseudo-labelled dataset and export it for the pipeline and website.

Usage: python training/train.py training/data/ppa/yolo/data.yaml [--epochs 40 --name ppa-v1]
Outputs training/runs/<name>/weights/best.pt plus best-640.onnx / best-1280.onnx.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from pbv.tracking import DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="ppa-v1")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data), imgsz=640, epochs=args.epochs, batch=args.batch, device=DEVICE or "cpu",
        project=str(Path("training/runs").resolve()), name=args.name, exist_ok=True, patience=12, workers=2,
        # The ball is ~3-8 px: keep zoom-out augmentation mild so it doesn't vanish; no vertical flips.
        scale=0.25, mosaic=0.5, close_mosaic=8, degrees=0.0, flipud=0.0, fliplr=0.5, plots=False,
    )
    best = Path("training/runs") / args.name / "weights" / "best.pt"
    for size in (640, 1280):
        out = YOLO(str(best)).export(format="onnx", imgsz=size, simplify=True, opset=17)
        shutil.move(out, best.with_name(f"best-{size}.onnx"))
    print(f"done: {best}")


if __name__ == "__main__":
    main()

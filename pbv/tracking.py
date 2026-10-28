"""YOLO-based detectors: players (ByteTrack IDs), ball, and pose."""
import numpy as np
import torch
from ultralytics import YOLO

PERSON, SPORTS_BALL, RACKET = 0, 32, 38  # COCO class ids; pickleball paddles detect as "tennis racket"
DEVICE = "mps" if torch.backends.mps.is_available() else None  # None = ultralytics picks CUDA/CPU


def is_own_body(box, w, h, margin=4):
    """POV: the wearer's arms/legs/hands show up as person boxes pinned to 2+ frame edges,
    or as short boxes sitting on the bottom edge (a hand holding the paddle)."""
    x1, y1, x2, y2 = box
    edges = (x1 <= margin) + (y1 <= margin) + (x2 >= w - margin) + (y2 >= h - margin)
    return edges >= 2 or (y2 >= h - margin and y2 - y1 < 0.25 * h)


class PlayerTracker:
    def __init__(self, model="yolov8m.pt", conf=0.3):
        self.model, self.conf = YOLO(model), conf

    def track(self, frame, court=None, max_players=4):
        """-> {player_id: {"bbox": [x1,y1,x2,y2], "court": [x_m, y_m]}} for people on the court.

        court=None is POV mode: no court position, the wearer's own body is dropped, and
        the 3 biggest remaining people (partner + opponents) are kept.
        """
        r = self.model.track(frame, persist=True, classes=[PERSON], conf=self.conf,
                             tracker="bytetrack.yaml", device=DEVICE, verbose=False)[0]
        if r.boxes.id is None:
            return {}
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        if court is None:
            h, w = frame.shape[:2]
            keep = [i for i in range(len(ids)) if not is_own_body(boxes[i], w, h)]
            max_players = min(max_players, 3)
        else:
            feet = court.to_court(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
            # Spectators/refs stand off-court; keep the biggest boxes near the court.
            keep = [i for i in range(len(ids)) if court.inside(feet[i], margin=2.0)]
        keep.sort(key=lambda i: -(boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        out = {str(ids[i]): {"bbox": boxes[i].round(1).tolist()} for i in keep[:max_players]}
        if court is not None:
            for i in keep[:max_players]:
                out[str(ids[i])]["court"] = feet[i].round(3).tolist()
        return out


class BallDetector:
    # shortcut: COCO "sports ball" misses many small/blurred pickleballs; pass a ball model
    # trained on pickleball frames (or swap in TrackNet) via --ball-model/--ball-class for real matches.
    def __init__(self, model="yolov8m.pt", cls=SPORTS_BALL, conf=0.15, imgsz=1280):
        self.model = YOLO(model)
        self.paddles = cls == SPORTS_BALL  # the COCO model also finds paddles in the same pass
        self.classes = None if cls < 0 else [cls] + ([RACKET] if self.paddles else [])
        self.conf, self.imgsz = conf, imgsz

    def detect(self, frame):
        """-> ([x, y] centre of the most confident ball or None, tallest paddle box or None)."""
        r = self.model.predict(frame, classes=self.classes, conf=self.conf, imgsz=self.imgsz,
                               device=DEVICE, verbose=False)[0]
        boxes, cls, conf = r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()
        paddle = [b for b, k in zip(boxes, cls) if self.paddles and k == RACKET]
        balls = [(c, b) for b, k, c in zip(boxes, cls, conf) if not (self.paddles and k == RACKET)]
        ball = None
        if balls:
            x1, y1, x2, y2 = max(balls)[1]
            ball = [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]
        paddle = max(paddle, key=lambda b: b[3] - b[1], default=None)
        return ball, None if paddle is None else [round(v, 1) for v in paddle]


class PoseEstimator:
    def __init__(self, model="yolov8n-pose.pt"):
        self.model = YOLO(model)

    def keypoints_for(self, frame, bbox):
        """-> 17x2 COCO keypoints of the pose closest to bbox, or None."""
        r = self.model.predict(frame, device=DEVICE, verbose=False)[0]
        if r.keypoints is None or not len(r.boxes):
            return None
        centres = r.boxes.xywh[:, :2].cpu().numpy()
        c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        return r.keypoints.xy[int(np.argmin(np.linalg.norm(centres - c, axis=1)))].cpu().numpy()

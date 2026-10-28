"""Homography between the camera image and the flat court plane."""
import json

import cv2
import numpy as np

from .constants import COURT_L, COURT_W, KITCHEN, NET_Y

# Order matters: far-left, far-right, near-right, near-left.
CORNERS_M = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])

# Line intersections a user can click when some corners are out of frame or hidden by the net.
LANDMARKS = {
    "far-left": (0, 0), "far-right": (COURT_W, 0), "near-right": (COURT_W, COURT_L), "near-left": (0, COURT_L),
    "near-baseline-center": (COURT_W / 2, COURT_L), "far-baseline-center": (COURT_W / 2, 0),
    "near-kitchen-left": (0, NET_Y + KITCHEN), "near-kitchen-center": (COURT_W / 2, NET_Y + KITCHEN),
    "near-kitchen-right": (COURT_W, NET_Y + KITCHEN),
    "far-kitchen-left": (0, NET_Y - KITCHEN), "far-kitchen-center": (COURT_W / 2, NET_Y - KITCHEN),
    "far-kitchen-right": (COURT_W, NET_Y - KITCHEN),
}


class Court:
    # shortcut: hand-clicked landmarks; upgrade to a court-keypoint model + RANSAC
    # (as Badminton-VisionAI does) for automatic calibration.
    def __init__(self, points_px, points_m=CORNERS_M):
        """points_px[i] is where court point points_m[i] (metres) appears in the image; 4+ points."""
        self.points_px = np.float32(points_px).reshape(-1, 2)
        self.points_m = np.float32(points_m).reshape(-1, 2)
        if len(self.points_px) < 4 or len(self.points_px) != len(self.points_m):
            raise ValueError("need 4+ matching image/court points")
        self.H, _ = cv2.findHomography(self.points_px, self.points_m)
        if self.H is None:
            raise ValueError("court points are degenerate (3 or more on one line?)")
        self.H_inv = np.linalg.inv(self.H)
        self.corners_px = self.to_image(CORNERS_M)  # court outline, even if corners are off-screen

    @classmethod
    def from_landmarks(cls, named_px):
        """{"near-left": (x, y), "near-kitchen-center": (x, y), ...} -> Court."""
        unknown = set(named_px) - set(LANDMARKS)
        if unknown:
            raise ValueError(f"unknown landmarks {sorted(unknown)}; choose from {list(LANDMARKS)}")
        return cls(list(named_px.values()), [LANDMARKS[n] for n in named_px])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["corners"]) if "corners" in d else cls(d["points_px"], d["points_m"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"points_px": self.points_px.tolist(), "points_m": self.points_m.tolist()}, f)

    @staticmethod
    def _apply(H, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def to_court(self, pts_px):
        """Image pixels -> court metres. Only valid for points on the ground (feet, bounces)."""
        return self._apply(self.H, pts_px)

    def to_image(self, pts_m):
        return self._apply(self.H_inv, pts_m)

    @staticmethod
    def inside(p, margin=0.0):
        return -margin <= p[0] <= COURT_W + margin and -margin <= p[1] <= COURT_L + margin

    @staticmethod
    def in_kitchen(p):
        return 0 <= p[0] <= COURT_W and abs(p[1] - NET_Y) <= KITCHEN

    @staticmethod
    def side(p):
        return "far" if p[1] < NET_Y else "near"

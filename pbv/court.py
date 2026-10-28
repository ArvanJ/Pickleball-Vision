"""Homography between the camera image and the flat court plane."""
import json

import cv2
import numpy as np

from .constants import COURT_L, COURT_W, KITCHEN, NET_Y

# Order matters: far-left, far-right, near-right, near-left.
CORNERS_M = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])


class Court:
    # shortcut: 4 hand-clicked corners, so the whole court must be in frame;
    # upgrade to a keypoint model + RANSAC (as Badminton-VisionAI does) for auto-calibration.
    def __init__(self, corners_px):
        self.corners_px = np.float32(corners_px).reshape(4, 2)
        self.H = cv2.getPerspectiveTransform(self.corners_px, CORNERS_M)
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["corners"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"corners": self.corners_px.tolist()}, f)

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

"""Frame annotation, mini-court overlay and heatmaps."""
import cv2
import numpy as np

from .constants import COURT_L, COURT_W, KITCHEN, NET_Y

FONT = cv2.FONT_HERSHEY_SIMPLEX
BALL = (0, 255, 255)
PALETTE = [(245, 135, 66), (96, 66, 245), (135, 245, 66), (66, 200, 245), (245, 66, 200)]


def name(pid):
    return "You" if pid == "me" else f"P{pid}"


def color(pid):
    return PALETTE[int(pid) % len(PALETTE)] if str(pid).isdigit() else PALETTE[0]


def court_canvas(scale=25, pad=15):
    """Top-down court image and a metres -> pixel mapper."""
    w, h = int(COURT_W * scale) + 2 * pad, int(COURT_L * scale) + 2 * pad
    img = np.full((h, w, 3), (70, 120, 60), np.uint8)
    return img, lambda x, y: (int(pad + x * scale), int(pad + y * scale))


def court_lines(img, P):
    white = (255, 255, 255)
    cv2.rectangle(img, P(0, 0), P(COURT_W, COURT_L), white, 2)
    for y in (NET_Y - KITCHEN, NET_Y + KITCHEN):
        cv2.line(img, P(0, y), P(COURT_W, y), white, 1)
    cv2.line(img, P(COURT_W / 2, 0), P(COURT_W / 2, NET_Y - KITCHEN), white, 1)
    cv2.line(img, P(COURT_W / 2, NET_Y + KITCHEN), P(COURT_W / 2, COURT_L), white, 1)
    cv2.line(img, P(-0.3, NET_Y), P(COURT_W + 0.3, NET_Y), (210, 210, 210), 3)
    return img


def heatmap_image(positions, scale=25, pad=15):
    img, P = court_canvas(scale, pad)
    acc = np.zeros(img.shape[:2], np.float32)
    for x, y in positions:
        px, py = P(x, y)
        if 0 <= px < acc.shape[1] and 0 <= py < acc.shape[0]:
            acc[py, px] += 1
    if acc.any():
        acc = cv2.GaussianBlur(acc, (0, 0), scale * 0.6)
        acc = (255 * acc / acc.max()).astype(np.uint8)
        heat = cv2.applyColorMap(acc, cv2.COLORMAP_JET)
        mask = acc > 12
        img[mask] = cv2.addWeighted(img, 0.35, heat, 0.65, 0)[mask]
    return court_lines(img, P)


def annotate(frame, frame_players, trail, court, bounces, label=""):
    """Boxes + IDs, ball trail, court outline + mini-court (fixed mode only), current shot label."""
    if court is not None:
        cv2.polylines(frame, [court.corners_px.astype(np.int32)], True, (255, 255, 255), 1)
    for pid, p in frame_players.items():
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color(pid), 2)
        cv2.putText(frame, name(pid), (x1, y1 - 6), FONT, 0.6, color(pid), 2)
    pts = [tuple(map(int, p)) for p in trail if not np.isnan(p[0])]
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, BALL, 2)
    if pts:
        cv2.circle(frame, pts[-1], 5, BALL, -1)

    if court is not None:
        mini, P = court_canvas(scale=12, pad=8)
        court_lines(mini, P)
        for b in bounces:
            cv2.circle(mini, P(*b), 3, BALL, -1)
        for pid, p in frame_players.items():
            cv2.circle(mini, P(*p["court"]), 5, color(pid), -1)
        h, w = mini.shape[:2]
        H, W = frame.shape[:2]
        if h + 10 < H and w + 10 < W:
            frame[10:10 + h, W - w - 10:W - 10] = mini
    if label:  # scaled so it stays readable on 4K
        s = frame.shape[1] / 1280
        cv2.putText(frame, label, (int(20 * s), int(40 * s)), FONT, s, (0, 0, 0), int(4 * s))
        cv2.putText(frame, label, (int(20 * s), int(40 * s)), FONT, s, (255, 255, 255), max(2, int(2 * s)))
    return frame

"""Click the 4 court corners to calibrate the homography.

Order: far-left, far-right, near-right, near-left. Enter saves, r resets, Esc aborts.
Usage: python calibrate.py match.mp4 outputs/match/court.json [--frame 0]
"""
import argparse
import sys

import cv2

from pbv.court import Court

ORDER = ["far-left", "far-right", "near-right", "near-left"]


def calibrate(video, dst, frame_no=0):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read frame {frame_no} of {video}")
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "PickleballVision - court calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        img = frame.copy()
        for name, p in zip(ORDER, pts):
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        msg = f"Click {ORDER[len(pts)]} corner" if len(pts) < 4 else "Enter = save, r = reset"
        cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            sys.exit("calibration aborted")
        if key == ord("r"):
            pts.clear()
        if key in (10, 13) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    Court(pts).save(dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("dst")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    calibrate(a.video, a.dst, a.frame)

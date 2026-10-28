"""Click court landmarks to calibrate the homography.

Landmarks are asked for one at a time (corners first). Click it, or press s to skip one that
isn't visible (e.g. a corner off-screen or behind the net). Needs 4+ landmarks, not 3 on one line.
Enter saves, u undoes the last point, r resets, Esc aborts.
Usage: python calibrate.py match.mp4 outputs/match/court.json [--frame 0]
"""
import argparse
import sys

import cv2

from pbv.court import LANDMARKS, Court


def calibrate(video, dst, frame_no=0):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read frame {frame_no} of {video}")
    names, picked, i = list(LANDMARKS), {}, [0]  # i: index of the landmark being asked for

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and i[0] < len(names):
            picked[names[i[0]]] = (x, y)
            i[0] += 1

    win = "Pickleball-Vision - court calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    scale = frame.shape[1] / 1280
    while True:
        img = frame.copy()
        for name, p in picked.items():
            cv2.circle(img, p, int(6 * scale), (0, 0, 255), -1)
            cv2.putText(img, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), 2)
        msg = (f"Click {names[i[0]]} (s = skip)" if i[0] < len(names) else "All landmarks asked") + \
              (f"  |  {len(picked)} points: Enter = save, u = undo" if len(picked) >= 4 else "")
        cv2.putText(img, msg, (20, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 255, 255), 2)
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            sys.exit("calibration aborted")
        if key == ord("s") and i[0] < len(names):
            i[0] += 1
        if key == ord("u") and picked:
            last = list(picked)[-1]
            del picked[last]
            i[0] = names.index(last)
        if key == ord("r"):
            picked.clear()
            i[0] = 0
        if key in (10, 13) and len(picked) >= 4:
            try:
                court = Court.from_landmarks(picked)
                break
            except ValueError as e:
                print(e)
    cv2.destroyAllWindows()
    court.save(dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("dst")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    calibrate(a.video, a.dst, a.frame)

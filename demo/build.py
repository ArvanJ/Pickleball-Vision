"""Build the replay demo page from pipeline outputs.

Usage: python demo/build.py demo/replay.html OUT_DIR[:label] [OUT_DIR[:label] ...]
Each OUT_DIR is an app.py output folder (stats.json + tracking.json). Frames are thinned to
<= 30 fps to keep the page small. Only detections go into the page, never video pixels.
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv.constants import COURT_L, COURT_W, KITCHEN, NET_Y  # noqa: E402
from pbv.court import Court  # noqa: E402

COURT_LINES_M = [((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_L)), ((COURT_W, COURT_L), (0, COURT_L)),
                 ((0, COURT_L), (0, 0)), ((0, NET_Y - KITCHEN), (COURT_W, NET_Y - KITCHEN)),
                 ((0, NET_Y + KITCHEN), (COURT_W, NET_Y + KITCHEN)), ((-0.3, NET_Y), (COURT_W + 0.3, NET_Y)),
                 ((COURT_W / 2, 0), (COURT_W / 2, NET_Y - KITCHEN)), ((COURT_W / 2, NET_Y + KITCHEN), (COURT_W / 2, COURT_L))]


def dataset(out, label):
    s = json.loads((out / "stats.json").read_text())
    t = json.loads((out / "tracking.json").read_text())
    step = max(1, round(s["fps"] / 30))
    cap = cv2.VideoCapture(s["video"])
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    court = Court.load(out / "court.json") if s["mode"] == "fixed" else None

    frames = []
    for i in range(0, len(t["players"]), step):
        players = [[pid, *[round(v) for v in pl["bbox"]], *([round(c, 2) for c in pl["court"]] if court else [])]
                   for pid, pl in t["players"][i].items()]
        b = t["ball"][i]
        frames.append({"p": players, "b": None if b is None else [round(v) for v in b]})
    bounces = []
    for e in s["events"]:
        if e["type"] == "bounce":
            x, y = (court.to_court([e["px"]])[0].round(2).tolist() if court else (None, None))
            bounces.append({"f": e["frame"] // step, "x": x, "y": y})
    lines = []
    if court:
        for a, b in COURT_LINES_M:
            (x1, y1), (x2, y2) = court.to_image([a, b]).round().tolist()
            lines.append([x1, y1, x2, y2])
    shots = [{"f": x["frame"] // step, "player": x["player"], "type": x["type"], "volley": x["volley"],
              "kmh": x["speed_kmh"], "faults": x["faults"], "form": x.get("form")} for x in s["shots"]]
    seen = sum(b is not None for b in t["ball"])
    return {
        "id": out.name.lower().replace("_", "-"), "label": label, "mode": s["mode"],
        "fps": s["fps"] / step, "start_s": s.get("start_s", 0), "W": W, "H": H, "court": bool(court),
        "frames": frames, "lines": lines, "bounces": bounces, "shots": shots,
        "rallies": [{"start": r["start"] // step, "end": r["end"] // step} for r in s["rallies"]],
        "players": s["players"], "shot_summary": s["shot_summary"],
        "ball_pct": round(100 * seen / max(len(t["ball"]), 1)),
    }


def main():
    dst, specs = Path(sys.argv[1]), sys.argv[2:]
    meta = json.loads((Path(__file__).parent / "meta.json").read_text())  # per-run subtitle + method text
    data = []
    for spec in specs:
        path, _, label = spec.partition(":")
        d = dataset(Path(path), label or Path(path).name)
        d.update(meta.get(Path(path).name, {"subtitle": "", "method": ""}))
        data.append(d)
    html = (Path(__file__).parent / "template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    dst.write_text(html)
    print(f"wrote {dst} ({len(html) / 1e6:.1f} MB, {sum(len(d['frames']) for d in data)} frames)")


if __name__ == "__main__":
    main()

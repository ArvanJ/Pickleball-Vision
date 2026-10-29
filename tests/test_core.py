"""Self-checks for the non-ML logic. Run: python -m pytest tests  (or python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv import analytics, draw, events  # noqa: E402
from pbv.constants import COURT_L, COURT_W, KITCHEN, NET_Y  # noqa: E402
from pbv.court import LANDMARKS, Court  # noqa: E402

COURT = Court([(300, 100), (700, 100), (900, 600), (100, 600)])


def test_homography_roundtrip():
    assert np.allclose(COURT.to_court([(300, 100), (900, 600)]), [[0, 0], [COURT_W, COURT_L]], atol=1e-3)
    p = np.array([[2.0, 5.0], [4.0, 11.0]])
    assert np.allclose(COURT.to_court(COURT.to_image(p)), p, atol=1e-3)
    assert Court.in_kitchen((3, NET_Y + 1)) and not Court.in_kitchen((3, NET_Y + KITCHEN + 0.5))


def test_court_from_landmarks_when_corners_hidden():
    # Only the near half is visible: rebuild the same homography from 4 non-corner landmarks.
    named = {n: COURT.to_image([LANDMARKS[n]])[0].tolist()
             for n in ("near-left", "near-baseline-center", "near-kitchen-left", "near-kitchen-center")}
    c = Court.from_landmarks(named)
    assert np.allclose(c.corners_px, COURT.corners_px, atol=0.5)
    assert np.allclose(c.to_court([(500, 300)]), COURT.to_court([(500, 300)]), atol=1e-3)
    for bad in ({"near-left": (0, 0)}, {"middle": (1, 1), **named}):
        try:
            Court.from_landmarks(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_clean_ball_fills_short_gaps_only():
    track = [[10.0 * i, 100.0] for i in range(30)]
    track[5] = track[6] = None                  # short gap -> filled
    for i in range(15, 27):                     # long gap -> left empty
        track[i] = None
    track[10] = [900.0, 900.0]                  # outlier -> dropped then interpolated
    b = events.clean_ball(track, max_gap=8)
    assert np.allclose(b[5], [50, 100]) and np.allclose(b[10], [100, 100])
    assert np.isnan(b[20]).all()


def _player(cx, cy_court, h=150):
    """Player standing at court (cx, cy_court) with an h px tall box."""
    fx, fy = COURT.to_image([[cx, cy_court]])[0]
    return {"bbox": [fx - h / 5, fy - h, fx + h / 5, fy], "court": [cx, cy_court]}


def test_find_events_hit_bounce_hit():
    near, far = _player(3, 12.5), _player(3, 1.0, h=60)  # far player is smaller on screen
    nb, fb = near["bbox"], far["bbox"]
    bounce_px = COURT.to_image([[4.5, 4.0]])[0]
    # Piecewise-linear image-y: near hit (max) -> apex (min, no player) -> bounce (max) -> far hit (min) -> down.
    keys = [(0, nb[1] + 20), (10, nb[1] + 60), (22, 60), (32, bounce_px[1]), (40, fb[1] + 40), (55, 400)]
    xs = [(0, nb[0] + 30), (10, nb[0] + 30), (22, 750), (32, bounce_px[0]), (40, fb[0] + 12), (55, 450)]
    f = np.arange(56)
    ball = np.c_[np.interp(f, *zip(*xs)), np.interp(f, *zip(*keys))]
    players = [{"1": near, "2": far} for _ in f]
    evs = events.find_events(ball, players)
    assert [(e["frame"], e["type"], e["player"]) for e in evs] == [(10, "hit", "1"), (32, "bounce", None), (40, "hit", "2")]


def test_classify_shots_types_and_faults():
    fps = 30
    pos = {"A": (3, 12.8), "B": (3, 0.5), "A2": (3, 12.8), "B2": (3, NET_Y - 1)}  # B2 at the kitchen
    frames = [{"1": _player(*pos["A"]), "2": _player(*pos["B"])} for _ in range(200)]
    for f in range(100, 200):
        frames[f]["2"] = _player(*pos["B2"])
    bpx = lambda x, y: COURT.to_image([[x, y]])[0].tolist()
    evs = [
        {"frame": 0, "type": "hit", "player": "1", "px": [0, 0]},
        {"frame": 30, "type": "bounce", "player": None, "px": bpx(3, 2.0)},      # serve lands deep
        {"frame": 40, "type": "hit", "player": "2", "px": [0, 0]},
        {"frame": 70, "type": "bounce", "player": None, "px": bpx(3, 12.0)},
        {"frame": 80, "type": "hit", "player": "1", "px": [0, 0]},               # third shot from baseline
        {"frame": 110, "type": "bounce", "player": None, "px": bpx(3, NET_Y - 1)},  # ... into the kitchen = drop
        {"frame": 120, "type": "hit", "player": "2", "px": [0, 0]},
        {"frame": 150, "type": "hit", "player": "1", "px": [0, 0]},              # volley, from baseline
        {"frame": 165, "type": "hit", "player": "2", "px": [0, 0]},              # volley from the kitchen = fault
    ]
    shots, rallies = events.classify_shots(evs, frames, COURT, fps)
    assert [s["type"] for s in shots][:3] == ["serve", "return", "drop"]
    assert shots[4]["volley"] and shots[5]["volley"]
    assert shots[5]["faults"] == ["kitchen_volley"] and shots[4]["faults"] == []
    assert len(rallies) == 1 and rallies[0]["shots"] == 6
    assert abs(shots[0]["speed_kmh"] - 10.8 / 1.0 * 3.6) < 0.5  # 10.8 m in 1 s
    summary = analytics.shot_stats(shots)
    assert summary["2"]["faults"] == 1 and summary["1"]["n_serve"] == 1


def test_player_stats_distance():
    fps = 30
    frames = [{"7": {"bbox": [0, 0, 1, 1], "court": [1.0 + f / fps, 1.0]}} for f in range(10 * fps)]  # 1 m/s
    st = analytics.player_stats(frames, fps)["7"]
    assert abs(st["distance_m"] - 10) < 0.2 and abs(st["avg_speed_ms"] - 1) < 0.05
    assert st["side"] == "far" and st["kitchen_pct"] == 0


def test_stroke_form():
    kp = np.zeros((17, 2))
    kp[5], kp[6] = (90, 100), (110, 100)       # shoulders
    kp[11], kp[12] = (95, 200), (105, 200)     # hips
    kp[7], kp[9] = (85, 150), (80, 160)        # left arm resting
    kp[8], kp[10] = (140, 150), (180, 230)     # right arm swinging low
    form = analytics.stroke_form(kp)
    assert form["arm"] == "right" and form["contact_below_waist"] and not form["overhead"]
    assert form["contact_height"] < 0                          # wrist 30 px below the hips
    kp[12] = (0, 0)
    assert analytics.stroke_form(kp) is None


def test_heatmap_renders():
    img = draw.heatmap_image([(3, 3)] * 20 + [(1, 10)] * 5)
    assert img.ndim == 3 and img.sum() > 0


# ---- POV mode ----

def test_camera_shift_and_compensation():
    import cv2
    from app import camera_shift
    rng = np.random.default_rng(0)
    a = (cv2.GaussianBlur(rng.random((540, 960)), (0, 0), 3) * 255).astype(np.uint8)
    f0 = cv2.merge([a] * 3)
    f1 = np.roll(f0, 20, axis=1)                      # scene content moves 20 px right
    g, _ = camera_shift(None, f0)
    _, shift = camera_shift(g, f1)
    assert abs(shift[0] - 20) < 1.5 and abs(shift[1]) < 1.5
    # A ball standing still in the world drifts with the pan on screen; compensation removes it.
    comp, _ = events.compensate([[100, 50], [120, 50], None], [[0, 0], [20, 0], [0, 0]])
    assert comp[0] == comp[1] == [100, 50] and comp[2] is None


def test_own_body_filter():
    from pbv.tracking import is_own_body
    assert is_own_body([900, 0, 1280, 180], 1280, 720)        # arm in the top-right corner
    assert is_own_body([300, 500, 900, 720], 1280, 720) is False  # touches only the bottom edge
    assert not is_own_body([830, 330, 880, 450], 1280, 720)   # opponent mid-frame
    assert is_own_body([650, 610, 740, 720], 1280, 720)       # own hand on the paddle, bottom edge
    assert not is_own_body([980, 290, 1075, 720], 1280, 720)  # partner standing close, feet cut off


def test_find_swings_needs_ball_and_own_paddle():
    fps, n = 30, 120
    cam = [[2.0, 0.0]] * n
    cam[30] = [150.0, 0.0]                            # swing jolt, ball visible
    cam[60] = [150.0, 0.0]                            # head turn, ball visible, no own paddle
    cam[100] = [150.0, 0.0]                           # walking jolt, no ball nearby
    ball = np.full((n, 2), np.nan)
    ball[20:70] = 100.0
    assert events.find_swings(cam, ball, fps, width=1920) == [30, 60]       # no paddle data: jolts only
    paddles = [None] * n
    paddles[28] = [1500, 100, 1800, 700]              # own paddle: 600 px of a 1080 px frame
    paddles[58] = [900, 500, 930, 560]                # someone else's paddle, far away
    assert events.find_swings(cam, ball, fps, 1920, paddles, 1080) == [30]


def test_filter_hits_drops_double_detections():
    evs = [{"frame": 0, "type": "hit", "player": "1"}, {"frame": 5, "type": "hit", "player": "7"},
           {"frame": 8, "type": "bounce", "player": None}, {"frame": 20, "type": "hit", "player": "1"},
           {"frame": 35, "type": "hit", "player": "2"}]
    assert [e["frame"] for e in events.filter_hits(evs, fps=30)] == [0, 8, 20, 35]
    side = {"1": "near", "7": "near", "2": "far"}       # frame 20: near side again 0.67 s later
    assert [e["frame"] for e in events.filter_hits(evs, 30, side=lambda e: side[e["player"]])] == [0, 8, 35]


def test_pov_shots_rallies_and_net_crossing():
    fps = 30
    evs = [{"frame": 0, "type": "hit", "player": "4", "side": "far"},
           {"frame": 20, "type": "bounce", "player": None},
           {"frame": 50, "type": "hit", "player": "4", "side": "far"},       # volley: no bounce after your swing
           {"frame": 60, "type": "hit", "player": "6", "side": "far"},       # same side 0.33 s later: impossible
           {"frame": 400, "type": "hit", "player": "5", "side": "near"},     # long gap -> new rally (partner serves)
           ]
    swings = [30, 410]                                # 410 follows your partner's hit, not the opponent's
    shots, rallies = events.pov_shots(evs, swings, fps)
    assert [(s["player"], s["type"]) for s in shots] == [("4", "serve"), ("me", "return"), ("4", "volley"), ("5", "serve")]
    assert [r["shots"] for r in rallies] == [3, 1]

def test_training_labels_keep_moving_ball_and_fill_gaps():
    from training.labels import ball_tracks, labels_from_tracks, link_tracks
    frames = []
    for i in range(20):
        c = [[300.0, 50.0, 6, 6, 0.4]]                    # static logo, never moves
        if i not in (8, 9):                               # ball moving 10 px/frame, missed twice
            c.append([100.0 + 10 * i, 200.0, 6, 6, 0.3])
        frames.append({"main": True, "c": c})
    frames.append({"main": False, "c": []})               # camera cut ends every track
    kept = ball_tracks(link_tracks(frames))
    assert len(kept) == 1 and kept[0][0][1] == 100.0
    labels = labels_from_tracks(kept)
    assert sorted(labels) == list(range(20))
    x, y, size, conf, filled = labels[8]
    assert abs(x - 180) < 1e-6 and filled and not labels[7][4]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Ball trajectory cleanup, hit/bounce detection, shot classification and rallies."""
import numpy as np

from .constants import COURT_L, KITCHEN, NET_Y


def clean_ball(track, max_jump=150, max_gap=8):
    """Per-frame [x, y] or None -> Nx2 array (NaN = unknown), outliers removed, short gaps filled."""
    a = np.array([p if p is not None else (np.nan, np.nan) for p in track], float).reshape(-1, 2)
    orig, known = a.copy(), np.flatnonzero(~np.isnan(a[:, 0]))
    # A detection far from both neighbouring detections is a false positive (head, shoe, sign).
    for k in range(1, len(known) - 1):
        p, i, n = known[k - 1:k + 2]
        if min(np.linalg.norm(orig[i] - orig[p]), np.linalg.norm(orig[i] - orig[n])) > max_jump:
            a[i] = np.nan

    idx, ok = np.arange(len(a)), ~np.isnan(a[:, 0])
    if ok.sum() < 2:
        return a
    out = np.column_stack([np.interp(idx, idx[ok], a[ok, d]) for d in range(2)])
    known = idx[ok]
    out[:known[0]] = np.nan
    out[known[-1] + 1:] = np.nan
    for s, e in zip(known[:-1], known[1:]):
        if e - s - 1 > max_gap:  # long gap = ball out of play, don't invent a path
            out[s + 1:e] = np.nan
    return out


def nearest_player(p, frame_players, reach=0.6):
    """Player whose bbox (grown by reach * bbox height) contains p, closest first."""
    best, best_d = None, np.inf
    for pid, pl in frame_players.items():
        x1, y1, x2, y2 = pl["bbox"]
        d = np.hypot(max(x1 - p[0], 0, p[0] - x2), max(y1 - p[1], 0, p[1] - y2))
        if d <= (y2 - y1) * reach and d < best_d:
            best, best_d = pid, d
    return best


def find_events(ball, players, window=5, min_prominence=8, min_gap=6, reach=0.6,
                near_side=lambda pl: pl["court"][1] > NET_Y):
    """Hits and bounces are local extrema of the ball's image y.

    A bounce (ball falls then rises) is always a local max of image y. A far-side hit
    is a local min (ball comes up the screen, then goes back down); a near-side hit is
    a local max next to a near-side player. In POV mode every visible player counts as near.
    """
    y, events = ball[:, 1], []
    for t in range(window, len(y) - window):
        w = y[t - window:t + window + 1]
        if np.isnan(w).any() or (events and t - events[-1]["frame"] < min_gap):
            continue
        left, right = w[:window], w[window + 1:]
        if y[t] == w.max() and min(y[t] - left.min(), y[t] - right.min()) >= min_prominence:
            pid = nearest_player(ball[t], players[t], reach)
            etype = "hit" if pid is not None and near_side(players[t][pid]) else "bounce"
        elif y[t] == w.min() and min(left.max() - y[t], right.max() - y[t]) >= min_prominence:
            pid = nearest_player(ball[t], players[t], reach)
            etype = "hit" if pid is not None else None  # else: apex of a lob, not an event
        else:
            continue
        if etype:
            events.append({"frame": t, "type": etype, "player": pid if etype == "hit" else None,
                           "px": [round(float(v), 1) for v in ball[t]]})
    return events


def filter_hits(events, fps, min_gap_s=0.35, side=None, same_side_s=1.0):
    """Drop hits that can't be separate shots.

    - re-detections of one contact: a hit < min_gap_s after the previous kept hit (time only,
      because player IDs churn and one person can show up under two IDs)
    - with side(event) -> "near"/"far": a hit by the same side < same_side_s after the previous
      kept hit, since the ball has to cross the net and come back first
    """
    out, last = [], None
    for e in events:
        if e["type"] == "hit":
            gap = e["frame"] - last["frame"] if last else np.inf
            if gap < min_gap_s * fps or (side and gap < same_side_s * fps and side(e) == side(last)):
                continue
            last = e
        out.append(e)
    return out


def shot_type(n_in_rally, hitter, landing, speed, airtime, volley, drive_speed=10.0):
    """Rule-based pickleball shot label. hitter/landing are court metres, speed m/s."""
    if n_in_rally == 1:
        return "serve"
    if n_in_rally == 2:
        return "return"
    if speed is not None and speed >= drive_speed:
        return "drive"
    if landing is not None and abs(landing[1] - NET_Y) <= KITCHEN:
        return "dink" if abs(hitter[1] - NET_Y) <= KITCHEN + 1.0 else "drop"
    if landing is not None and abs(landing[1] - NET_Y) > COURT_L / 2 - 2.0 and airtime > 1.0:
        return "lob"
    return "volley" if volley else "groundstroke"


def classify_shots(events, players, court, fps, rally_gap_s=2.5, drive_speed=10.0):
    """-> (shots, rallies). Every hit becomes a shot with type, speed, landing and faults."""
    hits = [i for i, e in enumerate(events) if e["type"] == "hit"]
    shots, rallies, prev = [], [], None
    for k, i in enumerate(hits):
        e, f = events[i], events[i]["frame"]
        new_rally = prev is None or (f - events[prev]["frame"]) / fps > rally_gap_s
        if new_rally:
            rallies.append({"start": f, "end": f, "shots": 0})
        rallies[-1]["shots"] += 1
        rallies[-1]["end"] = f
        volley = not new_rally and not any(b["type"] == "bounce" for b in events[prev + 1:i])

        nxt = hits[k + 1] if k + 1 < len(hits) else len(events)
        landing_ev = next((b for b in events[i + 1:nxt] if b["type"] == "bounce"), None)
        # destination: the bounce, else the next hitter in the same rally (a re-hit by the same
        # player, or a hit in the next point, says nothing about this shot's speed)
        nxt_ev = events[nxt] if nxt < len(events) else None
        same_rally = nxt_ev and nxt_ev["player"] != e["player"] and (nxt_ev["frame"] - f) / fps <= rally_gap_s
        target_ev = landing_ev or (nxt_ev if same_rally else None)
        hitter = np.array(players[f][e["player"]]["court"])
        landing = speed = None
        airtime = 0.0
        if target_ev:
            if landing_ev:
                landing = court.to_court([landing_ev["px"]])[0]
                tgt = landing
            else:  # volleyed: use the next hitter's position as the ball's destination
                tgt = np.array(players[target_ev["frame"]][target_ev["player"]]["court"])
            airtime = (target_ev["frame"] - f) / fps
            speed = float(np.linalg.norm(tgt - hitter) / airtime)

        n = rallies[-1]["shots"]
        faults = []
        if volley and court.in_kitchen(hitter):
            faults.append("kitchen_volley")
        if landing is not None and not court.inside(landing):
            faults.append("out")
        if landing is not None and court.side(landing) == court.side(hitter):
            faults.append("net")  # bounced on the hitter's own side
        shots.append({
            "frame": f, "time_s": round(f / fps, 2), "player": e["player"], "rally": len(rallies),
            "n_in_rally": n, "type": shot_type(n, hitter, landing, speed, airtime, volley, drive_speed),
            "volley": volley, "speed_kmh": None if speed is None else round(speed * 3.6, 1),
            "hitter": hitter.round(2).tolist(),
            "landing": None if landing is None else landing.round(2).tolist(), "faults": faults,
        })
        prev = i
    for r in rallies:
        r["duration_s"] = round((r["end"] - r["start"]) / fps, 2)
    return shots, rallies


def compensate(track, cam_shift):
    """POV: undo the cumulative camera pan so ball positions share one reference view.

    Only valid locally (pan, not rotation/fisheye), which is all hit/bounce detection needs.
    """
    off = np.cumsum(np.asarray(cam_shift, float).reshape(-1, 2), axis=0)
    return [None if p is None else [p[0] - o[0], p[1] - o[1]] for p, o in zip(track, off)], off


def find_swings(cam_shift, ball, fps, width, paddles=None, height=None, rel_thresh=0.02,
                window_s=0.3, rally_s=1.0, own_paddle=0.2):
    """POV: your own swing jolts the camera -> spikes in camera motion while the ball is in play.

    Head turns jolt the camera too, so when paddle boxes are available a jolt also needs your
    own paddle in view (a paddle box taller than own_paddle * frame height) within window_s.
    """
    # shortcut: thresholds come from one hand-checked 30 s clip (chest mount); re-check per mount.
    m = np.linalg.norm(np.asarray(cam_shift, float).reshape(-1, 2), axis=1)
    thr = max(4 * np.median(m), rel_thresh * width)
    w, r = max(1, int(window_s * fps)), int(rally_s * fps)
    seen = ~np.isnan(ball[:, 0])
    own = None
    if paddles and any(paddles):
        own = np.array([p is not None and p[3] - p[1] > own_paddle * height for p in paddles])
    swings = []
    for t in range(len(m)):
        if m[t] < thr or m[t] < m[max(0, t - w):t + w + 1].max():
            continue
        if not seen[max(0, t - r):t + r + 1].any() or (swings and t - swings[-1] < w):
            continue
        if own is not None and not own[max(0, t - w):t + w + 1].any():
            continue
        swings.append(t)
    return swings


def pov_shots(events, swings, fps, rally_gap_s=2.5, same_side_s=0.5):
    """POV: no court, so a shot is who hit when (you = "me"), grouped into rallies.

    Hit events carry a "side" ("near" = you/partner, "far" = across the net). The ball must
    cross the net between hits, so:
    - a hit by the same side as the previous one within same_side_s is dropped;
    - a camera jolt is your swing only if the latest hit before it came from the far side
      0.15-2 s earlier (head turns and walking jolt the camera too; your serves are missed).
    Not done: dropping far-side-only "rallies" (ball passed between points) also dropped real
    points whose near-side hits were missed - worse recall for little precision on test footage.
    """
    hits = sorted((e["frame"], e["player"], e.get("side", "far")) for e in events if e["type"] == "hit")

    def is_swing(f):
        prior = [h for h in hits if h[0] < f]
        return bool(prior) and prior[-1][2] == "far" and 0.15 * fps < f - prior[-1][0] <= 2 * fps

    kept = []
    for h in sorted(hits + [(f, "me", "near") for f in swings if is_swing(f)]):
        if kept and h[2] == kept[-1][2] and h[0] - kept[-1][0] < same_side_s * fps:
            continue
        kept.append(h)
    groups = []
    for h in kept:
        if not groups or (h[0] - groups[-1][-1][0]) / fps > rally_gap_s:
            groups.append([])
        groups[-1].append(h)

    bounces = [e["frame"] for e in events if e["type"] == "bounce"]
    shots, rallies = [], []
    for g in groups:
        rallies.append({"start": g[0][0], "end": g[-1][0], "shots": len(g),
                        "duration_s": round((g[-1][0] - g[0][0]) / fps, 2)})
        for n, (f, pid, side) in enumerate(g, 1):
            volley = n > 1 and not any(g[n - 2][0] < b < f for b in bounces)
            shots.append({"frame": f, "time_s": round(f / fps, 2), "player": pid, "side": side,
                          "rally": len(rallies), "n_in_rally": n,
                          "type": shot_type(n, None, None, None, 0.0, volley),
                          "volley": volley, "speed_kmh": None, "faults": []})
    return shots, rallies

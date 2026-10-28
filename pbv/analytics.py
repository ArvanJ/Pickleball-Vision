"""Per-player movement, shot and stroke-form metrics."""
from collections import Counter

import numpy as np

from .court import Court

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def player_stats(players, fps, sample_s=0.2, max_speed=8.0):
    """Distance, speed, kitchen time and side per player id. Tracks shorter than 1 s are noise."""
    tracks = {}
    for f, fp in enumerate(players):
        for pid, p in fp.items():
            tracks.setdefault(pid, []).append((f, *p["court"]))
    step, stats = max(1, round(fps * sample_s)), {}
    for pid, tr in tracks.items():
        if len(tr) < fps:
            continue
        a = np.array(tr)
        s = a[::step]  # subsample so box jitter doesn't inflate distance
        dt = np.diff(s[:, 0]) / fps
        d = np.linalg.norm(np.diff(s[:, 1:], axis=0), axis=1)
        v = d / np.maximum(dt, 1e-9)
        ok = v <= max_speed  # faster than a sprinting human = ID swap / bad box
        dist = float(d[ok].sum())
        sides = Counter(Court.side(p) for p in a[:, 1:])
        stats[pid] = {
            "seconds": round(len(tr) / fps, 1),
            "side": sides.most_common(1)[0][0],
            "distance_m": round(dist, 1),
            "avg_speed_ms": round(dist / max(float(dt[ok].sum()), 1e-9), 2),
            "max_speed_ms": round(float(v[ok].max()), 2) if ok.any() else 0.0,
            "kitchen_pct": round(100 * float(np.mean([Court.in_kitchen(p) for p in a[:, 1:]])), 1),
        }
    return stats


def shot_stats(shots):
    """Shot counts by type, volleys, faults, and shot speed per player."""
    per = {}
    for s in shots:
        p = per.setdefault(s["player"], {"shots": 0, "volleys": 0, "faults": 0, "_speeds": []})
        p["shots"] += 1
        p["volleys"] += s["volley"]
        p["faults"] += len(s["faults"])
        p[f"n_{s['type']}"] = p.get(f"n_{s['type']}", 0) + 1
        if s["speed_kmh"] is not None:
            p["_speeds"].append(s["speed_kmh"])
    for p in per.values():
        sp = p.pop("_speeds")
        p["avg_shot_kmh"] = round(float(np.mean(sp)), 1) if sp else None
        p["max_shot_kmh"] = max(sp) if sp else None
    return per


def _angle(a, b, c):
    """Angle at b in degrees."""
    u, w = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))))


def stroke_form(kp):
    """Contact-frame form from 17x2 COCO keypoints, or None if the arm/hips weren't seen."""
    kp = np.asarray(kp, float)
    hip = kp[[L_HIP, R_HIP]].mean(0)
    # The paddle arm is the one whose wrist is furthest from the body at contact.
    wrist = max((L_WRIST, R_WRIST), key=lambda w: np.linalg.norm(kp[w] - hip))
    shoulder, elbow = (L_SHOULDER, L_ELBOW) if wrist == L_WRIST else (R_SHOULDER, R_ELBOW)
    if not kp[[shoulder, elbow, wrist, L_HIP, R_HIP]].all(axis=1).all():  # (0,0) = keypoint missing
        return None
    shoulder_y = kp[[L_SHOULDER, R_SHOULDER]][:, 1].min()
    return {
        "arm": "left" if wrist == L_WRIST else "right",
        "elbow_angle": round(_angle(kp[shoulder], kp[elbow], kp[wrist]), 1),
        "contact_below_waist": bool(kp[wrist, 1] > hip[1]),  # image y grows downward
        "overhead": bool(kp[wrist, 1] < shoulder_y),
    }

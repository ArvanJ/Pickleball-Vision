# 🏓 Pickleball-Vision

**Computer-vision analytics for pickleball video, from a tripod camera or a camera you wear while playing.**

Pickleball-Vision turns raw match footage into player tracking, ball trajectories, hits and
bounces, shot types, rallies, rule faults, stroke form from body pose, court heatmaps and an
annotated replay. It also comes with an interactive dashboard.

It adapts the pipeline of [Badminton-VisionAI](https://github.com/amramer/Badminton-VisionAI)
(YOLO + ByteTrack tracking, homography court mapping, shot detection, dashboard) to pickleball's
court and rules. It adds the pose-based stroke analysis popularised by
[IsoCourt](https://www.isocourt.fit/). It also adds a **POV mode** for footage from a
chest/head-mounted action camera, which most recreational players actually have.

---

## Contents

- [Features](#features)
- [Two modes: fixed vs POV](#two-modes-fixed-vs-pov)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [Dashboard](#dashboard)
- [Project structure](#project-structure)
- [Measured accuracy](#measured-accuracy)
- [Tuning](#tuning)
- [Recording tips](#recording-tips)
- [Testing](#testing)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Credits and licences](#credits-and-licences)

---

## Features

### 🎥 Computer-vision pipeline
- **Player detection and tracking.** YOLOv8 finds people and ByteTrack keeps a persistent ID for each one across frames.
  In fixed mode, people standing off the court (spectators, other courts) are filtered out.
  In POV mode, your own arms, hands and legs are filtered out.
- **Ball tracking.** YOLO detects the ball every frame. Isolated false detections (heads, shoes, signs) are
  removed, and short gaps are filled by interpolation. Long gaps are left empty: the ball is out of play.
- **Paddle detection.** In POV mode, paddles (COCO "tennis racket") are detected in the same pass as the ball,
  to recognise your own paddle in frame.
- **Court mapping (fixed mode).** You click 4 or more court landmarks once: corners, kitchen-line ends and centres,
  or baseline centres. Corners that are off-screen or hidden behind the net can be skipped. A homography then maps every
  pixel on the ground to real court metres (20 × 44 ft court, 7 ft non-volley zone).
- **Camera-motion compensation (POV mode).** Frame-to-frame camera pan is measured with phase correlation and
  removed from the ball track, so hits and bounces are found relative to the world, not to your moving head.
- **Pose estimation.** YOLOv8-pose keypoints at each contact frame give stroke form.

### 🏓 Shot intelligence
- **Hit and bounce detection.** These come from turning points in the ball's path plus how close it is to a player.
  A bounce always shows up as the ball's lowest point on screen. A hit is a turning point next to a player.
- **Your own swings (POV).** A sudden camera jolt while the ball is in play counts as your swing, but only
  if your own paddle is visible and an opponent hit the ball to you just before.
- **Shot types (fixed):** serve, return, drive, dink, drop (e.g. third-shot drop), lob, volley, groundstroke.
- **Shot types (POV):** serve, return, volley, groundstroke.
- **Volleys.** A shot is a volley when there was no bounce since the previous hit.
- **Shot speed (fixed).** Horizontal court-plane speed from the hitter to where the ball lands (or to the next hitter).
- **Landing spots (fixed).** Bounce positions in court metres.
- **Rallies.** Hits are grouped into rallies (a gap of more than 2.5 s starts a new rally), with shot counts and durations.

### ⚖️ Rule checks
| Fault | How it's detected | Mode |
|---|---|---|
| Kitchen volley | Volley hit while the hitter's feet are in the non-volley zone | fixed |
| Out | Bounce lands outside the court lines | fixed |
| Net | Ball bounces on the hitter's own side | fixed |
| Serve above waist | At serve contact the paddle wrist is clearly above the waist (pose) | fixed |

### 🧍 Stroke form (pose)
At every shot by a tracked player:
- **Paddle arm** (left or right)
- **Elbow angle** at contact
- **Contact height** (0 = waist, 1 = shoulders), plus **below waist** and **overhead** flags

### 📊 Player metrics (fixed)
- Distance covered (m)
- Average and top movement speed (m/s)
- Percentage of time in the kitchen
- Court side
- Positional **heatmap** per player

### 🖥️ Outputs and dashboard
- Annotated video with player boxes and IDs, ball trail, a live mini-court (fixed), and shot/fault labels
- `stats.json` with every number above
- A Streamlit dashboard: replay, player table, shot mix chart, heatmaps, shot log with a fault filter, rally table
- **Caching.** Tracking (the slow part) is saved to `tracking.json` and reused when only analysis settings change.

---

## Two modes: fixed vs POV

| | `--mode fixed` (default) | `--mode pov` |
|---|---|---|
| Camera | Tripod or fence mount, whole court in view | Worn by a player (chest, head, paddle) |
| Setup | Click 4+ court landmarks once | None |
| Player tracking, ball trail, rallies | ✅ | ✅ (camera pan removed) |
| Other players' hits + stroke form | ✅ | ✅ |
| Your own swings | as a tracked player | ✅ camera jolt + your paddle in view |
| Shot types | serve, return, drive, dink, drop, lob, volley, groundstroke | serve, return, volley, groundstroke |
| Speed, landing spots, heatmaps, distance, kitchen % | ✅ | ❌ needs a fixed view of the court |
| Faults | kitchen volley, out, net, serve above waist | — (POV "serves" are too unreliable to fault) |

**Why POV can't measure court positions:** a homography maps the image to the court only while the
camera stays still. A worn camera moves dozens of pixels every frame and often sees only part of the
court through a fisheye lens. So any court-plane number would be wrong.

---

## How it works

```
                         ┌────────────────────────── fixed mode ──────────────────────────┐
video ─► frames ─┬─► court homography (4+ clicked landmarks) ─► feet → court metres
                 │
                 ├─► YOLOv8 + ByteTrack ─► players {id: bbox, court xy}
                 ├─► YOLOv8 ─► ball xy (+ paddles in POV)
                 └─► phase correlation ─► camera shift per frame            (POV only)
                                 │
                                 ▼
       ball cleanup: outliers out, short gaps interpolated, camera pan removed (POV)
                                 │
                                 ▼
       events: turning points in ball y ─► hit (near a player) / bounce
               + your swings: camera jolt + own paddle + opponent hit just before   (POV)
               + de-duplication: one contact per 0.35 s, one hit per side per 1 s (fixed) / 0.5 s (POV)
                                 │
                                 ▼
       shots: type, volley, speed + landing (fixed), rally grouping, rule faults
                                 │
                                 ▼
       YOLOv8-pose at each contact ─► paddle arm, elbow angle, contact height
                                 │
                                 ▼
       stats.json · heatmaps · annotated.mp4 ─► Streamlit dashboard
```

**Hits vs bounces (fixed mode).** In the image, a bounce always appears as a local *maximum* of the
ball's y-coordinate (it falls, then rises). A far-side hit is a local *minimum* (the ball comes up the
screen toward the far player, then goes back down). A near-side hit is a local maximum right next to a
near-side player.

**Shot classification (fixed mode)**, in priority order:
1. First hit of a rally → **serve**; second → **return**
2. Speed ≥ `--drive-speed` (10 m/s) → **drive**
3. Lands in the kitchen → **dink** if hit from near the kitchen line, otherwise **drop**
4. Lands deep (within 2 m of the baseline) after more than 1 s in the air → **lob**
5. Otherwise **volley** (no bounce since the last hit) or **groundstroke**

---

## Installation

Requires Python 3.10+. Runs on macOS (Apple-silicon GPU is used automatically), Linux and Windows
(CUDA if available, otherwise CPU).

```bash
git clone https://github.com/ArvanJ/Pickleball-Vision.git
cd Pickleball-Vision
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

YOLO weights (`yolov8m.pt`, `yolov8n-pose.pt`) download automatically on first run (~60 MB).

---

## Quick start

**Fixed camera:**
```bash
python app.py data/match.mp4
```
A window opens and asks for court landmarks one at a time, starting with the four corners. Click each
one, or press **s** to skip a landmark you can't see (for example a corner off the edge of the frame or
behind the net). Once at least 4 are placed (not all on one line), press **Enter**. **u** undoes,
**r** resets and **Esc** aborts. The calibration is saved and reused.

Landmarks: `far-left`, `far-right`, `near-right`, `near-left`, `near-baseline-center`,
`far-baseline-center`, `near-kitchen-left|center|right`, `far-kitchen-left|center|right`.

**POV camera:**
```bash
python app.py data/my_gopro.mp4 --mode pov
```

**Analyse a slice of a long video** (recommended: POV clips are often 10–20 minutes):
```bash
python app.py data/my_gopro.mp4 --mode pov --start 120 --max-frames 1800
```

**Open the dashboard:**
```bash
streamlit run dashboard.py
```

**Headless / scripted calibration** (all 4 corners visible):
```bash
python app.py data/match.mp4 --corners 412,318,1508,318,1840,1012,80,1012
```
Or with any 4+ named landmarks, e.g. a floor-level phone that sees only the near half:
```bash
python app.py IMG_1577.MOV --court-points "near-left=1537,1613;near-baseline-center=3330,1613;near-kitchen-left=1642,1398;near-kitchen-center=2354,1404"
```

---

## Command-line reference

### `app.py`
| Flag | Default | Description |
|---|---|---|
| `video` | — | Input video (any format OpenCV reads: MP4, MOV, AVI…) |
| `--mode {fixed,pov}` | `fixed` | Camera type |
| `--out DIR` | `outputs/<name>` or `outputs/<name>-pov` | Output folder |
| `--corners x1,y1,…,x4,y4` | — | Fixed mode: court corners in pixels (far-left, far-right, near-right, near-left); skips the click window |
| `--court-points "name=x,y;…"` | — | Fixed mode: 4+ named landmarks (see Quick start) when some corners aren't visible |
| `--start SECONDS` | `0` | Where to start in the video |
| `--max-frames N` | all | How many frames to analyse |
| `--player-model` | `yolov8m.pt` | YOLO model for people |
| `--ball-model` | `yolov8m.pt` | YOLO model for the ball |
| `--ball-class` | `32` | Class id of the ball in `--ball-model` (`-1` = any class). Use `0` for a single-class custom model |
| `--pose-model` | `yolov8n-pose.pt` | YOLO pose model |
| `--no-pose` | off | Skip stroke-form analysis |
| `--drive-speed` | `10.0` | Fixed mode: shot speed (m/s) above which a shot is a drive |
| `--force` | off | Re-run tracking even if a matching `tracking.json` exists |

### `calibrate.py`
```bash
python calibrate.py match.mp4 outputs/match/court.json --frame 300
```
Re-do the court calibration on any frame.

### `demo/build.py`
```bash
python demo/build.py demo/replay.html "outputs/IMG_1577:Fixed · floor phone" "outputs/clip-pov:POV · chest cam"
```
Builds a single self-contained HTML replay from one or more output folders. It shows synced camera and
court views, a shot timeline, a player table and a shot log. Only detections are embedded, never video
pixels, so the file is safe to share. Per-run captions come from `demo/meta.json`.

---

## Outputs

Each run writes to `outputs/<video name>[-pov]/`:

| File | Contents |
|---|---|
| `annotated.mp4` | Replay with boxes, IDs, ball trail, mini-court (fixed), shot + fault labels. H.264 so it plays in browsers |
| `stats.json` | Everything below |
| `tracking.json` | Raw per-frame detections (cache) plus the settings that produced them |
| `court.json` | Court calibration: clicked image points and their court coordinates (fixed) |
| `heatmap_P<id>.png` | Court-coverage heatmap per player (fixed) |

### `stats.json` structure
```jsonc
{
  "video": "...", "mode": "fixed", "start_s": 0, "fps": 30, "frames": 900,
  "players": {                         // fixed: movement metrics; POV: seconds visible
    "3": {"seconds": 28.1, "side": "near", "distance_m": 41.2,
          "avg_speed_ms": 1.46, "max_speed_ms": 4.9, "kitchen_pct": 37.5}
  },
  "shot_summary": {                    // per player ("me" = you, in POV)
    "3": {"shots": 9, "volleys": 4, "faults": 1, "n_dink": 5, "n_drive": 2,
          "avg_shot_kmh": 31.4, "max_shot_kmh": 52.0}
  },
  "rallies": [{"start": 120, "end": 410, "shots": 7, "duration_s": 9.7}],
  "shots": [{
    "frame": 120, "time_s": 4.0, "player": "3", "rally": 1, "n_in_rally": 1,
    "type": "serve", "volley": false, "speed_kmh": 38.2,
    "hitter": [3.1, 13.2], "landing": [1.9, 2.4],          // court metres (fixed)
    "faults": [],
    "form": {"arm": "right", "elbow_angle": 142.0,
             "contact_below_waist": true, "overhead": false}
  }],
  "events": [{"frame": 150, "type": "bounce", "player": null, "px": [812.0, 604.5]}]
}
```
Court coordinates: `x` runs across the court (0–6.10 m) and `y` along it (0 = far baseline,
13.41 m = near baseline, 6.71 m = net).

---

## Dashboard

`streamlit run dashboard.py` lists every analysed video in `outputs/` and shows:

- **Headline numbers:** duration, rallies, shots, faults
- **Annotated replay**
- **Player table:** movement metrics, shot counts by type, volleys, faults, shot speeds
- **Shot mix:** bar chart of shot types per player
- **Court coverage:** heatmaps (fixed mode)
- **Shot log:** every shot with timing, type, speed, landing spot, stroke form and faults, filterable to faults only
- **Rallies:** start, end, shot count, duration

---

## Project structure

```
Pickleball-Vision/
├── app.py              # Pipeline entry point (both modes)
├── calibrate.py        # Click-to-calibrate court corners
├── dashboard.py        # Streamlit dashboard
├── pbv/
│   ├── constants.py    # Official court geometry (metres)
│   ├── court.py        # Image ↔ court homography; kitchen / side / in-bounds checks
│   ├── tracking.py     # YOLO player tracker (ByteTrack), ball + paddle detector, pose estimator
│   ├── events.py       # Ball cleanup, hits/bounces, de-duplication, shot types, rallies, faults,
│   │                   #   POV camera compensation, swing detection
│   ├── analytics.py    # Player movement stats, shot summaries, stroke form from keypoints
│   └── draw.py         # Frame annotation, mini-court, heatmaps
├── demo/
│   ├── build.py        # Builds the shareable HTML replay from output folders
│   ├── template.html   # Replay page (camera view, court view, timeline, tables)
│   └── meta.json       # Per-run captions for the replay
├── tests/
│   └── test_core.py    # Self-checks for all non-ML logic
└── requirements.txt
```

---

## Measured accuracy

> These are honest numbers from limited footage. Treat them as a baseline, not a benchmark.

**POV mode** was hand-checked on a 30-second, 4K/30 fps chest-mounted clip of recreational doubles
(3 live points, about 12 hits). Every detection was compared frame by frame with the video:

| Version | Detections | Precision | Recall |
|---|---|---|---|
| First version | 30 | 37% | 92% |
| + de-duplication, net-crossing rule, own-paddle check for your swings | 16 | **56%** | **75%** |

- Ball detected in **56%** of frames with the stock COCO model.
- Players detected reliably when in view. **IDs are not stable:** the same person gets a new ID after leaving and re-entering the frame.
- **Main remaining errors:**
  - players passing or bouncing the ball between points look like hits
  - your swings when your paddle is out of view are missed
  - players on neighbouring courts are occasionally picked up

**Fixed mode** was run on one minute of indoor doubles filmed by an iPhone lying on the floor
behind the near baseline (4K/60 fps). What it showed:
- Calibration works from 4 non-corner landmarks. The projected kitchen line, centre line and far
  sideline land on the painted lines.
- The ball was detected in only **23%** of frames. From floor level the far half of the court is about
  70 pixels tall, so the pickleball is a few pixels and far-side hits are missed.
- Near-side strokes by the player closest to the camera are found. False hits come from bouncing the
  ball before a serve and from carrying it between points. The net-crossing rule (one hit per side
  per second) and only computing speed toward a bounce or a different player came from this check.
- **A raised camera matters more than any setting.** See [Recording tips](#recording-tips).

**Speed:** about 3–6 frames per second on an Apple M-series GPU at 4K (two YOLO models per frame),
so a 10-minute POV clip takes roughly an hour. Use `--start` / `--max-frames`.

---

## Tuning

All thresholds are plain function arguments with docstrings:

| What | Where | Default |
|---|---|---|
| Rally gap | `classify_shots` / `pov_shots` `rally_gap_s` | 2.5 s |
| Duplicate-contact window | `filter_hits` `min_gap_s` | 0.35 s |
| One hit per side window | `filter_hits` `same_side_s` (fixed) / `pov_shots` `same_side_s` (POV) | 1.0 s / 0.5 s |
| Serve-above-waist fault | `add_stroke_form` in `app.py` | wrist > half-way from waist to shoulders |
| Drive speed | `--drive-speed` | 10 m/s |
| Dink / drop / lob zones | `shot_type` | kitchen ± 1 m, last 2 m before the baseline |
| Hit reach around a player | `find_events` `reach` | 0.6 × player height |
| Turning-point prominence | `find_events` `min_prominence` | 8 px |
| Swing jolt threshold (POV) | `find_swings` `rel_thresh` | 2% of frame width per frame (or 4× median motion) |
| Own-paddle size (POV) | `find_swings` `own_paddle` | 20% of frame height |
| Ball gap filling | `clean_ball` `max_gap` | 8 frames |

Because tracking is cached, changing any of these and re-running `app.py` only re-does the analysis
(seconds to a few minutes), not the detection.

---

## Recording tips

**Fixed mode (best results):**
- Tripod or fence mount behind a baseline, as high as possible (2–3 m+). A phone on the floor
  squeezes the far half of the court into a sliver, and the ball becomes nearly invisible
- Ideally all four corners are in frame. If not, make sure at least 4 landmarks are visible (e.g. both
  near corners plus the two ends of the near kitchen line)
- 60 fps if your camera supports it. The ball moves fast and blurs at 30 fps
- Keep the camera still for the whole recording. If it moves, re-run `calibrate.py`

**POV mode:**
- Chest mount is steadier than head mount, which helps both camera compensation and swing detection
- Point it slightly up so the far court and net stay in frame during ready position
- Wide/fisheye is fine; the pipeline doesn't need straight lines in POV mode

---

## Testing

```bash
python tests/test_core.py        # or: python -m pytest tests
```

Covers:
- homography round-trips, kitchen checks, and calibration from non-corner landmarks
- ball outlier removal and gap filling
- hit/bounce detection on a synthetic trajectory
- shot classification and faults (serve, return, drop, kitchen volley)
- movement metrics and stroke form from keypoints
- heatmap rendering
- camera-shift sign and compensation
- own-body filtering, swing detection with and without paddle data
- duplicate-hit removal (time and same-side) and POV rally/volley logic

---

## Limitations and roadmap

**Current limitations:**
- **Ball detection** uses COCO's generic "sports ball" class. Small, motion-blurred pickleballs are often missed.
- **Fixed mode calibration is manual**, and the whole court must be visible.
- **Fixed-mode speeds** are average horizontal speeds to the bounce point, not the speed off the paddle.
- **Shot types and faults are rule-based** heuristics.
- **POV mode** has no court geometry. Player IDs churn, and ball handling between points can look like play.

**Roadmap:**
- [ ] Pickleball-specific ball detector trained on labelled frames (Roboflow Universe has several datasets), passed via `--ball-model`
- [ ] TrackNet-style heatmap ball tracker for blur and occlusion (as Badminton-VisionAI uses)
- [ ] Automatic court-keypoint detection to replace manual calibration
- [ ] Player re-identification (appearance embeddings) for stable IDs, especially in POV
- [ ] Learned shot classifier from labelled rallies
- [ ] PDF coach report and Docker images
- [ ] A labelled evaluation set with a scoring script in the repo

---

## Credits and licences

- Pipeline design inspired by [Badminton-VisionAI](https://github.com/amramer/Badminton-VisionAI) by amramer
- Stroke-analysis idea inspired by [IsoCourt](https://www.isocourt.fit/)
- Detection, tracking and pose by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) with ByteTrack
- Dashboard built with [Streamlit](https://streamlit.io)

Note that Ultralytics YOLO is AGPL-3.0 licensed. If you distribute or host this project as a service,
check that licence's terms.

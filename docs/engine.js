// Pickleball-Vision analysis engine for the browser: a direct port of pbv/court.py,
// pbv/events.py and pbv/analytics.py (fixed-camera mode). Keep it in step with the Python.

export const FT = 0.3048;
export const COURT_W = 20 * FT, COURT_L = 44 * FT, NET_Y = COURT_L / 2, KITCHEN = 7 * FT;

export const LANDMARKS = {
  "far-left": [0, 0], "far-right": [COURT_W, 0], "near-right": [COURT_W, COURT_L], "near-left": [0, COURT_L],
  "near-baseline-center": [COURT_W / 2, COURT_L], "far-baseline-center": [COURT_W / 2, 0],
  "near-kitchen-left": [0, NET_Y + KITCHEN], "near-kitchen-center": [COURT_W / 2, NET_Y + KITCHEN],
  "near-kitchen-right": [COURT_W, NET_Y + KITCHEN],
  "far-kitchen-left": [0, NET_Y - KITCHEN], "far-kitchen-center": [COURT_W / 2, NET_Y - KITCHEN],
  "far-kitchen-right": [COURT_W, NET_Y - KITCHEN],
};

export const COURT_LINES = [
  [[0, 0], [COURT_W, 0]], [[COURT_W, 0], [COURT_W, COURT_L]], [[COURT_W, COURT_L], [0, COURT_L]], [[0, COURT_L], [0, 0]],
  [[0, NET_Y - KITCHEN], [COURT_W, NET_Y - KITCHEN]], [[0, NET_Y + KITCHEN], [COURT_W, NET_Y + KITCHEN]],
  [[COURT_W / 2, 0], [COURT_W / 2, NET_Y - KITCHEN]], [[COURT_W / 2, NET_Y + KITCHEN], [COURT_W / 2, COURT_L]],
  [[-0.3, NET_Y], [COURT_W + 0.3, NET_Y]],
];

// ---------- court homography ----------

function solve(A, b) {  // Gaussian elimination with partial pivoting; A is n x n
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function homography(src, dst) {  // least-squares DLT with h33 = 1; exact for 4 points
  const AtA = Array.from({ length: 8 }, () => new Array(8).fill(0)), Atb = new Array(8).fill(0);
  src.forEach(([x, y], i) => {
    const [u, v] = dst[i];
    for (const [row, rhs] of [[[x, y, 1, 0, 0, 0, -u * x, -u * y], u], [[0, 0, 0, x, y, 1, -v * x, -v * y], v]]) {
      for (let a = 0; a < 8; a++) { Atb[a] += row[a] * rhs; for (let c = 0; c < 8; c++) AtA[a][c] += row[a] * row[c]; }
    }
  });
  const h = solve(AtA, Atb);
  return h && [...h, 1];
}

function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map(v => v / det);
}

const apply = (H, [x, y]) => {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
};

export class Court {
  /** pointsPx[i] is where court point pointsM[i] (metres) appears in the image; 4+ points. */
  constructor(pointsPx, pointsM) {
    if (pointsPx.length < 4 || pointsPx.length !== pointsM.length) throw new Error("need 4+ matching image/court points");
    this.pointsPx = pointsPx; this.pointsM = pointsM;
    this.H = homography(pointsPx, pointsM);
    this.Hinv = this.H && invert3(this.H);
    if (!this.H || !this.Hinv) throw new Error("court points are degenerate (3 or more on one line?)");
  }
  static fromLandmarks(named) {
    const names = Object.keys(named), unknown = names.filter(n => !(n in LANDMARKS));
    if (unknown.length) throw new Error(`unknown landmarks: ${unknown.join(", ")}`);
    return new Court(names.map(n => named[n]), names.map(n => LANDMARKS[n]));
  }
  toCourt(p) { return apply(this.H, p); }
  toImage(p) { return apply(this.Hinv, p); }
  static inside(p, margin = 0) { return p[0] >= -margin && p[0] <= COURT_W + margin && p[1] >= -margin && p[1] <= COURT_L + margin; }
  static inKitchen(p) { return p[0] >= 0 && p[0] <= COURT_W && Math.abs(p[1] - NET_Y) <= KITCHEN; }
  static side(p) { return p[1] < NET_Y ? "far" : "near"; }
}

// ---------- players: on-court filter + simple IoU tracker ----------

const area = b => (b[2] - b[0]) * (b[3] - b[1]);
function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  return w * h / (area(a) + area(b) - w * h);
}

/** People whose feet stand on (or within margin m of) the court, biggest first. */
export function onCourt(people, court, maxPlayers = 4, margin = 2) {
  return people
    .map(p => ({ ...p, court: court.toCourt([(p.bbox[0] + p.bbox[2]) / 2, p.bbox[3]]) }))
    .filter(p => Court.inside(p.court, margin))
    .sort((a, b) => area(b.bbox) - area(a.bbox))
    .slice(0, maxPlayers);
}

// shortcut: greedy IoU matching instead of ByteTrack; IDs can swap when players cross.
export class Tracker {
  constructor(maxAge = 30, minIou = 0.2) { this.tracks = []; this.next = 1; this.maxAge = maxAge; this.minIou = minIou; }
  update(dets) {
    const pairs = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => { const v = iou(t.bbox, d.bbox); if (v >= this.minIou) pairs.push([v, ti, di]); }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), usedD = new Set(), out = {};
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      Object.assign(this.tracks[ti], { bbox: dets[di].bbox, age: 0 });
      out[this.tracks[ti].id] = dets[di];
    }
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.age++; });
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      const t = { id: String(this.next++), bbox: d.bbox, age: 0 };
      this.tracks.push(t); out[t.id] = d;
    });
    this.tracks = this.tracks.filter(t => t.age <= this.maxAge);
    const r = {};
    for (const [id, d] of Object.entries(out)) r[id] = { bbox: d.bbox.map(v => Math.round(v * 10) / 10), court: d.court.map(v => Math.round(v * 1000) / 1000) };
    return r;
  }
}

// ---------- ball cleanup + events (port of pbv/events.py) ----------

/** Per-frame [x, y] or null -> array of [x, y] or null: outliers removed, short gaps filled. */
export function cleanBall(track, maxJump = 150, maxGap = 8) {
  const a = track.map(p => (p ? [...p] : null));
  const known = [];
  track.forEach((p, i) => { if (p) known.push(i); });
  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
  // A detection far from both neighbouring detections is a false positive (head, shoe, sign).
  for (let k = 1; k < known.length - 1; k++) {
    const [p, i, n] = [known[k - 1], known[k], known[k + 1]];
    if (Math.min(dist(track[i], track[p]), dist(track[i], track[n])) > maxJump) a[i] = null;
  }
  const ok = [];
  a.forEach((p, i) => { if (p) ok.push(i); });
  if (ok.length < 2) return a;
  const out = new Array(a.length).fill(null);
  for (let k = 0; k < ok.length; k++) {
    const s = ok[k];
    out[s] = a[s];
    if (k + 1 < ok.length) {
      const e = ok[k + 1];
      if (e - s - 1 > maxGap) continue;  // long gap = ball out of play, don't invent a path
      for (let i = s + 1; i < e; i++) {
        const t = (i - s) / (e - s);
        out[i] = [a[s][0] + t * (a[e][0] - a[s][0]), a[s][1] + t * (a[e][1] - a[s][1])];
      }
    }
  }
  return out;
}

/** Player whose bbox (grown by reach * bbox height) contains p, closest first. */
export function nearestPlayer(p, framePlayers, reach = 0.6) {
  let best = null, bestD = Infinity;
  for (const [pid, pl] of Object.entries(framePlayers)) {
    const [x1, y1, x2, y2] = pl.bbox;
    const d = Math.hypot(Math.max(x1 - p[0], 0, p[0] - x2), Math.max(y1 - p[1], 0, p[1] - y2));
    if (d <= (y2 - y1) * reach && d < bestD) { best = pid; bestD = d; }
  }
  return best;
}

/** Hits and bounces = local extrema of the ball's image y (see find_events in pbv/events.py). */
export function findEvents(ball, players, { window = 5, minProminence = 8, minGap = 6, reach = 0.6, from = 0 } = {}) {
  const events = [];
  for (let t = Math.max(window, from); t < ball.length - window; t++) {
    const w = ball.slice(t - window, t + window + 1);
    if (w.some(p => !p) || (events.length && t - events[events.length - 1].frame < minGap)) continue;
    const ys = w.map(p => p[1]), y = ys[window];
    const left = ys.slice(0, window), right = ys.slice(window + 1);
    const max = Math.max(...ys), min = Math.min(...ys);
    let etype = null, pid = null;
    if (y === max && Math.min(y - Math.min(...left), y - Math.min(...right)) >= minProminence) {
      pid = nearestPlayer(ball[t], players[t], reach);
      etype = pid !== null && players[t][pid].court[1] > NET_Y ? "hit" : "bounce";
    } else if (y === min && Math.min(Math.max(...left) - y, Math.max(...right) - y) >= minProminence) {
      pid = nearestPlayer(ball[t], players[t], reach);
      etype = pid !== null ? "hit" : null;  // else: apex of a lob, not an event
    } else continue;
    if (etype) events.push({ frame: t, type: etype, player: etype === "hit" ? pid : null, px: ball[t].map(v => Math.round(v * 10) / 10) });
  }
  return events;
}

/** Drop re-detections of one contact, and same-side hits too soon for the ball to cross and return. */
export function filterHits(events, fps, { minGapS = 0.35, side = null, sameSideS = 1.0 } = {}) {
  const out = [];
  let last = null;
  for (const e of events) {
    if (e.type === "hit") {
      const gap = last ? e.frame - last.frame : Infinity;
      if (gap < minGapS * fps || (side && gap < sameSideS * fps && side(e) === side(last))) continue;
      last = e;
    }
    out.push(e);
  }
  return out;
}

export function shotType(n, hitter, landing, speed, airtime, volley, driveSpeed = 10) {
  if (n === 1) return "serve";
  if (n === 2) return "return";
  if (speed != null && speed >= driveSpeed) return "drive";
  if (landing && Math.abs(landing[1] - NET_Y) <= KITCHEN) return Math.abs(hitter[1] - NET_Y) <= KITCHEN + 1 ? "dink" : "drop";
  if (landing && Math.abs(landing[1] - NET_Y) > COURT_L / 2 - 2 && airtime > 1) return "lob";
  return volley ? "volley" : "groundstroke";
}

const r2 = v => Math.round(v * 100) / 100;

export function classifyShots(events, players, court, fps, { rallyGapS = 2.5, driveSpeed = 10 } = {}) {
  const hits = events.map((e, i) => (e.type === "hit" ? i : -1)).filter(i => i >= 0);
  const shots = [], rallies = [];
  let prev = null;
  hits.forEach((i, k) => {
    const e = events[i], f = e.frame;
    const newRally = prev === null || (f - events[prev].frame) / fps > rallyGapS;
    if (newRally) rallies.push({ start: f, end: f, shots: 0 });
    const rally = rallies[rallies.length - 1];
    rally.shots++; rally.end = f;
    const volley = !newRally && !events.slice(prev + 1, i).some(b => b.type === "bounce");
    const nxt = k + 1 < hits.length ? hits[k + 1] : events.length;
    const landingEv = events.slice(i + 1, nxt).find(b => b.type === "bounce") || null;
    // destination: the bounce, else the next hitter in the same rally
    const nxtEv = nxt < events.length ? events[nxt] : null;
    const sameRally = nxtEv && nxtEv.player !== e.player && (nxtEv.frame - f) / fps <= rallyGapS;
    const targetEv = landingEv || (sameRally ? nxtEv : null);
    const hitter = players[f][e.player].court;
    let landing = null, speed = null, airtime = 0;
    if (targetEv) {
      const tgt = landingEv ? (landing = court.toCourt(landingEv.px)) : players[targetEv.frame][targetEv.player].court;
      airtime = (targetEv.frame - f) / fps;
      speed = Math.hypot(tgt[0] - hitter[0], tgt[1] - hitter[1]) / airtime;
    }
    const faults = [];
    if (volley && Court.inKitchen(hitter)) faults.push("kitchen_volley");
    if (landing && !Court.inside(landing)) faults.push("out");
    if (landing && Court.side(landing) === Court.side(hitter)) faults.push("net");
    shots.push({
      frame: f, time_s: r2(f / fps), player: e.player, rally: rallies.length, n_in_rally: rally.shots,
      type: shotType(rally.shots, hitter, landing, speed, airtime, volley, driveSpeed), volley,
      speed_kmh: speed == null ? null : Math.round(speed * 36) / 10, hitter: hitter.map(r2),
      landing: landing && landing.map(r2), faults,
    });
    prev = i;
  });
  rallies.forEach(r => { r.duration_s = r2((r.end - r.start) / fps); });
  return { shots, rallies };
}

// ---------- analytics (port of pbv/analytics.py) ----------

export function playerStats(players, fps, { sampleS = 0.2, maxSpeed = 8 } = {}) {
  const tracks = {};
  players.forEach((fp, f) => { for (const [pid, p] of Object.entries(fp)) (tracks[pid] ||= []).push([f, ...p.court]); });
  const step = Math.max(1, Math.round(fps * sampleS)), stats = {};
  for (const [pid, tr] of Object.entries(tracks)) {
    if (tr.length < fps) continue;  // tracks shorter than 1 s are noise
    const s = tr.filter((_, i) => i % step === 0);
    let dist = 0, time = 0, vmax = 0;
    for (let i = 1; i < s.length; i++) {
      const dt = (s[i][0] - s[i - 1][0]) / fps, d = Math.hypot(s[i][1] - s[i - 1][1], s[i][2] - s[i - 1][2]);
      const v = d / Math.max(dt, 1e-9);
      if (v <= maxSpeed) { dist += d; time += dt; vmax = Math.max(vmax, v); }  // faster = ID swap / bad box
    }
    const near = tr.filter(p => p[2] >= NET_Y).length;
    stats[pid] = {
      seconds: Math.round(tr.length / fps * 10) / 10, side: near * 2 >= tr.length ? "near" : "far",
      distance_m: Math.round(dist * 10) / 10, avg_speed_ms: r2(dist / Math.max(time, 1e-9)), max_speed_ms: r2(vmax),
      kitchen_pct: Math.round(1000 * tr.filter(p => Court.inKitchen([p[1], p[2]])).length / tr.length) / 10,
    };
  }
  return stats;
}

export function shotStats(shots) {
  const per = {};
  for (const s of shots) {
    const p = (per[s.player] ||= { shots: 0, volleys: 0, faults: 0, _speeds: [] });
    p.shots++; p.volleys += s.volley ? 1 : 0; p.faults += s.faults.length;
    p["n_" + s.type] = (p["n_" + s.type] || 0) + 1;
    if (s.speed_kmh != null) p._speeds.push(s.speed_kmh);
  }
  for (const p of Object.values(per)) {
    const sp = p._speeds; delete p._speeds;
    p.avg_shot_kmh = sp.length ? Math.round(sp.reduce((a, b) => a + b, 0) / sp.length * 10) / 10 : null;
    p.max_shot_kmh = sp.length ? Math.max(...sp) : null;
  }
  return per;
}

/** Plain-language notes from the numbers (no AI service needed). */
export function coachNotes(result) {
  const { players, shots, rallies, ballPct, summary } = result, notes = [];
  if (ballPct < 35) notes.push(`The ball was tracked in only ${ballPct}% of frames, so some shots are missing. Raise the camera (2–3 m on a tripod behind a baseline) and record at 60 fps.`);
  const regulars = Object.entries(players).filter(([, p]) => p.seconds >= 5);
  const hangBack = regulars.filter(([pid, p]) => p.kitchen_pct < 15 && (summary[pid]?.shots || 0) > 0).map(([pid]) => "P" + pid);
  if (hangBack.length)
    notes.push(`${hangBack.join(", ")} hit from deep and rarely reached the kitchen line. Most points are won at the kitchen, so look for chances to move up after the third shot.`);
  const mover = regulars.sort((a, b) => b[1].distance_m - a[1].distance_m)[0];
  if (mover) notes.push(`P${mover[0]} covered the most ground: ${mover[1].distance_m} m, top speed ${mover[1].max_speed_ms} m/s.`);
  const kv = shots.filter(s => s.faults.includes("kitchen_volley"));
  if (kv.length) notes.push(`${kv.length} possible kitchen volley${kv.length > 1 ? "s" : ""} (feet in the kitchen on a volley). Check them in the shot log.`);
  const types = shots.reduce((a, s) => ((a[s.type] = (a[s.type] || 0) + 1), a), {});
  if ((types.drive || 0) > 2 * (types.dink || 0) + 2) notes.push(`Drives (${types.drive}) far outnumber dinks (${types.dink || 0}). Mixing in soft dinks and drops makes it harder for opponents to attack.`);
  if (types.drop) notes.push(`${types.drop} drop shot${types.drop > 1 ? "s" : ""} landed in the kitchen.`);
  const long = rallies.reduce((m, r) => Math.max(m, r.shots), 0);
  if (rallies.length) notes.push(`${rallies.length} rallies detected; the longest had ${long} shots.`);
  return notes;
}

/** Everything from per-frame players + raw ball detections. */
export function analyse(players, ballRaw, court, fps, width, eventOpts = {}) {
  const ball = cleanBall(ballRaw, 0.08 * width);
  const events = filterHits(findEvents(ball, players, eventOpts), fps, { side: e => Court.side(players[e.frame][e.player].court) });
  const { shots, rallies } = classifyShots(events, players, court, fps);
  const seen = ballRaw.filter(Boolean).length;
  const result = {
    fps, frames: players.length, ball, events, shots, rallies,
    players: playerStats(players, fps), summary: shotStats(shots),
    ballPct: Math.round(100 * seen / Math.max(ballRaw.length, 1)),
  };
  result.notes = coachNotes(result);
  return result;
}

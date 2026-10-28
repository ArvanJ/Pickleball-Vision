// Checks docs/engine.js against the Python pipeline on a real output folder.
// Usage: node tests/engine_parity.mjs outputs/IMG_1577
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { Court, analyse, LANDMARKS } from "../docs/engine.js";

const dir = process.argv[2] || "outputs/IMG_1577";
const py = JSON.parse(readFileSync(`${dir}/stats.json`));
const tr = JSON.parse(readFileSync(`${dir}/tracking.json`));
const cal = JSON.parse(readFileSync(`${dir}/court.json`));
const court = cal.corners ? Court.fromLandmarks({ "far-left": cal.corners[0], "far-right": cal.corners[1], "near-right": cal.corners[2], "near-left": cal.corners[3] })
  : new Court(cal.points_px, cal.points_m);

// Homography round-trip on every calibration point.
(cal.points_px || []).forEach((p, i) => court.toCourt(p).forEach((v, k) => assert.ok(Math.abs(v - cal.points_m[i][k]) < 0.01)));
assert.equal(Object.keys(LANDMARKS).length, 12);

const js = analyse(tr.players, tr.ball, court, py.fps, tr.width);
const ev = e => `${e.frame}:${e.type}:${e.player}`;
assert.deepEqual(js.events.map(ev), py.events.map(ev), "events differ");
// serve_above_waist needs the pose model, which the web engine doesn't run.
const shot = s => `${s.frame}:${s.player}:${s.type}:${s.volley}:${s.speed_kmh}:${s.faults.filter(f => f !== "serve_above_waist")}`;
assert.deepEqual(js.shots.map(shot), py.shots.map(shot), "shots differ");
for (const [pid, p] of Object.entries(py.players)) {
  const q = js.players[pid];
  assert.ok(q, `player ${pid} missing`);
  assert.ok(Math.abs(q.distance_m - p.distance_m) <= 0.2 && q.side === p.side && Math.abs(q.kitchen_pct - p.kitchen_pct) <= 0.2,
    `player ${pid}: js ${JSON.stringify(q)} vs py ${JSON.stringify(p)}`);
}
console.log(`parity ok: ${js.events.length} events, ${js.shots.length} shots, ${Object.keys(js.players).length} players, ball ${js.ballPct}%`);
console.log("notes:", js.notes);

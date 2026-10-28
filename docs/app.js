import { Court, LANDMARKS, COURT_LINES, COURT_W, COURT_L, NET_Y, KITCHEN, Tracker, onCourt, analyse } from "./engine.js?v=4";
import { loadDetector, detect, detectorBackend } from "./detector.js?v=4";

const $ = id => document.getElementById(id);
const video = $("video"), overlay = $("overlay");
const NAMES = Object.keys(LANDMARKS);
const TYPE_COLORS = { serve: "#7b5cd6", return: "#3a8fd6", drive: "#e0623a", dink: "#3f9a6a", drop: "#2fa8a0", lob: "#c58a1c", volley: "#d64f8a", groundstroke: "#6f7f8f" };
const PALETTE = ["#e0623a", "#3a8fd6", "#3f9a6a", "#c58a1c", "#7b5cd6", "#d64f8a"];
const color = pid => PALETTE[(parseInt(pid, 10) || 0) % PALETTE.length];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

const S = {
  mode: "live", court: null, cal: {}, calIdx: 0, calibrating: false, modelReady: {}, size: 640,
  running: false, cancel: false, stream: null, wakeLock: null,
  tracker: null, players: [], ball: [], t0: 0, fps: 30, start: 0, rate: 30, result: null, replay: false,
};

// ---------- small helpers ----------
function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }
function setStatus(msg) { $("status").textContent = msg; }
const vw = () => video.videoWidth, vh = () => video.videoHeight;

function ensureModel() {
  const size = S.size;
  S.modelReady[size] ||= loadDetector(size, setStatus)
    .then(ep => { setStatus(`Model ready · ${size} px · ${ep === "webgpu" ? "GPU" : "CPU"}`); return ep; })
    .catch(e => { delete S.modelReady[size]; showError(e.message); throw e; });
  return S.modelReady[size];
}
const runDetect = () => detect(video, vw(), vh(), S.size);
$("detail").addEventListener("change", e => {
  S.size = +e.target.value; msPerFrame = S.size === 1280 ? 500 : 130; updateEta();
  if (vw()) ensureModel().catch(() => {});
});

function fitCanvas() {
  const r = overlay.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  overlay.width = Math.round(r.width * d); overlay.height = Math.round(r.height * d);
}
function mapping() {  // video pixels <-> overlay pixels (object-fit: contain)
  const s = Math.min(overlay.width / vw(), overlay.height / vh());
  return { s, ox: (overlay.width - vw() * s) / 2, oy: (overlay.height - vh() * s) / 2 };
}

// ---------- overlay ----------
function drawOverlay(idx) {
  fitCanvas();
  const g = overlay.getContext("2d"), d = window.devicePixelRatio || 1;
  g.clearRect(0, 0, overlay.width, overlay.height);
  if (!vw()) return;
  const { s, ox, oy } = mapping(), X = x => ox + x * s, Y = y => oy + y * s;
  if (S.court) {
    g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 2 * d; g.beginPath();
    for (const [a, b] of COURT_LINES) {
      const p = S.court.toImage(a), q = S.court.toImage(b);
      g.moveTo(X(p[0]), Y(p[1])); g.lineTo(X(q[0]), Y(q[1]));
    }
    g.stroke();
  }
  if (S.calibrating) {
    g.font = `600 ${13 * d}px IBM Plex Sans, sans-serif`;
    for (const [name, [x, y]] of Object.entries(S.cal)) {
      g.fillStyle = "#e0623a"; g.beginPath(); g.arc(X(x), Y(y), 6 * d, 0, 7); g.fill();
      g.fillStyle = "#fff"; g.fillText(name, X(x) + 9 * d, Y(y) - 8 * d);
    }
    return;
  }
  const fp = S.players[idx];
  if (!fp) return;
  g.font = `600 ${13 * d}px IBM Plex Sans, sans-serif`;
  for (const [pid, p] of Object.entries(fp)) {
    const [x1, y1, x2, y2] = p.bbox;
    g.strokeStyle = g.fillStyle = color(pid); g.lineWidth = 2 * d;
    g.strokeRect(X(x1), Y(y1), (x2 - x1) * s, (y2 - y1) * s); g.fillText("P" + pid, X(x1), Y(y1) - 5 * d);
  }
  const trail = S.ball.slice(Math.max(0, idx - 10), idx + 1).filter(Boolean);
  g.strokeStyle = "#d4ec3a"; g.lineWidth = 3 * d; g.beginPath();
  trail.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
  g.stroke();
  if (S.ball[idx]) { g.fillStyle = "#d4ec3a"; g.beginPath(); g.arc(X(S.ball[idx][0]), Y(S.ball[idx][1]), 6 * d, 0, 7); g.fill(); }
}

// ---------- calibration ----------
const calKey = () => `pv-cal-${vw()}x${vh()}`;
function savedCal() { try { return JSON.parse(localStorage.getItem(calKey())); } catch { return null; } }

function startCalibration() {
  Object.assign(S, { calibrating: true, cal: {}, calIdx: 0, court: null });
  $("calbar").hidden = false; $("cal-confirm").hidden = true; $("cal-err").hidden = true; $("cal-ask").hidden = false; $("cal-tools").hidden = false;
  $("cal-saved").hidden = !savedCal();
  updateCal();
}
function updateCal() {
  const n = Object.keys(S.cal).length;
  $("cal-name").textContent = S.calIdx < NAMES.length ? NAMES[S.calIdx] : "(all asked)";
  $("cal-count").textContent = `${n} point${n === 1 ? "" : "s"}`;
  $("cal-done").disabled = n < 4;
  drawOverlay(0);
}
function finishCalibration(court) {
  S.court = court; S.calibrating = false;
  $("cal-ask").hidden = true; $("cal-tools").hidden = true; $("cal-confirm").hidden = false;
  drawOverlay(0);
}
// Press, drag to line up the crosshair in the magnifier, release to place the point.
let aim = null;  // [x, y] in video pixels while a finger or mouse is down
function toVideo(e) {
  fitCanvas();
  const r = overlay.getBoundingClientRect(), d = window.devicePixelRatio || 1, { s, ox, oy } = mapping();
  return [((e.clientX - r.left) * d - ox) / s, ((e.clientY - r.top) * d - oy) / s];
}
function drawLoupe() {
  if (!aim) return;
  const g = overlay.getContext("2d"), d = window.devicePixelRatio || 1, { s, ox, oy } = mapping();
  const R = 60 * d, zoom = 3, cx = ox + aim[0] * s, cy = Math.max(R + 4 * d, oy + aim[1] * s - R - 40 * d);
  const src = R / (s * zoom);  // half-size of the magnified area, in video pixels
  g.save(); g.beginPath(); g.arc(cx, cy, R, 0, 7); g.clip();
  g.drawImage(video, aim[0] - src, aim[1] - src, 2 * src, 2 * src, cx - R, cy - R, 2 * R, 2 * R);
  g.restore();
  g.strokeStyle = "#fff"; g.lineWidth = 2 * d; g.beginPath(); g.arc(cx, cy, R, 0, 7); g.stroke();
  g.strokeStyle = "#e0623a"; g.lineWidth = 1.5 * d; g.beginPath();
  g.moveTo(cx - 12 * d, cy); g.lineTo(cx + 12 * d, cy); g.moveTo(cx, cy - 12 * d); g.lineTo(cx, cy + 12 * d); g.stroke();
}
const inFrame = ([x, y]) => x >= 0 && y >= 0 && x <= vw() && y <= vh();
overlay.addEventListener("pointerdown", e => {
  if (!S.calibrating || S.calIdx >= NAMES.length || !vw()) return;
  try { overlay.setPointerCapture(e.pointerId); } catch { /* keep going without capture */ }
  aim = toVideo(e); drawOverlay(0); drawLoupe();
});
overlay.addEventListener("pointermove", e => { if (aim) { aim = toVideo(e); drawOverlay(0); drawLoupe(); } });
overlay.addEventListener("pointerup", () => {
  if (!aim) return;
  const p = aim; aim = null;
  if (inFrame(p) && S.calIdx < NAMES.length) { S.cal[NAMES[S.calIdx]] = p.map(Math.round); S.calIdx++; }
  updateCal();
});
overlay.addEventListener("pointercancel", () => { aim = null; drawOverlay(0); });
$("cal-skip").onclick = () => { if (S.calIdx < NAMES.length) S.calIdx++; updateCal(); };
$("cal-undo").onclick = () => {
  const keys = Object.keys(S.cal);
  if (!keys.length) return;
  const last = keys[keys.length - 1]; delete S.cal[last]; S.calIdx = NAMES.indexOf(last); updateCal();
};
$("cal-done").onclick = () => {
  try { finishCalibration(Court.fromLandmarks(S.cal)); $("cal-err").hidden = true; }
  catch (e) { $("cal-err").textContent = e.message; $("cal-err").hidden = false; }
};
$("cal-saved").onclick = () => { const c = savedCal(); finishCalibration(new Court(c.px, c.m)); };
$("cal-redo").onclick = startCalibration;
$("cal-ok").onclick = () => {
  try { localStorage.setItem(calKey(), JSON.stringify({ px: S.court.pointsPx, m: S.court.pointsM })); } catch { /* storage blocked: fine */ }
  $("calbar").hidden = true;
  if (S.mode === "live") runLive(); else { $("clip-run").disabled = false; drawOverlay(0); }
};

// ---------- shared session ----------
function resetSession() {
  Object.assign(S, { tracker: new Tracker(), players: [], ball: [], result: null, replay: false });
  $("results").hidden = true;
}
function pushFrame(det) {
  S.players.push(S.tracker.update(onCourt(det.people, S.court)));
  S.ball.push(det.ball);
}
function runAnalysis(live) {
  // Live inference runs at a few fps, so scale the event windows (defined at 30 fps) to the real rate.
  const opts = live ? { window: Math.max(2, Math.round(S.fps / 6)), minGap: Math.max(2, Math.round(S.fps / 5)) } : {};
  S.result = analyse(S.players, S.ball, S.court, S.fps, vw(), opts);
  return S.result;
}

// ---------- live ----------
async function startCamera() {
  showError("");
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
  } catch (e) {
    showError("The camera couldn't start. Allow camera access for this site in your browser settings, then try again.");
    return;
  }
  video.srcObject = S.stream; video.controls = false;
  await video.play();
  $("placeholder").hidden = true; $("cam-start").hidden = true; $("cam-stop").hidden = false;
  ensureModel().catch(() => {});
  try { S.wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* optional */ }
  startCalibration();
}
async function runLive() {
  await ensureModel();
  resetSession();
  S.running = true; S.t0 = performance.now(); $("results").hidden = false;
  let lastAnalysis = 0;
  while (S.running) {
    const det = await runDetect();
    if (!S.running || !S.court) break;  // stopped or switched tabs while detecting
    pushFrame(det);
    const el = (performance.now() - S.t0) / 1000;
    S.fps = S.players.length / Math.max(el, 0.001);
    drawOverlay(S.players.length - 1);
    if (el - lastAnalysis > 1) {
      lastAnalysis = el;
      runAnalysis(true); renderResults(S.players.length - 1);
      setStatus(`Live · ${S.fps.toFixed(1)} fps · ${detectorBackend() === "webgpu" ? "GPU" : "CPU"}`);
    }
    await new Promise(r => requestAnimationFrame(r));
  }
}
function stopCamera() {
  S.running = false;
  S.stream?.getTracks().forEach(t => t.stop()); S.stream = null; video.srcObject = null;
  S.wakeLock?.release?.(); S.wakeLock = null;
  $("cam-start").hidden = false; $("cam-stop").hidden = true; $("calbar").hidden = true; S.calibrating = false;
  if (S.players.length) { runAnalysis(true); renderResults(S.players.length - 1); }
}
$("cam-start").onclick = startCamera;
$("cam-stop").onclick = stopCamera;

// ---------- clip ----------
const seek = t => new Promise(res => { video.addEventListener("seeked", res, { once: true }); video.currentTime = t; });

async function loadClip(file) {
  showError("");
  if (!file) return;
  $("file-name").textContent = file.name;
  video.srcObject = null; video.src = URL.createObjectURL(file); video.controls = false;
  try {
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = rej; });
  } catch {
    showError("This browser can't play that video. iPhone .MOV (HEVC) files need Safari or Chrome on a Mac; otherwise convert it to MP4 (H.264).");
    return;
  }
  $("placeholder").hidden = true; $("clip-options").hidden = false; $("clip-run").hidden = false; $("clip-run").disabled = true;
  $("t-start").max = Math.floor(video.duration); $("t-end").max = Math.ceil(video.duration);
  $("t-start").value = 0; $("t-end").value = Math.min(60, Math.floor(video.duration));
  updateEta();
  await seek(0);
  ensureModel().catch(() => {});
  startCalibration();
}
function clipRange() {
  const a = Math.max(0, +$("t-start").value || 0), b = Math.min(video.duration || 0, +$("t-end").value || 0);
  return [a, Math.max(a, b)];
}
let msPerFrame = 500;  // first guess (High detail); replaced by the measured rate
function updateEta() {
  const [a, b] = clipRange(), n = Math.round((b - a) * +$("rate").value);
  $("clip-eta").textContent = n ? `${n} frames · about ${Math.ceil(n * msPerFrame / 60000)} min` : "";
}
["t-start", "t-end", "rate"].forEach(id => $(id).addEventListener("input", updateEta));
$("t-start").addEventListener("change", () => { if (vw()) seek(clipRange()[0]).then(() => drawOverlay(0)); });

function setStep(name) {
  let seen = false;
  for (const li of $("steps").children) {
    const me = li.dataset.step === name;
    li.className = me ? "active" : seen ? "" : "done";
    if (me) seen = true;
  }
  if (name === "end") for (const li of $("steps").children) li.className = "done";
}

// Detections for evenly spaced frames from a to b. Playing slowly and catching frames is far faster
// than seeking frame by frame (each seek decodes from the previous keyframe).
async function sampleClip(a, b, n, onDet) {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) {  // old browsers: seek per frame
    for (let i = 0; i < n && !S.cancel; i++) { await seek(a + i / S.rate); onDet(i, await runDetect()); }
    return;
  }
  await seek(a);
  video.muted = true; video.playbackRate = 0.25;
  let busy = false;
  await new Promise(resolve => {
    const onFrame = async (now, meta) => {
      if (S.cancel || meta.mediaTime >= b || video.ended) { video.pause(); resolve(); return; }
      video.requestVideoFrameCallback(onFrame);
      if (busy) return;
      busy = true;
      const t1 = performance.now();
      const det = await runDetect();  // the frame is captured before the first await inside detect
      onDet(Math.round((meta.mediaTime - a) * S.rate), det);
      // Advance one grid step of video per detection.
      const sec = (performance.now() - t1) / 1000;
      video.playbackRate = Math.min(4, Math.max(0.0625, (1 / S.rate) / Math.max(sec, 0.001)));
      busy = false;
    };
    video.requestVideoFrameCallback(onFrame);
    video.addEventListener("ended", resolve, { once: true });
    video.play().catch(resolve);
  });
  video.playbackRate = 1;
}

async function runClip() {
  await ensureModel();
  resetSession();
  const [a, b] = clipRange();
  S.start = a; S.rate = S.fps = +$("rate").value; S.cancel = false;
  const n = Math.max(1, Math.round((b - a) * S.rate));
  $("progress").hidden = false; $("clip-cancel").hidden = false; $("clip-run").disabled = true; setStep("frames");
  const t0 = performance.now(), grid = new Array(n).fill(null);
  let got = 0;
  setStep("detect");
  await sampleClip(a, b, n, (i, det) => {
    if (i < 0 || i >= n || grid[i]) return;
    grid[i] = det; got++;
    msPerFrame = (performance.now() - t0) / got;
    if (got % 10 === 0) {
      $("bar").value = i / n;
      $("eta").textContent = `${i + 1} / ${n} frames · ${Math.ceil((n - i) * msPerFrame / 1000)} s left`;
    }
  });
  // Grid slots the sampler stepped over keep the last players and have no ball (short gaps get interpolated).
  const last = grid.findLastIndex(Boolean);
  let prev = { people: [], ball: null };
  for (let i = 0; i <= last; i++) { const det = grid[i] || { people: prev.people, ball: null }; pushFrame(det); prev = det; }
  if (!S.players.length) { $("progress").hidden = true; $("clip-run").disabled = false; return; }
  setStep("events"); await new Promise(r => setTimeout(r, 30));
  runAnalysis(false);
  setStep("notes"); await new Promise(r => setTimeout(r, 30));
  setStep("end"); $("bar").value = 1; $("clip-cancel").hidden = true;
  const secs = Math.round((performance.now() - t0) / 1000);
  $("eta").textContent = (S.cancel ? `Stopped early: ${S.players.length} frames. ` : "") + `Analyzed ${got} of ${S.players.length} frames in ${secs} s.`;
  $("clip-run").disabled = false; $("clip-run").textContent = "Analyze again";
  S.replay = true; video.controls = true;
  await seek(a); renderResults(0); drawOverlay(0);
}
const replayIdx = () => Math.round((video.currentTime - S.start) * S.rate);
function onReplay() {
  if (!S.replay) return;
  const i = Math.min(Math.max(replayIdx(), 0), S.players.length - 1);
  drawOverlay(i); renderNow(i); drawCourt(i);
  if (!video.paused) video.requestVideoFrameCallback ? video.requestVideoFrameCallback(onReplay) : requestAnimationFrame(onReplay);
}
video.addEventListener("play", onReplay);
video.addEventListener("seeked", () => { if (S.replay) onReplay(); });

$("file").addEventListener("change", e => loadClip(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); loadClip(e.dataTransfer.files[0]); });
$("clip-run").onclick = runClip;
$("clip-cancel").onclick = () => { S.cancel = true; };

// ---------- results ----------
const frameTime = i => (S.mode === "live" ? i / S.fps : S.start + i / S.rate);

function renderNow(idx) {
  const r = S.result; if (!r) return;
  const shot = [...r.shots].reverse().find(s => s.frame <= idx);
  const el = $("now");
  if (shot && (idx - shot.frame) / r.fps < 2.5) {
    el.innerHTML = `P${shot.player} · ${shot.type}${shot.speed_kmh ? ` · ${Math.round(shot.speed_kmh)} km/h` : ""}` +
      shot.faults.map(f => ` <span class="fault">· ${f.replaceAll("_", " ")}</span>`).join("");
  } else el.textContent = "";
}

function drawCourt(idx) {
  const c = $("court"), g = c.getContext("2d"), d = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
  if (!r.width) return;
  c.width = Math.round(r.width * d); c.height = Math.round(r.height * d);
  const pad = 16 * d, s = Math.min((c.width - 2 * pad) / COURT_W, (c.height - 2 * pad) / (COURT_L + 2));
  const ox = (c.width - COURT_W * s) / 2, oy = (c.height - COURT_L * s) / 2, X = x => ox + x * s, Y = y => oy + y * s;
  g.fillStyle = css("--panel"); g.fillRect(0, 0, c.width, c.height);
  g.fillStyle = css("--court"); g.fillRect(X(0), Y(0), COURT_W * s, COURT_L * s);
  g.fillStyle = css("--kitchen"); g.fillRect(X(0), Y(NET_Y - KITCHEN), COURT_W * s, 2 * KITCHEN * s);
  g.strokeStyle = css("--court-line"); g.lineWidth = 2 * d; g.beginPath();
  for (const [a, b] of COURT_LINES) { g.moveTo(X(a[0]), Y(a[1])); g.lineTo(X(b[0]), Y(b[1])); }
  g.stroke();
  const res = S.result;
  if (res) for (const e of res.events) {
    const age = (idx - e.frame) / res.fps;
    if (e.type !== "bounce" || age < 0 || age > 3) continue;
    const [x, y] = S.court.toCourt(e.px);
    g.globalAlpha = 1 - age / 3; g.fillStyle = css("--ball"); g.beginPath(); g.arc(X(x), Y(y), 5 * d, 0, 7); g.fill();
  }
  g.globalAlpha = 1;
  for (const [pid, p] of Object.entries(S.players[idx] || {})) {
    g.fillStyle = color(pid); g.beginPath(); g.arc(X(p.court[0]), Y(p.court[1]), 7 * d, 0, 7); g.fill();
    g.fillStyle = css("--ink"); g.font = `600 ${11 * d}px IBM Plex Sans, sans-serif`; g.fillText("P" + pid, X(p.court[0]) + 9 * d, Y(p.court[1]) + 4 * d);
  }
}

function renderResults(idx) {
  const r = S.result; if (!r) return;
  $("results").hidden = false;
  const faults = r.shots.reduce((a, s) => a + s.faults.length, 0);
  $("facts").innerHTML = [["Analyzed", fmt(r.frames / r.fps)], ["Rallies", r.rallies.length], ["Shots", r.shots.length],
    ["Faults", faults], ["Ball tracked", r.ballPct + "%"]].map(([k, v]) => `<div class="fact"><span>${k}</span><b>${v}</b></div>`).join("");
  $("notes").innerHTML = (r.notes.length ? r.notes : ["Keep playing: notes appear once a few shots are detected."]).map(n => `<li>${n}</li>`).join("");
  $("shots").innerHTML = `<tr><th class="num">Time</th><th>Player</th><th>Shot</th><th class="num">km/h</th><th>Faults</th></tr>` +
    [...r.shots].reverse().map(s => `<tr class="${S.replay ? "jump" : ""}" data-f="${s.frame}"><td class="num">${fmt(frameTime(s.frame))}</td>` +
      `<td><span class="sw" style="background:${color(s.player)}"></span>P${s.player}</td>` +
      `<td><span class="tag" style="background:${TYPE_COLORS[s.type]}">${s.type}</span>${s.volley ? " volley" : ""}</td>` +
      `<td class="num">${s.speed_kmh ?? "–"}</td><td>${s.faults.map(f => f.replaceAll("_", " ")).join(", ") || "–"}</td></tr>`).join("");
  if (S.replay) for (const tr of $("shots").querySelectorAll("tr[data-f]")) tr.onclick = () => seek(Math.max(S.start, frameTime(+tr.dataset.f) - 1));
  const pids = Object.keys(r.players).concat(Object.keys(r.summary).filter(p => !(p in r.players)));
  $("players").innerHTML = `<tr><th>Player</th><th>Side</th><th class="num">Seen s</th><th class="num">Distance m</th><th class="num">Avg m/s</th><th class="num">Kitchen %</th><th class="num">Shots</th><th class="num">Avg km/h</th></tr>` +
    pids.map(p => { const a = r.players[p] || {}, b = r.summary[p] || {};
      return `<tr><td><span class="sw" style="background:${color(p)}"></span>P${p}</td><td>${a.side ?? "–"}</td><td class="num">${a.seconds ?? "–"}</td><td class="num">${a.distance_m ?? "–"}</td>` +
        `<td class="num">${a.avg_speed_ms ?? "–"}</td><td class="num">${a.kitchen_pct ?? "–"}</td><td class="num">${b.shots ?? 0}</td><td class="num">${b.avg_shot_kmh ?? "–"}</td></tr>`; }).join("");
  renderNow(idx); drawCourt(idx);
}

$("download").onclick = () => {
  const r = S.result; if (!r) return;
  const out = { app: "Pickleball-Vision web", mode: S.mode, fps: r.fps, frames: r.frames, start_s: S.mode === "clip" ? S.start : 0,
    court: { points_px: S.court.pointsPx, points_m: S.court.pointsM }, ball_pct: r.ballPct,
    players: r.players, shot_summary: r.summary, rallies: r.rallies, shots: r.shots, events: r.events, notes: r.notes };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = `pickleball-vision-${S.mode}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  a.click();
};

// ---------- tabs ----------
function setMode(mode) {
  if (S.running) stopCamera();
  S.mode = mode; S.calibrating = false; S.court = null; S.replay = false; S.cancel = true;
  // Live needs speed; recorded clips can afford the slower, more thorough ball search.
  S.size = mode === "live" ? 640 : 1280; $("detail").value = S.size; msPerFrame = S.size === 1280 ? 500 : 130;
  $("tab-live").setAttribute("aria-selected", mode === "live"); $("tab-clip").setAttribute("aria-selected", mode === "clip");
  for (const id of ["live-intro", "live-controls"]) $(id).hidden = mode !== "live";
  for (const id of ["clip-intro", "clip-controls"]) $(id).hidden = mode !== "clip";
  $("clip-run").hidden = mode !== "clip" || !video.src; $("progress").hidden = true; $("calbar").hidden = true; $("results").hidden = true;
  video.removeAttribute("src"); video.srcObject = null; video.controls = false; video.load();
  $("placeholder").hidden = false; $("placeholder").textContent = mode === "live" ? "Start the camera to begin." : "Choose a video to begin.";
  $("clip-options").hidden = true; $("clip-run").hidden = true; showError("");
  drawOverlay(0);
}
$("tab-live").onclick = () => setMode("live");
$("tab-clip").onclick = () => setMode("clip");
window.addEventListener("resize", () => { drawOverlay(S.replay ? replayIdx() : S.players.length - 1); if (S.result) drawCourt(S.replay ? replayIdx() : S.players.length - 1); });
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { if (S.result) drawCourt(S.replay ? replayIdx() : S.players.length - 1); });
if (location.hash === "#clip") setMode("clip");
if (!navigator.mediaDevices?.getUserMedia) $("cam-start").disabled = true;

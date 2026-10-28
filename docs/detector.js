// YOLOv8n (COCO) in the browser via onnxruntime-web: people, ball ("sports ball") and paddles ("tennis racket").
// Two exports of the same weights: 640 px input (fast) and 1280 px (about 4x slower, finds much smaller balls).
/* global ort */

const PERSON = 0, BALL = 32, RACKET = 38;
const CONF = { [PERSON]: 0.3, [BALL]: 0.15, [RACKET]: 0.3 };
const sessions = {};  // size -> {session, canvas, ctx, input}
let backend = "";

export async function loadDetector(size = 640, onStatus = () => {}) {
  if (sessions[size]) return backend;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const url = `models/yolov8n-${size}.onnx`;
  for (const ep of navigator.gpu ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      onStatus(`Loading ${size} px model (${ep === "webgpu" ? "GPU" : "CPU"})…`);
      const session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = size;
      sessions[size] = { session, canvas, ctx: canvas.getContext("2d", { willReadFrequently: true }), input: new Float32Array(3 * size * size) };
      backend = ep;
      return ep;
    } catch (e) {
      console.warn(`onnxruntime ${ep} failed`, e);
    }
  }
  throw new Error("This browser can't run the detection model. Try a recent Chrome, Edge or Safari.");
}

export const detectorBackend = () => backend;

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h);
}

function nms(boxes, thr = 0.5) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of boxes) if (keep.every(k => iou(k.bbox, b.bbox) < thr)) keep.push(b);
  return keep;
}

/** source: <video> or <canvas>; w/h its pixel size. -> {people, ball, paddles} in source pixels. */
export async function detect(source, w, h, size = 640) {
  const { session, ctx, input } = sessions[size];
  const scale = Math.min(size / w, size / h), nw = Math.round(w * scale), nh = Math.round(h * scale);
  const dx = (size - nw) / 2, dy = (size - nh) / 2;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, 0, 0, w, h, dx, dy, nw, nh);  // the frame is captured here, before any await
  const px = ctx.getImageData(0, 0, size, size).data, plane = size * size;
  for (let i = 0, j = 0; i < plane; i++, j += 4) {
    input[i] = px[j] / 255; input[plane + i] = px[j + 1] / 255; input[2 * plane + i] = px[j + 2] / 255;
  }
  const out = await session.run({ images: new ort.Tensor("float32", input, [1, 3, size, size]) });
  const res = out[session.outputNames[0]], o = res.data, N = res.dims[2];  // [1, 84, N]: cx, cy, w, h, 80 scores
  const found = { [PERSON]: [], [BALL]: [], [RACKET]: [] };
  for (let i = 0; i < N; i++) {
    for (const c of [PERSON, BALL, RACKET]) {
      const s = o[(4 + c) * N + i];
      if (s < CONF[c]) continue;
      const cx = (o[i] - dx) / scale, cy = (o[N + i] - dy) / scale, bw = o[2 * N + i] / scale, bh = o[3 * N + i] / scale;
      found[c].push({ conf: s, bbox: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2] });
    }
  }
  const balls = nms(found[BALL]);
  const ball = balls.length ? [(balls[0].bbox[0] + balls[0].bbox[2]) / 2, (balls[0].bbox[1] + balls[0].bbox[3]) / 2].map(v => Math.round(v * 10) / 10) : null;
  return { people: nms(found[PERSON]), ball, paddles: nms(found[RACKET]).map(b => b.bbox) };
}

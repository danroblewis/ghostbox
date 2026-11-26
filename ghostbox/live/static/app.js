// Ghostbox Live client.
// Timeline: all times are seconds since the server's epoch. The waterfall is drawn a fixed
// LAG behind live (analysis needs ~settle+hop of future), so every view scrolls smoothly.
// Audio for stream time t plays at client time t + offset + delay.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k !== null && k !== undefined && k !== "") e.append(k);
  return e;
};
const fmtF = hz => hz >= 1e6 ? (hz / 1e6).toFixed(4) + " MHz" : (hz / 1e3).toFixed(1) + " kHz";
const fmtFshort = hz => hz >= 1e6 ? (hz / 1e6).toFixed(3) + " MHz" : (hz / 1e3).toFixed(1) + " kHz";
const nowS = () => performance.now() / 1000;
const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };
const b64bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const fillRange = r => r.style.setProperty("--fill", ((r.value - r.min) / (r.max - r.min) * 100) + "%");
const pad2 = n => String(n).padStart(2, "0");
const hms = d => `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
const clockTime = t => S.epoch ? hms(new Date((S.epoch + t) * 1000)) : "";

// ---------------- state ----------------
var SV = { state: { state: "idle" }, meta: null, file: null, imgs: {}, saved: [], open: false };
const S = {
  cfg: null, checks: [], presets: {}, sens: {}, views: [], status: {},
  epoch: 0, fmin: 0, fmax: 1, fr: 25, hop: 1, lag: 2.75,
  offsets: [], offset: null,
  view: lsGet("gb_view", "power"), chunks: new Map(),   // t0 key -> chunk (current view)
  dets: new Map(),                                      // id -> {box, ended, playing:[[a,b]], pcm:[]}
  recs: [], tx: new Map(), chainTx: new Map(),
  delay: +lsGet("gb_delay", 4), pps: +lsGet("gb_pps", 25),
  follow: true, viewTop: 0,
  smin: 0, smax: 255, cmap: lsGet("gb_cmap", "magma"),
  playing: null,                                        // {key, src, start, dur}
  minConf: +lsGet("gb_minconf", 0.55),                  // display filter: waterfall boxes and the list
};
const cv = $("#wf"), ctx = cv.getContext("2d");
const TH = {};                                          // canvas colours, read from the active theme
function readTheme() {
  const cs = getComputedStyle(document.documentElement), v = k => cs.getPropertyValue(k).trim();
  Object.assign(TH, { bg: v("--wf-bg") || "#0c0d0e", grid: v("--wf-grid") || "rgba(255,255,255,.06)", text: v("--wf-label") || "#aaa",
    mark: v("--mark") || "#e3a33b", accent: v("--accent") || "#ddd", panel: v("--panel") || "#1b1c1e",
    fg: v("--text") || "#e4e4e2", fg2: v("--text-2") || "#a2a3a3",
    mono: v("--mono") || "ui-monospace,monospace", sans: v("--font") || "system-ui,sans-serif" });
}
readTheme();
const THEMES = window.GB_THEMES;                        // defined in index.html
function setTheme(name) {
  if (!THEMES.some(([k]) => k === name)) name = "instrument";
  lsSet("gb_theme", name);
  const link = $("#themeCss"); link.onload = readTheme; link.href = `/static/themes/${name}.css`;
  // night vision only works if the waterfall is red too; remember the user's map to restore later
  if (name === "nightvision" && S.cmap !== "red") { lsSet("gb_cmap_day", S.cmap); setCmap("red"); }
  else if (name !== "nightvision" && S.cmap === "red") setCmap(lsGet("gb_cmap_day", "magma"));
}
function setCmap(name) { S.cmap = name; lsSet("gb_cmap", name); $("#cmap").value = name; repaintAll(); SCOPE.repaint = true; if (SV.open) drawSurvey(); if (DT.open) drawDetailSpec(); }

// ---------------- colour maps ----------------
const CMAPS = {
  magma: [[0, 4, 4, 16], [0.25, 50, 18, 92], [0.5, 150, 40, 110], [0.75, 245, 110, 60], [1, 252, 250, 180]],
  viridis: [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140], [0.75, 94, 201, 98], [1, 253, 231, 37]],
  gray: [[0, 0, 0, 0], [1, 235, 235, 235]],
  red: [[0, 0, 0, 0], [0.45, 90, 6, 3], [0.8, 200, 30, 20], [1, 255, 70, 50]],
  // Classic SDR# rainbow: dark blue, cyan, green, yellow, red, white
  rainbow: [[0, 0, 0, 30], [0.14, 0, 0, 150], [0.3, 0, 170, 255], [0.45, 0, 215, 70], [0.62, 250, 240, 0], [0.8, 255, 40, 0], [1, 255, 255, 255]],
  // Turbo / jet: black-navy, blue, cyan, yellow, deep red
  turbo: [[0, 20, 10, 50], [0.12, 50, 60, 190], [0.27, 40, 150, 250], [0.4, 30, 225, 210], [0.53, 110, 250, 100], [0.66, 225, 225, 50], [0.8, 250, 140, 30], [0.92, 205, 45, 10], [1, 125, 5, 5]],
  // Inferno: black, deep purple, crimson, orange, pale yellow (noise floor near true black)
  inferno: [[0, 0, 0, 4], [0.2, 40, 11, 84], [0.4, 101, 21, 110], [0.55, 159, 42, 99], [0.7, 212, 72, 66], [0.85, 245, 125, 21], [1, 252, 255, 164]],
  // Monochrome phosphor
  amber: [[0, 0, 0, 0], [0.45, 95, 52, 0], [0.8, 220, 140, 10], [1, 255, 205, 110]],
  green: [[0, 0, 0, 0], [0.45, 10, 80, 20], [0.8, 50, 200, 70], [1, 170, 255, 170]],
};
function makeLUT() {
  const stops = CMAPS[S.cmap] || CMAPS.magma, base = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255; let j = 0; while (j < stops.length - 2 && x > stops[j + 1][0]) j++;
    const [a, b] = [stops[j], stops[j + 1]]; const t = (x - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) base[i * 4 + c] = a[c + 1] + (b[c + 1] - a[c + 1]) * t;
    base[i * 4 + 3] = 255;
  }
  const out = new Uint8ClampedArray(256 * 4), span = Math.max(1, S.smax - S.smin);   // smin/smax may be fractional (auto)
  for (let v = 0; v < 256; v++) {
    const i = Math.max(0, Math.min(255, Math.round((v - S.smin) / span * 255)));
    out.set(base.subarray(i * 4, i * 4 + 4), v * 4);
  }
  return out;
}
if (!CMAPS[S.cmap]) S.cmap = "magma";                   // e.g. a map from an older version
let LUT = makeLUT();

// ---------------- waterfall chunks ----------------
function paintChunk(c) {
  const img = c.ctx.createImageData(c.w, c.n);
  for (let r = 0; r < c.n; r++) {                 // newest row at the top of the chunk canvas
    const src = (c.n - 1 - r) * c.w, dst = r * c.w * 4;
    for (let x = 0; x < c.w; x++) { const v = c.vals[src + x] * 4, o = dst + x * 4; img.data[o] = LUT[v]; img.data[o + 1] = LUT[v + 1]; img.data[o + 2] = LUT[v + 2]; img.data[o + 3] = 255; }
  }
  c.ctx.putImageData(img, 0, 0);
}
// Chunks are painted with the current colour scale; when the scale changes, visible chunks are
// repainted lazily while drawing (a few per frame), so continuous auto-scaling stays cheap.
S.lutVer = 0; S.autoScale = lsGet("gb_autoscale", "1") === "1"; S.visible = [];
function addChunk(m) {
  if ((m.mode || "power") !== S.view) return;
  const key = m.t0.toFixed(3);
  if (S.chunks.has(key)) return;
  const c = { t0: m.t0, dt: m.dt, n: m.n, w: m.w, fmin: m.fmin, fmax: m.fmax, vals: b64bytes(m.rows) };
  c.cv = document.createElement("canvas"); c.cv.width = c.w; c.cv.height = c.n; c.ctx = c.cv.getContext("2d");
  paintChunk(c); c.ver = S.lutVer;
  S.chunks.set(key, c);
}
function repaintAll() { LUT = makeLUT(); S.lutVer++; }
function autoScale() {                                     // follow what's on screen: noise floor (20th pct) .. peaks (99.8th)
  if (!S.autoScale || !S.visible.length) return;
  const v = [];
  for (const c of S.visible) for (let i = 0; i < c.vals.length; i += 23) v.push(c.vals[i]);
  if (v.length < 200) return;
  v.sort((a, b) => a - b);
  const lo = v[Math.floor(v.length * 0.20)], hi = Math.max(lo + 8, v[Math.floor(v.length * 0.998)]);
  const nlo = S.smin + 0.35 * (lo - S.smin), nhi = S.smax + 0.35 * (hi - S.smax);
  if (Math.abs(nlo - S.smin) >= 0.75 || Math.abs(nhi - S.smax) >= 0.75) { S.smin = nlo; S.smax = nhi; repaintAll(); }
}
setInterval(autoScale, 400);
function evictChunks(top) {
  const keepFrom = top - 15 * 60, keepTo = top + 120;
  for (const [k, c] of S.chunks) if (c.t0 < keepFrom || c.t0 > keepTo) S.chunks.delete(k);
}
let histBusy = false, histTimer = null;
async function fetchHistory(t0, t1) {
  if (histBusy) return; histBusy = true;
  try {
    const r = await (await fetch(`/api/history?view=${S.view}&t0=${t0}&t1=${t1}`)).json();
    for (const c of r.chunks) addChunk(c);
    for (const b of r.boxes) if (!S.dets.has(b.id)) S.dets.set(b.id, { box: b, ended: true, playing: [], pcm: [] });
  } catch (e) {} finally { histBusy = false; }
}
function ensureHistory(top, H) {
  clearTimeout(histTimer);
  histTimer = setTimeout(() => {
    const span = H / S.pps, t0 = top - span - 10;
    let covered = 0; for (const c of S.chunks.values()) if (c.t0 + c.n * c.dt > t0 && c.t0 < top) covered += c.n * c.dt;
    if (covered < (top - t0) * 0.8) fetchHistory(t0 - 30, top + 2);
  }, 150);
}

// ---------------- time ----------------
function liveTop() { return S.offset === null ? null : nowS() - S.offset - S.lag; }
// The playhead is the stream time being heard: live minus S.delay. Live listening is just a small delay;
// clicking the waterfall moves the playhead anywhere in history and playback continues from there.
// While following, the view keeps the playhead line S.lineY seconds below the top edge.
function playheadTime() { return S.paused ? S.pausedAt : nowS() - S.offset - S.delay; }
function setPaused(p) {
  if (p === !!S.paused || S.offset === null) return;
  if (p) { S.pausedAt = playheadTime(); S.paused = true; }
  else { S.delay = Math.max(S.lag, nowS() - S.offset - S.pausedAt); S.paused = false; }
  $("#pp").innerHTML = S.paused ? ICON_PLAY : ICON_PAUSE; $("#pp").title = S.paused ? "Resume playback (Space)" : "Pause playback (Space)";
  $("#pp").classList.toggle("on", S.paused);
  restartPlayback(); updateNav();
}
function topTime() {
  const live = liveTop(); if (live === null) return null;
  return S.follow ? Math.min(live, playheadTime() + S.lineY) : S.viewTop;
}
function syncClock(t) {
  if (typeof t !== "number") return;
  S.offsets.push(nowS() - t); if (S.offsets.length > 20) S.offsets.shift();
  S.offset = Math.min(...S.offsets);
}
const xOf = f => (f - S.fmin) / (S.fmax - S.fmin) * cv.width;

// ---------------- draw ----------------
function draw() {
  requestAnimationFrame(draw);
  const dpr = devicePixelRatio, W = cv.clientWidth * dpr, H = cv.clientHeight * dpr;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  const top = topTime(); if (top === null) return;
  const pps = S.pps * dpr, span = H / pps;
  ctx.imageSmoothingEnabled = false;
  S.visible = []; let repaints = 0;
  for (const c of S.chunks.values()) {
    const tEnd = c.t0 + c.n * c.dt;
    if (tEnd < top - span || c.t0 > top) continue;
    S.visible.push(c);
    if (c.ver !== S.lutVer && repaints < 16) { paintChunk(c); c.ver = S.lutVer; repaints++; }
    const x0 = xOf(c.fmin), x1 = xOf(c.fmax);
    ctx.drawImage(c.cv, x0, (top - tEnd) * pps, x1 - x0, c.n * c.dt * pps + 0.6);
  }
  // grid: clock times on the left, frequency across the top
  ctx.font = `${11 * dpr}px ${TH.mono}`;
  ctx.strokeStyle = TH.grid; ctx.fillStyle = TH.text; ctx.lineWidth = 1;
  const tick = span > 120 ? 30 : span > 50 ? 10 : 5;
  for (let t = Math.floor(top / tick) * tick; t > top - span; t -= tick) {
    const y = (top - t) * pps; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    if (y > 30 * dpr) ctx.fillText(clockTime(t), 6 * dpr, y - 4 * dpr);
  }
  const step = niceStep((S.fmax - S.fmin) / Math.max(2, Math.min(7, cv.clientWidth / 120)));   // fewer labels on narrow screens
  for (let f = Math.ceil(S.fmin / step) * step; f <= S.fmax; f += step) {
    const x = xOf(f); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillText(fmtFshort(f), x + 4 * dpr, 14 * dpr);
  }
  // marks: moments someone bookmarked
  ctx.save(); ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.strokeStyle = TH.text; ctx.fillStyle = TH.text; ctx.globalAlpha = 0.8; ctx.lineWidth = dpr;
  for (const m of S.marks || []) {
    const y = (top - m.t) * pps; if (y < 0 || y > H) continue;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillText("Mark " + clockTime(m.t), W - 110 * dpr, y - 4 * dpr);
  }
  ctx.restore();
  // detections
  const now = nowS(), chains = new Map();
  for (const d of S.dets.values()) {
    if (!audible(d)) { d.box._rect = null; continue; }      // below the filter: not drawn, not heard
    const b = d.box; const y0 = (top - b.t1) * pps, y1 = (top - b.t0) * pps;
    if (y1 < -20 || y0 > H + 20) { d.box._rect = null; continue; }
    const pad = Math.max(5 * dpr, xOf(b.fmin + 3400) - xOf(b.fmin));
    const x0 = xOf(b.fmin) - pad, x1 = xOf(b.fmax) + pad;
    const playing = d.playing.some(([a, z]) => now >= a && now <= z);
    d.box._rect = [x0 / dpr, y0 / dpr, x1 / dpr, y1 / dpr];
    const k = Math.min(1, Math.max(0, b.conf));
    ctx.save();
    ctx.strokeStyle = TH.mark;
    if (playing) { ctx.globalAlpha = 0.16; ctx.fillStyle = TH.mark; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
    ctx.globalAlpha = playing ? 1 : 0.35 + 0.65 * k;
    ctx.lineWidth = (playing ? 2 : 1 + k * 0.6) * dpr; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.restore();
    if (b.path && b.path.length > 1 && b.fmax - b.fmin > 500) {
      ctx.strokeStyle = TH.mark; ctx.lineWidth = 1.2 * dpr; ctx.beginPath();
      b.path.forEach(([t, f], i) => { const x = xOf(f), y = (top - t) * pps; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    const ly = Math.max(14 * dpr, y0 + 11 * dpr);
    ctx.fillStyle = TH.mark; ctx.globalAlpha = playing ? 1 : 0.55 + 0.45 * k;
    ctx.fillText(`${fmtFshort((b.fmin + b.fmax) / 2)}  ${(b.conf * 100).toFixed(0)}%`, x1 + 5 * dpr, ly);
    ctx.globalAlpha = 1;
    const tx = S.tx.get(b.id);
    if (tx && tx.text && $("#caps").checked) caption(`“${tx.text}”`, x1 + 5 * dpr, ly + 16 * dpr, tx.no_speech_prob > 0.6, playing);
    if (!chains.has(b.chain)) chains.set(b.chain, []);
    chains.get(b.chain).push(b);
  }
  ctx.save(); ctx.setLineDash([2 * dpr, 4 * dpr]); ctx.strokeStyle = TH.mark; ctx.globalAlpha = 0.6; ctx.lineWidth = dpr;
  for (const list of chains.values()) {
    list.sort((a, b) => a.t0 - b.t0);
    for (let i = 1; i < list.length; i++) {
      const a = list[i - 1], b = list[i];
      ctx.beginPath(); ctx.moveTo(xOf(a.path[a.path.length - 1][1]), (top - a.t1) * pps); ctx.lineTo(xOf(b.path[0][1]), (top - b.t0) * pps); ctx.stroke();
    }
  }
  ctx.restore();
  if ($("#caps").checked) for (const [cid, list] of chains) {
    const tx = S.chainTx.get(cid); if (!tx || !tx.text || list.length < 2) continue;
    const b = list[list.length - 1];
    caption(`Chain: “${tx.text}”`, xOf(b.fmax) + 8 * dpr, (top - b.t0) * pps + 16 * dpr, tx.no_speech_prob > 0.6, false);
  }
  // playback line (draggable): the stream time being heard right now
  const yPlay = (top - playheadTime()) * pps;
  if (yPlay >= -2 && yPlay <= H + 2) {
    ctx.save(); ctx.strokeStyle = TH.mark; ctx.globalAlpha = dragging ? 1 : 0.8; ctx.lineWidth = (dragging ? 1.5 : 1) * dpr;
    ctx.setLineDash([6 * dpr, 4 * dpr]); ctx.beginPath(); ctx.moveTo(0, yPlay); ctx.lineTo(W, yPlay); ctx.stroke(); ctx.setLineDash([]);
    const back = S.delay < 60 ? `−${S.delay.toFixed(2)} s` : `−${Math.floor(S.delay / 60)}:${pad2(Math.floor(S.delay % 60))}`;
    const label = S.paused ? `Paused  ${clockTime(playheadTime())}` : S.delay < 15 ? `Playback  ${back}` : `Playback  ${clockTime(playheadTime())}  (${back})`;
    const lw = ctx.measureText(label).width;
    ctx.globalAlpha = 1; ctx.fillStyle = TH.bg; ctx.fillRect(W - lw - 18 * dpr, yPlay - 9 * dpr, lw + 12 * dpr, 18 * dpr);
    ctx.fillStyle = TH.mark; ctx.fillText(label, W - lw - 12 * dpr, yPlay + 4 * dpr);
    ctx.restore();
  }
  if (!S.follow) {
    ctx.fillStyle = TH.text;
    ctx.fillText(`History — ${(liveTop() - top).toFixed(0)} s before live`, 6 * dpr, H - 26 * dpr);
  }
  ensureHistory(top, H / dpr);
  drawScope();
  if (Math.random() < 0.01) evictChunks(top);
  updateRowProgress();
  updateVoiceGains();
}
function caption(text, x, y, doubtful, playing, color) {
  const dpr = devicePixelRatio;
  if (text.length > 90) text = text.slice(0, 88) + "…”";
  ctx.save();
  ctx.font = `${12 * dpr}px ${TH.sans}`;
  const w = ctx.measureText(text).width;
  if (x + w > cv.width - 6 * dpr) x = Math.max(4 * dpr, cv.width - w - 10 * dpr);
  ctx.globalAlpha = 0.85; ctx.fillStyle = TH.bg; ctx.fillRect(x - 4 * dpr, y - 13 * dpr, w + 8 * dpr, 18 * dpr); ctx.globalAlpha = 1;
  ctx.fillStyle = doubtful ? TH.text : (color || (playing ? TH.mark : TH.fg));
  ctx.fillText(text, x, y);
  ctx.restore();
}
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x))); const m = x / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }

// ---------------- waterfall interaction ----------------
let dragging = false, dragTop = 0, downAt = null;
function canvasPos(e) { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top, r]; }
function playLineY() { const top = topTime(); return top === null ? -1e9 : (top - playheadTime()) * S.pps; }
function boxAt(x, y) {
  for (const d of S.dets.values()) { const r = d.box._rect; if (r && x >= r[0] - 4 && x <= r[2] + 4 && y >= r[1] - 4 && y <= r[3] + 4) return d; }
  return null;
}
function setPlayhead(t, lineY) {
  if (S.paused) { S.paused = false; $("#pp").innerHTML = ICON_PAUSE; $("#pp").classList.remove("on"); }
  const maxBack = 60 * ((S.cfg && S.cfg.display.history_min) || 20);
  S.delay = Math.max(S.lag, Math.min(maxBack, Math.round((nowS() - S.offset - t) * 4) / 4));
  S.lineY = Math.max(0, Math.min(lineY, S.delay - S.lag));
  S.follow = true; updateNav();
}
function returnToLive() {
  if (S.paused) { S.paused = false; $("#pp").innerHTML = ICON_PAUSE; $("#pp").classList.remove("on"); }
  S.delay = Math.max(S.lag, +lsGet("gb_delay", 4)); S.lineY = S.delay - S.lag; S.follow = true;
  updateNav(); restartPlayback();
}
function updateNav() {
  $("#followBtn").style.display = S.follow ? "none" : "inline-flex";
  $("#liveBtn").style.display = S.paused || S.delay > Math.max(8, +lsGet("gb_delay", 4) + 2) ? "inline-flex" : "none";
}
cv.addEventListener("mousedown", e => {
  const [x, y] = canvasPos(e);
  if (Math.abs(y - playLineY()) < 7) { dragging = true; dragTop = topTime(); e.preventDefault(); return; }
  downAt = [x, y];
});
window.addEventListener("mouseup", e => {
  if (dragging) {
    dragging = false;
    if (S.delay < 15) lsSet("gb_delay", S.delay);          // remember small delays as the live preference
    restartPlayback(); return;
  }
  if (!downAt) return;
  const [x, y] = canvasPos(e); const moved = Math.hypot(x - downAt[0], y - downAt[1]); downAt = null;
  if (moved > 4) return;
  const d = boxAt(x, y);
  if (d) { togglePlay(detKey(d), d); return; }
  const top = topTime(); if (top === null) return;          // empty spot: play from here
  setPlayhead(top - y / S.pps, y / S.pps); restartPlayback();
});
cv.addEventListener("mousemove", e => {
  const [x, y, r] = canvasPos(e); const top = topTime(); if (top === null) return;
  if (dragging) { setPlayhead(dragTop - y / S.pps, y / S.pps); return; }
  cv.style.cursor = Math.abs(y - playLineY()) < 7 ? "ns-resize" : boxAt(x, y) ? "pointer" : "crosshair";
  const f = S.fmin + x / r.width * (S.fmax - S.fmin), tip = $("#tip");
  tip.style.display = "block"; tip.style.left = Math.min(x + 14, r.width - 170) + "px"; tip.style.top = (y + 14) + "px";
  tip.textContent = `${fmtF(f)} · ${clockTime(top - y / S.pps)}`;
});
cv.addEventListener("mouseleave", () => { $("#tip").style.display = "none"; });
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const top = topTime(); if (top === null) return;
  if (e.ctrlKey || e.metaKey) {                            // zoom the time axis around the cursor
    const [, y] = canvasPos(e); const tAt = top - y / S.pps;
    S.pps = Math.max(3, Math.min(160, S.pps * Math.exp(-e.deltaY * 0.0025))); lsSet("gb_pps", S.pps);
    if (!S.follow) S.viewTop = tAt + y / S.pps;
    else S.lineY = Math.max(0, Math.min(S.delay - S.lag, S.lineY));
    return;
  }
  if (S.follow) { S.follow = false; S.viewTop = top; }       // browse without moving the playhead
  S.viewTop = Math.min(liveTop(), S.viewTop - e.deltaY / S.pps);
  updateNav();
}, { passive: false });
$("#followBtn").addEventListener("click", () => { S.follow = true; updateNav(); });
$("#pp").addEventListener("click", () => setPaused(!S.paused));
window.addEventListener("keydown", e => {
  if (e.code !== "Space" || /input|select|textarea|button/i.test(e.target.tagName)) return;
  e.preventDefault(); setPaused(!S.paused);
});
$("#liveBtn").addEventListener("click", returnToLive);
function jumpTo(t) {
  const H = cv.clientHeight; S.follow = false; S.viewTop = Math.min(liveTop(), t + H / S.pps * 0.4);
  updateNav();
}

// ---------------- audio ----------------
let AC = null, master = null;
function ensureAudio() {
  if (!AC) {
    AC = new (window.AudioContext || window.webkitAudioContext)(); master = AC.createGain(); master.connect(AC.destination);
    master.gain.value = +$("#vol").value;
    SCOPE.an = AC.createAnalyser(); SCOPE.an.fftSize = 2048; SCOPE.an.smoothingTimeConstant = 0; master.connect(SCOPE.an);
  }
  if (AC.state !== "running") AC.resume().then(restartPlayback);   // schedule once audio is actually running
}
// Audio is always on. Browsers only allow it to start from a user action, so the first click or
// key press anywhere starts it. To stop listening, pause playback (then only clicked clips play).
function startAudioOnce() { ensureAudio(); if (AC && AC.state === "running") { window.removeEventListener("pointerdown", startAudioOnce, true); window.removeEventListener("keydown", startAudioOnce, true); } }
window.addEventListener("pointerdown", startAudioOnce, true);
window.addEventListener("keydown", startAudioOnce, true);
$("#vol").addEventListener("input", e => { fillRange(e.target); if (master) master.gain.value = +e.target.value; });

// ---------------- audio scope: a lane per voice + log-frequency spectrogram of the mix ----------------
const SCOPE = { an: null, img: null, lo: -100, hi: -40, cols: [], repaint: false };
const scv = $("#scope"), sctx = scv.getContext("2d");
const confOf = d => Math.max(d.box.conf_max ?? 0, d.box.conf ?? 0);
const audible = d => confOf(d) >= S.minConf;
// Each voice has its own gain node; it follows the Min confidence filter while it plays, so a
// detection below the filter stays silent and one that rises above it partway fades in.
function updateVoiceGains() {
  if (!AC) return;
  for (const d of S.dets.values()) {
    if (!d.vnode) continue;
    const target = audible(d) ? 1 : 0;
    if (d.gainTarget !== target) { d.gainTarget = target; d.vnode.gain.setTargetAtTime(target, AC.currentTime, 0.03); }
  }
}
function playingVoices() {
  const now = nowS(), out = [];
  for (const d of S.dets.values()) if (audible(d) && d.playing.some(([a, z]) => now >= a && now <= z)) out.push(d);
  return out.sort((a, b) => b.box.conf - a.box.conf).slice(0, 4);
}
const F_LO = 80, F_HI = 4000;
function scopeCol(img, q, H) {                              // one 2-px column of the audio spectrogram
  const cm = makeColLUT();
  for (let y = 0; y < H; y++) {
    const v = q ? q[y] * 4 : -1;
    for (let x = 0; x < 2; x++) { const o = (y * 2 + x) * 4;
      if (v < 0) { img.data[o] = img.data[o + 1] = img.data[o + 2] = 0; } else { img.data[o] = cm[v]; img.data[o + 1] = cm[v + 1]; img.data[o + 2] = cm[v + 2]; }
      img.data[o + 3] = 255; }
  }
}
function drawScope() {
  const dpr = devicePixelRatio, W = scv.clientWidth * dpr, H = scv.clientHeight * dpr;
  if (!W || !H) return;
  if (scv.width !== W || scv.height !== H) { scv.width = W; scv.height = H; SCOPE.img = null; }
  const laneW = Math.round(W * 0.4), specX = laneW + 8 * dpr, specW = W - specX;
  const live = AC && AC.state === "running" && SCOPE.an;
  const voices = live ? playingVoices() : [];
  // lanes: one waveform per voice that is playing right now
  sctx.fillStyle = TH.bg; sctx.fillRect(0, 0, laneW, H);
  sctx.font = `${10.5 * dpr}px ${TH.mono}`;
  if (!voices.length) { sctx.strokeStyle = TH.grid; sctx.beginPath(); sctx.moveTo(0, H / 2); sctx.lineTo(laneW, H / 2); sctx.stroke(); }
  const lh = H / Math.max(1, voices.length), buf = new Float32Array(1024);
  voices.forEach((d, i) => {
    const y0 = i * lh, mid = y0 + lh / 2;
    if (i) { sctx.strokeStyle = TH.grid; sctx.beginPath(); sctx.moveTo(0, y0); sctx.lineTo(laneW, y0); sctx.stroke(); }
    if (d.an) {
      d.an.getFloatTimeDomainData(buf);
      sctx.save(); sctx.globalAlpha = 0.45 + 0.55 * clamp01(d.box.conf); sctx.strokeStyle = TH.mark; sctx.lineWidth = 1.1 * dpr; sctx.beginPath();
      for (let k = 0; k < buf.length; k++) { const x = k / buf.length * laneW, y = mid - buf[k] * lh * 0.42; k ? sctx.lineTo(x, y) : sctx.moveTo(x, y); }
      sctx.stroke(); sctx.restore();
    }
    sctx.fillStyle = TH.text; sctx.fillText(`${fmtFshort((d.box.fmin + d.box.fmax) / 2)}  ${Math.round(d.box.conf * 100)}%`, 6 * dpr, y0 + 12 * dpr);
  });
  // spectrogram of the mix: log frequency axis (80 Hz-4 kHz), dB, automatic colour scaling
  if (!SCOPE.img || SCOPE.img.width !== Math.max(1, Math.round(specW)) || SCOPE.img.height !== H) {
    SCOPE.img = document.createElement("canvas"); SCOPE.img.width = Math.max(1, Math.round(specW)); SCOPE.img.height = H;
    const c = SCOPE.img.getContext("2d"); c.fillStyle = TH.bg; c.fillRect(0, 0, SCOPE.img.width, H);
    SCOPE.repaint = true;
  }
  const ic = SCOPE.img.getContext("2d");
  ic.drawImage(SCOPE.img, -2, 0);                          // scroll left 2 px per frame
  const col = ic.createImageData(2, H);
  if (live) {
    const db = new Float32Array(SCOPE.an.frequencyBinCount); SCOPE.an.getFloatFrequencyData(db);
    const hz = AC.sampleRate / SCOPE.an.fftSize, colVals = new Float32Array(H);
    for (let y = 0; y < H; y++) {
      const f = F_LO * Math.pow(F_HI / F_LO, 1 - y / H);
      colVals[y] = db[Math.min(db.length - 1, Math.round(f / hz))];
    }
    const sorted = Array.from(colVals).filter(Number.isFinite).sort((a, b) => a - b);
    if (sorted.length && sorted[sorted.length - 1] > -140) {   // adapt the colour range to the signal, slowly
      const p20 = sorted[Math.floor(sorted.length * 0.2)], p99 = sorted[Math.floor(sorted.length * 0.99)];
      SCOPE.lo += 0.03 * (p20 - SCOPE.lo);
      SCOPE.hi += (p99 > SCOPE.hi ? 0.2 : 0.02) * (p99 - SCOPE.hi);
    }
    const span = Math.max(12, SCOPE.hi - SCOPE.lo), q = new Uint8Array(H);
    for (let y = 0; y < H; y++) q[y] = Math.max(0, Math.min(255, Math.round((colVals[y] - SCOPE.lo) / span * 255)));
    scopeCol(col, q, H); SCOPE.cols.push(q);
  } else { scopeCol(col, null, H); SCOPE.cols.push(null); }
  const keep = Math.ceil(SCOPE.img.width / 2) + 1; if (SCOPE.cols.length > keep) SCOPE.cols.splice(0, SCOPE.cols.length - keep);
  if (SCOPE.repaint) {                                     // palette changed or resized: recolour all of it
    SCOPE.repaint = false; const one = ic.createImageData(2, H);
    SCOPE.cols.forEach((q, i) => { scopeCol(one, q, H); ic.putImageData(one, SCOPE.img.width - 2 * (SCOPE.cols.length - i), 0); });
  } else ic.putImageData(col, SCOPE.img.width - 2, 0);
  sctx.drawImage(SCOPE.img, specX, 0);
  sctx.fillStyle = TH.text;
  for (const [f, lab] of [[100, "100"], [300, "300"], [1000, "1k"], [3000, "3k"]]) {
    const y = H * (1 - Math.log(f / F_LO) / Math.log(F_HI / F_LO));
    sctx.fillText(lab, specX + 4 * dpr, y + 4 * dpr);
  }
  // caption: only when it says something
  const d = voices[0], cap = $("#scopeCap");
  const text = !AC || AC.state !== "running" ? (FIELD.on ? "Tap anywhere to start audio." : "Click anywhere to start audio.")
    : d && S.tx.get(d.box.id) && S.tx.get(d.box.id).text ? `“${S.tx.get(d.box.id).text}”` : S.paused ? "Paused" : "";
  if (cap.textContent !== text) cap.textContent = text;
}
let COLLUT = null, COLLUT_KEY = "";
function makeColLUT() {                                    // the colour map at full range (the scope scales itself)
  if (COLLUT_KEY === S.cmap) return COLLUT;
  const keep = [S.smin, S.smax]; S.smin = 0; S.smax = 255; COLLUT = makeLUT(); [S.smin, S.smax] = keep; COLLUT_KEY = S.cmap;
  return COLLUT;
}

// Playback scheduler: every detection plays when the playhead reaches it. Audio comes from the
// chunks received live (with their stream times) or, for older detections, from their WAV files.
// Only what the playhead reaches within the next 30 s is scheduled; moving the playhead restarts it.
const PB = { gen: 0, sources: [], gain: null };
function streamOut() { if (!PB.gain) { PB.gain = AC.createGain(); PB.gain.connect(master); } return PB.gain; }
function pcmBuffer(pcm, rate) {
  const buf = AC.createBuffer(1, pcm.length, rate), ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  return buf;
}
async function decodeClips(urls) {
  const parts = [];
  for (const u of urls) { try { const r = await fetch(u); if (r.ok) parts.push(await AC.decodeAudioData(await r.arrayBuffer())); } catch (e) {} }
  if (!parts.length) return null;
  const n = parts.reduce((a, p) => a + p.length, 0), buf = AC.createBuffer(1, n, parts[0].sampleRate);
  let o = 0; for (const p of parts) { buf.getChannelData(0).set(p.getChannelData(0), o); o += p.length; }
  return buf;
}
function playAt(d, buf, tStart, rate) {                     // buf: AudioBuffer or Int16 PCM starting at stream time tStart
  const dur = buf.duration ?? buf.length / rate, now = nowS(), when = tStart + S.offset + S.delay;
  const skip = Math.max(0, now - when); if (skip >= dur) return;
  d.playing.push([Math.max(when, now), when + dur]); if (d.playing.length > 300) d.playing.shift();
  if (!AC || AC.state !== "running") return;
  const src = AC.createBufferSource(); src.buffer = buf.duration !== undefined ? buf : pcmBuffer(buf, rate);
  if (!d.vnode) {                                          // one analyser per voice, so each can be drawn on its own
    d.vnode = AC.createGain(); d.an = AC.createAnalyser(); d.an.fftSize = 1024;
    d.gainTarget = audible(d) ? 1 : 0; d.vnode.gain.value = d.gainTarget;
    d.vnode.connect(d.an); d.an.connect(streamOut());
  }
  src.connect(d.vnode); src.start(AC.currentTime + Math.max(0, when - now), skip);
  PB.sources.push(src); src.onended = () => { const i = PB.sources.indexOf(src); if (i >= 0) PB.sources.splice(i, 1); };
}
function restartPlayback() {
  PB.gen++;
  for (const s of PB.sources) { try { s.stop(); } catch (e) {} }
  PB.sources = [];
  for (const d of S.dets.values()) { d.playing = []; d.sched = new Set(); d.wavSched = false; }
  schedulePlayback();
}
function schedulePlayback() {
  if (S.offset === null || S.paused) return;
  const ph = playheadTime(), horizon = ph + 30;
  for (const d of S.dets.values()) {
    if (!d.sched) d.sched = new Set();
    const b = d.box;
    if (b.t1 < ph - 0.5 || b.t0 > horizon) continue;
    const a0 = b.audio_t0 ?? b.t0;                         // audio starts before the box (pre-roll)
    const chunks = d.chunks || [], whole = chunks.length && chunks[0].t <= a0 + 0.1;
    const wavs = b.clips && b.clips.length;
    if (whole || (chunks.length && !wavs)) {
      for (const c of chunks) {
        const key = c.t.toFixed(3);
        if (d.sched.has(key) || c.t + c.pcm.length / c.rate < ph || c.t > horizon) continue;
        d.sched.add(key); playAt(d, c.pcm, c.t, c.rate);
      }
    } else if (wavs && !d.wavSched) {
      d.wavSched = true; const gen = PB.gen;
      if (!AC) {                                           // no audio yet: still light the box as the playhead passes
        const when = a0 + S.offset + S.delay; d.playing.push([when, when + (b.t1 - a0)]); continue;
      }
      (d.wavBuf ? Promise.resolve(d.wavBuf) : (AC ? decodeClips(b.clips) : Promise.resolve(null))).then(buf => {
        if (buf) d.wavBuf = buf;
        if (gen === PB.gen && buf) playAt(d, buf, a0);
      });
    }
  }
}
setInterval(schedulePlayback, 2000);

function onAudio(m) {                                      // live chunk: keep it, and schedule if the playhead will reach it
  const d = S.dets.get(m.id); if (!d || S.offset === null) return;
  const pcm = new Int16Array(b64bytes(m.pcm).buffer);
  d.pcm.push(pcm); if (d.pcm.length > 120) d.pcm.shift();
  d.chunks = d.chunks || []; d.chunks.push({ t: m.t, pcm, rate: m.rate }); if (d.chunks.length > 150) d.chunks.shift();
  if (!d.sched) d.sched = new Set();
  const key = m.t.toFixed(3);
  if (!S.paused && !d.sched.has(key) && m.t + pcm.length / m.rate >= playheadTime()) { d.sched.add(key); playAt(d, pcm, m.t, m.rate); }
}

// click-to-play: a detection (live PCM or its recording) or a saved recording
const detKey = d => d.box.clips && d.box.clips.length ? "clip:" + d.box.clips[0] : "det:" + d.box.id;
const recKey = r => "clip:" + r.clip;
async function togglePlay(key, det, rec) {
  ensureAudio();
  if (S.playing) { const old = S.playing.key; try { S.playing.src.stop(); } catch (e) {} S.playing = null; setPlayingUI(old, false); if (PB.gain) PB.gain.gain.value = 1; if (old === key) return; }
  let buf = null;
  const urls = rec ? [rec.clip] : (det && det.box.clips) || [];
  if (urls.length) buf = await decodeClips(urls);
  if (!buf && det && det.pcm.length) {
    const n = det.pcm.reduce((a, p) => a + p.length, 0); buf = AC.createBuffer(1, n, 16000);
    const ch = buf.getChannelData(0); let o = 0; for (const p of det.pcm) { for (let i = 0; i < p.length; i++) ch[o + i] = p[i] / 32768; o += p.length; }
  }
  if (!buf) return;
  const src = AC.createBufferSource(); src.buffer = buf; src.connect(master); src.start();
  if (!det && rec) det = [...S.dets.values()].find(d => (d.box.clips || []).includes(rec.clip));
  S.playing = { key, src, start: AC.currentTime, dur: buf.duration, det };
  if (det) det.playing.push([nowS(), nowS() + buf.duration]);
  src.onended = () => { if (S.playing && S.playing.src === src) { S.playing = null; setPlayingUI(key, false); if (PB.gain) PB.gain.gain.value = 1; } };
  if (PB.gain) PB.gain.gain.value = 0;                     // a clicked clip has the floor
  setPlayingUI(key, true);
}
function updateRowProgress() {
  if (!S.playing || !AC) return;
  const p = Math.min(100, (AC.currentTime - S.playing.start) / S.playing.dur * 100);
  const row = document.querySelector(`.det[data-key="${CSS.escape(S.playing.key)}"]`);
  if (row) row.style.setProperty("--p", p + "%");
}

// ---------------- detections list ----------------
// Live detections render on their own; saved ones load in pages of PAGE as you scroll (server-side
// search), at most MAX_ROWS kept in the page. Rows are updated in place, never re-rendered wholesale.
const ICON_PLAY = '<svg viewBox="0 0 8 8"><path d="M1.5 0.8 7 4 1.5 7.2z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 8 8"><path d="M1.2 1h2v6h-2zM4.8 1h2v6h-2z"/></svg>';
const PAGE = 100, MAX_ROWS = 1000;
const L = { q: "", total: 0, loading: false, done: false, gen: 0 };
const clamp01 = v => Math.max(0, Math.min(1, v));
function confStyle(row, conf) {
  const k = clamp01(conf);
  row.style.setProperty("--a", (0.2 + 0.8 * k).toFixed(3));
  row.style.setProperty("--ca", (3 + 32 * k * k).toFixed(1) + "%");   // background tint: faint at low, strong at 100%
  row.style.setProperty("--cb", (1 + 12 * k * k).toFixed(1) + "%");
  row.classList.toggle("hi", k >= 0.9);
}
const ICON_SAVE = '<svg viewBox="0 0 10 10"><path d="M1 1h6.5L9 2.5V9H1zM2.5 1v3h4.5V1M2.5 9V6h5v3" fill="none" stroke="currentColor" stroke-width="1"/></svg>';
const ICON_SAVED = '<svg viewBox="0 0 10 10"><path d="M1.5 5.2 4 7.6 8.5 2.4" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>';
function rowFor({ key, f, t, conf, tx, pending, live, onOpen, onPlay, saved, onSave }) {
  const playing = S.playing && S.playing.key === key;
  let text = "", cls = "x none";
  if (tx) { if (tx.text && tx.no_speech_prob <= 0.6) { text = `“${tx.text}”`; cls = "x"; } else if (tx.text) { text = `“${tx.text}”`; cls = "x doubt"; } else text = "No words"; }
  else if (pending) text = "Transcribing…";
  const btn = el("button", { class: "play" + (playing ? " on" : ""), title: playing ? "Pause" : "Play", onclick: e => { e.stopPropagation(); onPlay(); } });
  btn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
  const row = el("div", { class: "det", "data-key": key, title: (live ? "Live — " : "") + "click for details", onclick: onOpen },
    btn, el("span", { class: "f" }, fmtFshort(f)), el("span", { class: "t" }, t), el("span", { class: cls, title: text }, text),
    el("span", { class: "c" }, (conf * 100).toFixed(0)), saveCell(saved, onSave),
    live ? el("i", { class: "live", title: "Live" }) : null);
  confStyle(row, conf);
  return row;
}
function saveCell(saved, onSave) {
  if (saved === undefined) return el("span");
  const b = el("button", { class: "sv" + (saved ? " done" : ""), title: saved ? "Saved to disk" : "Keep: save to disk",
    onclick: e => { e.stopPropagation(); if (!saved && onSave) { b.disabled = true; onSave(); } } });
  b.innerHTML = saved ? ICON_SAVED : ICON_SAVE;
  return b;
}
async function saveDetection(r) {
  const res = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: r.key }) });
  if (!res.ok) alert("This detection's audio is no longer available (dropped from memory and not in the IQ recording).");
}
function onSaved(m) {
  const i = S.recs.findIndex(r => r.key === m.old_key); if (i < 0) return;
  const oldKey = recKey(S.recs[i]); S.recs[i] = { ...S.recs[i], ...m.meta };
  const row = $(`#savedList .det[data-key="${CSS.escape(oldKey)}"]`); if (row) row.replaceWith(recRow(S.recs[i]));
  for (const d of S.dets.values()) if (d.box.clips) d.box.clips = d.box.clips.map(u => u === "/api/clip?key=" + m.old_key ? m.meta.clip : u);
}
function recRow(r) {
  const sameSession = r.epoch && Math.abs(r.epoch - S.epoch) < 1e-3;
  const minConf = S.cfg ? S.cfg.whisper.min_confidence : 1;
  const recent = Date.now() - new Date(r.utc_start).getTime() < 180e3;
  return rowFor({ key: recKey(r), f: r.freq_hz.mean, t: hms(new Date(r.utc_start)), conf: r.confidence.max, tx: r.transcript,
    pending: recent && !r.transcript && r.confidence.max >= minConf && S.cfg && S.cfg.whisper.enabled,
    onOpen: () => openDetail({ key: r.key }), onPlay: () => togglePlay(recKey(r), null, r),
    saved: r.saved !== false, onSave: () => saveDetection(r) });
}
let liveTimer = null;
function renderLive() {
  if (liveTimer) return;
  liveTimer = setTimeout(() => {
    liveTimer = null;
    const ds = [...S.dets.values()].filter(d => !d.ended && d.box.detected !== false && audible(d)).sort((a, b) => b.box.t0 - a.box.t0);
    $("#liveList").replaceChildren(...ds.map(d => {
      const b = d.box;
      return rowFor({ key: detKey(d), f: (b.fmin + b.fmax) / 2, t: clockTime(b.t0), conf: b.conf, tx: S.tx.get(b.id), live: true,
        onOpen: () => openDetail({ live: d }), onPlay: () => togglePlay(detKey(d), d) });
    }));
    updateCount(ds.length);
  }, 200);
}
let liveCount = 0;
function updateCount(n) {
  if (n !== undefined) liveCount = n;
  $("#detCount").textContent = `${liveCount ? liveCount + " live, " : ""}${L.total} ${L.q ? "found" : "saved"}`;
}
async function loadPage(reset) {
  if (reset) { L.gen++; S.recs = []; L.done = false; L.loading = false; $("#savedList").replaceChildren(); }
  if (L.loading || L.done) return;
  const gen = L.gen; L.loading = true;
  try {
    const r = await (await fetch(`/api/detections?offset=${S.recs.length}&limit=${PAGE}&q=${encodeURIComponent(L.q)}&min_conf=${S.minConf}`)).json();
    if (gen !== L.gen) return;
    L.total = r.total; S.recs.push(...r.items);
    $("#savedList").append(...r.items.map(recRow));
    L.done = S.recs.length >= r.total || S.recs.length >= MAX_ROWS;
  } catch (e) {} finally { if (gen === L.gen) L.loading = false; }
  $("#listMore").textContent = S.recs.length >= MAX_ROWS && S.recs.length < L.total
    ? `Showing the newest ${MAX_ROWS}. Search to find older ones.` : "";
  updateCount();
}
$("#dets").addEventListener("scroll", e => { const t = e.target; if (t.scrollTop + t.clientHeight > t.scrollHeight - 400) loadPage(false); });
let searchTimer = null;
$("#search").addEventListener("input", e => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { L.q = e.target.value.trim(); loadPage(true); }, 250); });
function matchesSearch(m) {
  if (m.confidence.max < S.minConf) return false;
  if (!L.q) return true;
  const f = m.freq_hz.mean, hay = [(m.transcript || {}).text, (m.chain_transcript || {}).text, (f / 1e6).toFixed(4), (f / 1e3).toFixed(1)].join(" ").toLowerCase();
  return hay.includes(L.q.toLowerCase());
}
function onRecording(meta) {
  const d = S.dets.get(meta.id); if (d && !(d.box.clips || []).includes(meta.clip)) (d.box.clips = d.box.clips || []).push(meta.clip);
  if (matchesSearch(meta)) {
    S.recs.unshift(meta); L.total++; $("#savedList").prepend(recRow(meta));
    if (S.recs.length > MAX_ROWS) { S.recs.pop(); $("#savedList").lastElementChild.remove(); }
  }
  updateCount(); renderLive();
}
function replaceRecRow(r) {
  const old = $(`#savedList .det[data-key="${CSS.escape(recKey(r))}"]`);
  if (old) old.replaceWith(recRow(r));
}
function onTranscript(m) {
  const res = { text: m.text, no_speech_prob: m.no_speech_prob, language: m.language, model: m.model };
  const keys = m.keys || [];
  if (m.kind === "track" && res.text) { (S.txLog = S.txLog || []).push({ text: res.text, doubt: res.no_speech_prob > 0.6, at: Date.now() }); if (S.txLog.length > 20) S.txLog.shift(); }
  if (m.kind === "track") { S.tx.set(m.id, res); for (const r of S.recs) if (keys.includes(r.key)) { r.transcript = res; replaceRecRow(r); } }
  else { S.chainTx.set(m.chain, res); for (const r of S.recs) if (keys.includes(r.key)) { r.chain_transcript = res; replaceRecRow(r); } }
  renderLive();
}
function setPlayingUI(key, on) {
  document.querySelectorAll(`.det[data-key="${CSS.escape(key)}"]`).forEach(row => {
    const b = row.querySelector(".play"); b.classList.toggle("on", on); b.innerHTML = on ? ICON_PAUSE : ICON_PLAY; b.title = on ? "Pause" : "Play";
    if (!on) row.style.setProperty("--p", "0%");
  });
}

// ---------------- config helpers ----------------
let pending = {}, pendTimer = null;
function merge(a, b) { for (const [k, v] of Object.entries(b)) { if (v && typeof v === "object" && !Array.isArray(v)) { a[k] = a[k] || {}; merge(a[k], v); } else a[k] = v; } return a; }
function patch(obj, wait = 300) {
  merge(pending, obj); merge(S.cfg, JSON.parse(JSON.stringify(obj)));
  clearTimeout(pendTimer);
  pendTimer = setTimeout(async () => { const p = pending; pending = {}; await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }); }, wait);
}
const setPath = (path, v) => { const o = {}; let cur = o; path.forEach((k, i) => { if (i === path.length - 1) cur[k] = v; else cur = cur[k] = {}; }); return o; };
const getPath = (obj, path) => path.reduce((o, k) => (o || {})[k], obj);
function slider(label, path, min, max, step, fmt = v => v) {
  const v = getPath(S.cfg, path);
  const out = el("output", {}, fmt(+v));
  const inp = el("input", { type: "range", min, max, step, value: v }); fillRange(inp);
  inp.addEventListener("input", () => { fillRange(inp); out.textContent = fmt(+inp.value); patch(setPath(path, +inp.value)); });
  return el("div", { class: "row" }, el("label", { title: label }, label), inp, out);
}
function toggle(checked, onchange) {
  const i = el("input", { type: "checkbox" }); i.checked = checked; i.addEventListener("change", () => onchange(i.checked));
  return el("label", { class: "switch" }, i, el("span"));
}
function selectRow(label, options, value, onchange) {
  const s = el("select", { class: "sel" }, ...options.map(o => Array.isArray(o) ? el("option", { value: o[0] }, o[1]) : el("option", { value: o }, o)));
  s.value = value; s.addEventListener("change", () => onchange(s.value));
  return el("div", { class: "row wide" }, el("label", {}, label), s);
}

// ---------------- top bar (receiver) ----------------
const RATES = { hackrf: [2e6, 4e6, 8e6, 10e6, 16e6, 20e6], rtlsdr: [1.024e6, 1.4e6, 1.8e6, 2.048e6, 2.4e6], file: [] };
function hackrfSplit(g) { const lna = Math.min(40, 8 * Math.floor(g / 2 / 8)); return [lna, Math.min(62, Math.max(0, 2 * Math.round((g - lna) / 2)))]; }
function buildTop() {
  const sdr = S.cfg.sdr, src = sdr.source;
  if (document.activeElement !== $("#freq")) $("#freq").value = (sdr.center_hz / 1e6).toFixed(4);
  const rates = RATES[src].length ? [...RATES[src]] : [sdr.sample_rate];
  if (!rates.includes(sdr.sample_rate)) rates.push(sdr.sample_rate);
  $("#rate").replaceChildren(...rates.map(r => el("option", { value: r }, `${+(r / 1e6).toFixed(3)} MHz span`)));
  const lo = sdr.fmin_hz ?? sdr.center_hz - sdr.sample_rate / 2, hi = sdr.fmax_hz ?? sdr.center_hz + sdr.sample_rate / 2;
  $("#range").textContent = `${(Math.max(0, lo) / 1e6).toFixed(3)} – ${(hi / 1e6).toFixed(3)} MHz`;
  $("#rate").value = sdr.sample_rate; $("#rate").disabled = src === "file";
  const g = $("#gain");
  g.max = src === "rtlsdr" ? 49.6 : 102; g.step = src === "rtlsdr" ? 0.1 : 2; g.value = sdr.gain_db ?? 36; g.disabled = src === "file"; fillRange(g);
  gainLabel();
  $("#amp").classList.toggle("on", !!sdr.amp); $("#amp").disabled = src !== "hackrf";
  for (const id of ["freq", "left", "right", "fleft", "fright"]) $("#" + id).disabled = src === "file";
}
function gainLabel() {
  const g = +$("#gain").value, src = S.cfg.sdr.source;
  $("#gainVal").textContent = src === "rtlsdr" ? (g === 0 ? "auto" : g.toFixed(1) + " dB") : src === "file" ? "—" : g + " dB";
  if (src === "hackrf") { const [l, v] = hackrfSplit(g); $("#gain").title = `HackRF LNA ${l} + VGA ${v}`; }
}
function shiftWindow(widths) {
  // widths of the visible window (the analysis range if one is set, else the whole capture)
  const sdr = S.cfg.sdr, span = (sdr.fmin_hz != null && sdr.fmax_hz != null) ? sdr.fmax_hz - sdr.fmin_hz : sdr.sample_rate;
  const d = widths * span;
  const p = { sdr: { center_hz: sdr.center_hz + d } };
  if (sdr.fmin_hz != null) p.sdr.fmin_hz = sdr.fmin_hz + d;
  if (sdr.fmax_hz != null) p.sdr.fmax_hz = sdr.fmax_hz + d;
  patch(p, 50); buildTop();
}
$("#left").addEventListener("click", () => shiftWindow(-0.5));
$("#right").addEventListener("click", () => shiftWindow(0.5));
$("#fleft").addEventListener("click", () => shiftWindow(-2));
$("#fright").addEventListener("click", () => shiftWindow(2));
$("#freq").addEventListener("keydown", e => { if (e.key === "Enter") e.target.blur(); });
$("#freq").addEventListener("change", e => {
  const f = parseFloat(e.target.value) * 1e6; if (!isFinite(f)) return;
  const sdr = S.cfg.sdr, d = f - sdr.center_hz, p = { sdr: { center_hz: f } };
  if (sdr.fmin_hz != null) p.sdr.fmin_hz = sdr.fmin_hz + d;
  if (sdr.fmax_hz != null) p.sdr.fmax_hz = sdr.fmax_hz + d;
  patch(p, 50);
});
function setSource(src) {
  const p = { sdr: { source: src } };
  if (src !== "file" && !RATES[src].includes(S.cfg.sdr.sample_rate)) p.sdr.sample_rate = src === "rtlsdr" ? 2.4e6 : 4e6;
  if (src === "rtlsdr" && S.cfg.sdr.center_hz < 24e6) p.sdr.center_hz = 119.5e6;
  patch(p, 50); buildTop(); if ($("#sdr").classList.contains("active")) buildSDR();
}
$("#rate").addEventListener("change", e => { patch({ sdr: { sample_rate: +e.target.value } }, 50); });
$("#gain").addEventListener("input", e => { fillRange(e.target); gainLabel(); });
$("#gain").addEventListener("change", e => patch({ sdr: { gain_db: +e.target.value } }, 50));
$("#amp").addEventListener("click", () => { patch({ sdr: { amp: !S.cfg.sdr.amp } }, 50); buildTop(); });


// ---------------- detector tab ----------------
function buildDetector() {
  const root = $("#detector"), c = S.cfg; root.replaceChildren();
  const isActive = v => Object.entries(v.detect).every(([k, x]) => Math.abs(c.detect[k] - x) < 1e-9)
    && Object.entries(v.checks).every(([id, o]) => Object.entries(o).every(([k, x]) => c.checks[id][k] === x));
  const seg = el("div", { class: "seg" }, ...Object.entries(S.sens).map(([k, v]) => el("button", {
    class: isActive(v) ? "on" : "",
    onclick: async e => {
      e.currentTarget.blur();
      const r = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ detect: v.detect, checks: v.checks }) });
      S.cfg = await r.json(); buildDetector();
    }
  }, v.label)));
  root.append(tuneCard()); tickReady();
  root.append(el("h2", {}, "Sensitivity"), seg);
  root.append(el("h2", {}, "Thresholds & tracking"),
    slider("Candidate threshold", ["detect", "candidate_threshold"], 0, 1, 0.01, v => v.toFixed(2)),
    slider("Capture floor (kept in memory)", ["detect", "threshold"], 0, 1, 0.01, v => v.toFixed(2)),
    slider("Novelty strength", ["detect", "novelty"], 0, 1.5, 0.05, v => v.toFixed(2)),
    slider("Novelty memory", ["detect", "novelty_tau_s"], 2, 120, 1, v => v + " s"),
    slider("Prominence strength", ["detect", "prominence"], 0, 1.5, 0.05, v => v.toFixed(2)),
    slider("Prominence width", ["detect", "prominence_hz"], 5000, 200000, 5000, v => (v / 1e3) + " kHz"),
    slider("Min duration", ["detect", "min_duration_s"], 0.1, 5, 0.05, v => v.toFixed(2) + " s"),
    slider("Candidates per frame", ["detect", "max_candidates"], 1, 30, 1),
    slider("Max tracks", ["detect", "max_tracks"], 1, 40, 1),
    slider("Max slide rate", ["detect", "max_slide_hz_s"], 0, 500000, 5000, v => (v / 1e3) + " kHz/s"),
    slider("Track gap", ["detect", "track_gap_s"], 0.05, 2, 0.05, v => v.toFixed(2) + " s"),
    slider("Hop link window", ["detect", "link_hops_s"], 0, 2, 0.05, v => v.toFixed(2) + " s"),
    slider("Min separation", ["detect", "min_sep_hz"], 1000, 50000, 500, v => (v / 1e3) + " kHz"));
  const w = c.whisper, ws = S.status.whisper || {};
  root.append(el("h2", {}, "Transcription"),
    el("div", { class: "card" + (w.enabled ? "" : " off") },
      el("div", { class: "hd" }, toggle(w.enabled, v => { patch({ whisper: { enabled: v } }); buildDetector(); }), "Speech recognition (local)", el("span", { class: "tag" }, ws.model ? `loaded ${ws.model}` : "loads on first use")),
      selectRow("Model", S.asrModels || [[w.model, w.model]], w.model, v => patch({ whisper: { model: v } })),
      ...(/^(tiny|base|small|medium|large)/.test(w.model) ? [selectRow("Language", ["auto", "en", "es", "fr", "de", "it", "pt", "ja", "zh"], w.language, v => patch({ whisper: { language: v } }))] : []),
      slider("Only above confidence", ["whisper", "min_confidence"], 0, 1, 0.01, v => v.toFixed(2)),
      el("div", { class: "row wide" }, el("label", {}, "Also transcribe whole hop chains"), toggle(w.chains, v => patch({ whisper: { chains: v } }))),
      slider("Workers (share one loaded model)", ["whisper", "workers"], 1, 4, 1),
      slider("Clips per batch", ["whisper", "batch_size"], 1, 32, 1),
      slider("CPU threads", ["whisper", "cpu_threads"], 2, 16, 1),
      slider("Skip clips waiting longer than", ["whisper", "max_age_s"], 10, 600, 10, v => v + " s"),
      el("div", { class: "row wide", style: "align-items:start" }, el("label", { title: "Removed from transcripts wherever they appear, ignoring case and punctuation. One per line." }, "Filtered phrases"),
        (() => { const ta = el("textarea", { class: "field", rows: 6, style: "resize:vertical;min-height:90px;font:inherit;line-height:1.35" });
          ta.value = (w.filter_phrases || []).join("\n");
          ta.addEventListener("change", () => patch({ whisper: { filter_phrases: ta.value.split("\n").map(x => x.trim()).filter(Boolean) } }, 50));
          return ta; })()),
      el("div", { class: "note" }, `Most confident clips are transcribed first. Done ${ws.done ?? 0}, skipped as stale ${ws.stale ?? 0}, dropped (queue full) ${ws.dropped ?? 0}.`),
      ws.error ? el("div", { class: "err" }, ws.error) : null));
  root.append(el("h2", {}, "Checks"),
    el("div", { class: "note", style: "margin-top:-4px" }, "Each check's weight sets how much it counts; 0 switches it off and it isn't computed at all. Calibration tries every check, including the ones at 0. ▲ marks the weight the last calibration chose."),
    el("div", { style: "display:flex;gap:8px;margin:6px 0 2px" }, el("button", { class: "btn", onclick: () => {
      if (!confirm("Put every check's weight back to its default?")) return;
      patch({ checks: Object.fromEntries(S.checks.map(ch => [ch.id, { weight: ch.default.weight }])) }, 50); buildDetector();
    } }, "Reset checks to defaults")));
  for (const g of S.groups) {
    const list = S.checks.filter(x => x.group === g.id); if (!list.length) continue;
    root.append(el("h3", { class: "chkgroup" }, g.label, g.note ? el("span", { class: "t3" }, " · " + g.note) : ""));
    for (const ch of list) root.append(checkRow(ch));
  }
  root.append(el("div", { style: "margin:14px 0 20px" }, el("button", { class: "btn", onclick: async () => { if (confirm("Reset every setting to its default?")) await fetch("/api/reset", { method: "POST" }); } }, "Reset all settings")));
}

// One row per check: name, ⓘ (tap: what it does), weight slider (0 = off), and ⋯ for its parameters.
function checkRow(ch) {
  const cc = S.cfg.checks[ch.id], calW = (TUNE.state.weights || {})[ch.id];
  const fmtW = v => v > 0 ? v.toFixed(2) : "off";
  const out = el("output", {}, fmtW(+cc.weight));
  const inp = el("input", { type: "range", min: 0, max: 3, step: 0.05, value: cc.weight, "aria-label": ch.label + " weight" }); fillRange(inp);
  const desc = el("div", { class: "desc", hidden: true }, ch.desc);
  const params = ch.params.length ? el("div", { class: "chkparams", hidden: true }, ...ch.params.map(p => slider(p.label, ["checks", ch.id, p.key], p.min, p.max, p.step))) : null;
  const tag = ch.family ? el("span", { class: "tag", title: "Alternative modulation hypothesis" }, ch.family === "dsb" ? "AM" : "SSB")
    : ch.id === "profile" && S.status.profile ? el("span", { class: "tag" }, `${S.status.profile.learned_min} min`) : null;
  const row = el("div", { class: "chk" + (+cc.weight > 0 ? "" : " off") },
    el("label", { title: ch.label }, el("span", { class: "nm" }, ch.label), tag),
    el("button", { class: "ib", title: "What it does", "aria-label": "About " + ch.label, onclick: () => { desc.hidden = !desc.hidden; } }, "ⓘ"),
    el("span", { class: "wt" }, inp, calW != null ? el("i", { class: "calmark", title: `Calibration chose ${fmtW(calW)}`, style: `left:calc(5px + (100% - 10px) * ${Math.min(1, calW / 3)})` }) : null),
    out,
    params ? el("button", { class: "ib", title: "Parameters", "aria-label": ch.label + " parameters", onclick: () => { params.hidden = !params.hidden; } }, "⋯") : el("span"));
  inp.addEventListener("input", () => {
    fillRange(inp); const v = +inp.value; out.textContent = fmtW(v); row.classList.toggle("off", v <= 0);
    patch(setPath(["checks", ch.id, "weight"], v));
  });
  const box = el("div", { class: "chkbox" }, row, desc);
  if (params) box.append(params);
  if (ch.id === "vad" && S.status.vad === false) box.append(el("div", { class: "err" }, "Silero VAD not available on the server."));
  return box;
}

// ---------------- calibration (auto-tune) ----------------
const TUNE = { state: { state: "idle" }, get minutes() { return S.cfg ? S.cfg.autotune.minutes : 2; } };
function tuneCard() {
  const st = TUNE.state, running = st.state === "running", at = S.cfg.autotune;
  const mins = el("select", { class: "sel" }, ...[1, 2, 5, 10].map(m => el("option", { value: m }, m + " min")));
  mins.value = at.minutes; mins.addEventListener("change", () => { patch({ autotune: { minutes: +mins.value } }, 50); setTimeout(pollReady, 300); });
  const go = el("button", { id: "tuneGo", class: "btn primary", disabled: running || !(TUNE.ready && TUNE.ready.ready), onclick: async () => {
    await fetch("/api/autotune", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ minutes: at.minutes, per_min: at.per_min }) });
  } }, running ? "Calibrating…" : "Recalibrate now");
  const out = el("div", { class: "note", style: "margin:8px 0 0" });
  if (running) out.textContent = st.message || "";
  else if (st.state === "error") { out.className = "err"; out.textContent = st.message; }
  else if (st.state === "done" && st.result) {
    const r = st.result;
    out.style.color = "var(--text-2)";
    const rows = (r.decisions || []).map(d => el("div", { class: "row", style: "min-height:22px;grid-template-columns:minmax(0,1fr) 70px" },
      el("label", { title: d.label }, d.label),
      el("span", { class: "num", style: `text-align:right;color:${d.on ? "var(--text)" : "var(--text-3)"}` }, d.on ? "× " + d.weight : "off")));
    out.replaceChildren(el("div", {}, `Calibrated on ${r.candidates} candidates, ${r.voice_like} speech-like.`),
      rows.length ? el("details", { style: "margin-top:6px" }, el("summary", {}, "Check settings chosen"), ...rows) : "");
  } else if (st.state === "quiet" || st.state === "reverted" || st.state === "survey") out.textContent = st.message;
  if (!running && (st.changes || []).length) {
    const f = v => v > 0 ? v.toFixed(2) : "off";
    out.append(el("div", { class: "note", style: "margin:6px 0 0" }, "Checks changed: " + st.changes.map(c => `${c.label} ${f(c.before)} → ${f(c.after)}`).join(" · ")));
  }
  const undo = ["done", "quiet", "survey"].includes(st.state) ? el("button", { class: "btn", onclick: async () => { await fetch("/api/autotune/revert", { method: "POST" }); } }, "Undo calibration") : null;
  return el("div", { class: "card", style: "border-top:0;padding-top:0" },
    el("h2", { style: "margin-top:12px" }, "Calibration"),
    el("div", { class: "row wide" }, el("label", { title: "Calibrates after any receiver change, and recalibrates whenever nothing is caught for the time below" }, "Recalibrate automatically"),
      toggle(!!at.auto, v => patch({ autotune: { auto: v }, ...(v ? { iq_ring: { enabled: true } } : {}) }, 50))),
    el("div", { class: "row wide" }, el("label", {}, "Analyse the last"), mins),
    slider("Max detections per min", ["autotune", "per_min"], 1, 30, 1, v => v + " / min"),
    slider("Voice-likeness floor", ["autotune", "floor"], 0, 0.9, 0.05, v => Math.round(v * 100) + "%"),
    readyBlock(),
    el("div", { style: "display:flex;gap:8px;margin-top:8px" }, go, undo),
    out);
}

// ---------------- band survey ----------------
// "New" on the map: a small dialog to configure and start a survey.
function newSurveyDialog() {
  const sdr = S.cfg.sdr, fs = sdr.sample_rate, step = fs * 0.8;
  const num = v => el("input", { class: "field", type: "number", step: "0.1", min: 0, value: v, style: "width:96px" });
  const lo = num(+lsGet("gb_sv_lo", sdr.source === "rtlsdr" ? 24 : 0.5)), hi = num(+lsGet("gb_sv_hi", 30));
  const dwell = el("select", { class: "sel" }, ...[[15, "15 s"], [30, "30 s"], [60, "1 min"], [120, "2 min"]].map(([v, l]) => el("option", { value: v }, l)));
  dwell.value = lsGet("gb_sv_dwell", 30);
  const est = el("div", { class: "note", style: "margin:8px 0 0" }), err = el("div", { class: "err", style: "margin:6px 0 0" });
  const upd = () => {
    const n = Math.max(1, Math.ceil(((+hi.value - +lo.value) * 1e6) / step));
    est.textContent = +hi.value > +lo.value ? `${n} step${n > 1 ? "s" : ""} of ${(step / 1e6).toFixed(2)} MHz (80% of the ${(fs / 1e6).toFixed(1)} MHz span) · about ${fmtMS(n * (+dwell.value + 6))}. Each step is calibrated on its own sample and its detections transcribed; the receiver comes back here afterwards.` : "From must be below To.";
    lsSet("gb_sv_lo", lo.value); lsSet("gb_sv_hi", hi.value); lsSet("gb_sv_dwell", dwell.value);
  };
  [lo, hi, dwell].forEach(x => x.addEventListener("input", upd)); upd();
  const close = () => { wrap.remove(); document.removeEventListener("keydown", key, true); };
  const key = e => { if (e.key === "Escape") { e.stopPropagation(); close(); } else if (e.key === "Enter") { e.preventDefault(); start.click(); } };
  const start = el("button", { class: "btn primary", onclick: async () => {
    start.disabled = true;
    const r = await (await fetch("/api/survey", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lo_hz: +lo.value * 1e6, hi_hz: +hi.value * 1e6, dwell_s: +dwell.value }) })).json();
    if (r.error) { err.textContent = r.error; start.disabled = false; return; }
    close(); SV.state = r; openSurvey();
  } }, "Start survey");
  const box = el("div", { class: "dlg", style: "width:min(460px,100%)" },
    el("div", { class: "dlghead" }, el("strong", {}, "New survey"), el("span", { style: "flex:1" }), el("button", { class: "btn", onclick: close }, "Cancel")),
    el("div", { class: "row wide", style: "margin-top:12px" }, el("label", {}, "From – to (MHz)"), el("div", { style: "display:flex;gap:6px;align-items:center" }, lo, el("span", { class: "t3" }, "–"), hi)),
    el("div", { class: "row wide" }, el("label", {}, "Time per step"), dwell),
    est, err,
    el("div", { style: "display:flex;justify-content:flex-end;margin-top:12px" }, start));
  const wrap = el("div", { id: "newSurvey", class: "modalwrap", onclick: e => { if (e.target === wrap) close(); } }, box);
  document.body.append(wrap); document.addEventListener("keydown", key, true); lo.focus();
}
async function loadSurveyList() {
  try { const r = await (await fetch("/api/survey?list=1")).json(); SV.saved = r.saved; SV.state = r.state; } catch (e) {}
}
// Top-bar Map button: the running/last survey, else the newest saved one.
async function openMap() {
  if (SV.open) { closeSurvey(); return; }
  await loadSurveyList();
  if (SV.state.state === "running" || (SV.meta && !SV.file)) return openSurvey();
  if (SV.saved.length) return openSurvey(SV.saved[0]);
  SV.open = true; SV.meta = null; drawSurvey();
}
async function openSurvey(file) {
  const r = await (await fetch("/api/survey" + (file ? "?file=" + encodeURIComponent(file) : ""))).json();
  if (!r.result) { SV.open = true; SV.meta = null; drawSurvey(); return; }
  const same = SV.meta && SV.file === (file || null) && SV.meta.started === r.result.started;
  const grew = same && SV.meta.steps.length !== r.result.steps.length;
  SV.meta = r.result; SV.file = file || null; SV.open = true;
  if (!same) { SV.imgs = {}; SV.v0 = r.result.lo; SV.v1 = r.result.hi; SV.sel = null; SV.hl = null; }
  else if (grew) SV.imgs = {};                             // a running survey finished a step: refetch images
  drawSurvey();
}
function onSurvey(m) {
  const was = SV.state.state; SV.state = m;
  if (SV.open && !SV.file && (m.steps_done || (m.state !== "running" && was === "running"))) openSurvey();
  if (m.state !== "running" && was === "running") loadSurveyList().then(() => { if ($("#detector").classList.contains("active")) buildDetector(); });
  const t = $("#svStatus"); if (t) t.textContent = t.title = m.state === "running" ? m.message || "" : m.state === "error" ? m.message || "" : "";
  const nb = $("#svNew"); if (nb) nb.textContent = m.state === "running" ? "Stop survey" : "New";
  if (!$("#detector").contains(document.activeElement) && m.state !== "running") buildDetector();
}
function closeSurvey() { SV.open = false; const p = $("#survey"); if (p) p.remove(); $("#mapBtn").classList.remove("on"); buildViewList(); }
// Tune here: the server moves the receiver and puts the calibration of the survey step under f live.
async function surveyTune(f) {
  const file = SV.file || (SV.meta && SV.meta.file);
  closeSurvey();
  try {
    const r = await (await fetch("/api/survey/tune", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ f, file: SV.file ? file : null }) })).json();
    if (r.error) toast(r.error); else if (r.message) toast(r.message);
  } catch (e) {
    const step = SV.meta ? SV.meta.step_hz : S.cfg.sdr.sample_rate * 0.8;
    patch({ sdr: { center_hz: Math.round(f), fmin_hz: Math.round(f - step / 2), fmax_hz: Math.round(f + step / 2) } }, 50);
  }
}
// The survey follows the toolbar: View picks the map, the colour map applies, Min confidence filters the boxes.
function surveyView() { return SV.meta && SV.meta.views.includes(S.view) ? S.view : "power"; }
async function surveyImage(view) {
  const key = view + "|" + S.cmap;
  if (SV.imgs[key]) return SV.imgs[key];
  const r = await fetch(`/api/survey/image?view=${encodeURIComponent(view)}` + (SV.file ? "&file=" + encodeURIComponent(SV.file) : ""));
  if (!r.ok) return null;
  const bytes = new Uint8Array(await r.arrayBuffer()), rows = +r.headers.get("X-Rows"), cols = +r.headers.get("X-Cols");
  // automatic brightness, like the waterfall: noise floor (20th percentile) .. peaks (99.8th)
  const smp = []; for (let i = 0; i < bytes.length; i += 37) smp.push(bytes[i]); smp.sort((a, b) => a - b);
  const lo = smp[Math.floor(smp.length * 0.2)] || 0, hi = Math.max(lo + 8, smp[Math.floor(smp.length * 0.998)] || 255);
  const base = makeColLUT(), lut = new Uint8ClampedArray(256 * 4);
  for (let v = 0; v < 256; v++) { const i = Math.max(0, Math.min(255, Math.round((v - lo) / (hi - lo) * 255))); lut.set(base.subarray(i * 4, i * 4 + 4), v * 4); }
  const parts = [];
  let c0 = 0;
  for (const st of SV.meta.steps) {                         // one canvas per step (keeps canvases small)
    const w = st.cols, c = document.createElement("canvas"); c.width = w; c.height = rows;
    const g = c.getContext("2d"), img = g.createImageData(w, rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < w; x++) {
      const v = bytes[y * cols + c0 + x], k = (y * w + x) * 4;
      img.data[k] = lut[v * 4]; img.data[k + 1] = lut[v * 4 + 1]; img.data[k + 2] = lut[v * 4 + 2]; img.data[k + 3] = 255;
    }
    g.putImageData(img, 0, 0); parts.push({ cv: c, st }); c0 += w;
  }
  return (SV.imgs[key] = parts);
}
const svTracks = () => (SV.meta ? SV.meta.steps.flatMap((st, si) => st.tracks.map(t => ({ ...t, step: si }))) : []);
function drawSurvey() {
  if (!SV.open) return;
  let p = $("#survey");
  if (!p) {
    p = el("div", { id: "survey" }); $("#wfwrap").append(p);
    const head = el("div", { class: "svhead" },
      el("select", { id: "svPick", class: "sel", title: "Recorded maps", onchange: e => openSurvey(e.target.value || undefined) }),
      el("span", { id: "svInfo", class: "t2" }), el("span", { id: "svStatus", class: "t3" }),
      el("span", { style: "flex:1 1 0;min-width:0" }), el("span", { id: "svSel", class: "num t2" }),
      el("button", { id: "svTune", class: "btn primary", style: "display:none", onclick: () => SV.sel != null && surveyTune(SV.sel) }, "Tune here"),
      el("button", { id: "svNew", class: "btn", title: "Survey a new range", onclick: () => SV.state.state === "running" ? fetch("/api/survey/stop", { method: "POST" }) : newSurveyDialog() }, "New"),
      el("button", { class: "btn", title: "Show the whole range (or double-click the map)", onclick: () => { if (SV.meta) { SV.v0 = SV.meta.lo; SV.v1 = SV.meta.hi; renderSurvey(); } } }, "Fit"),
      el("button", { class: "btn", onclick: closeSurvey }, "Close"));
    const map = el("canvas", { id: "svMap", class: "svmap" }), chart = el("canvas", { id: "svChart", class: "svchart" });
    p.append(head, el("div", { class: "svmapwrap" }, map, el("div", { id: "svTip", class: "svtip" })), el("div", { id: "svAxis", class: "svaxis" }), chart,
      el("div", { class: "svsamples" }, el("h3", {}, "Best samples"), el("div", { id: "svSamples" })));
    for (const c of [map, chart]) surveyInteract(c);
  }
  const m = SV.meta, run = SV.state.state === "running";
  $("#mapBtn").classList.add("on"); buildViewList();
  $("#svNew").textContent = run ? "Stop survey" : "New";
  const pick = $("#svPick"), cur = SV.file || "";
  pick.replaceChildren(...(run || (m && !SV.file) ? [el("option", { value: "" }, run ? "Survey in progress" : "Latest survey")] : []),
    ...SV.saved.map(f => el("option", { value: f }, f.replace(/\.json$/, "").replace(/_/g, " "))));
  if (!pick.options.length) pick.append(el("option", { value: "" }, "No surveys yet"));
  pick.value = cur;
  if (!m) { $("#svInfo").textContent = "No surveys yet: click New to start one."; for (const id of ["svMap", "svChart"]) { const c = $("#" + id); c.getContext("2d").clearRect(0, 0, c.width, c.height); } $("#svAxis").replaceChildren(); $("#svSamples").replaceChildren(); return; }
  $("#svInfo").textContent = m ? `${fmtFshort(m.lo)} – ${fmtFshort(m.hi)} · ${m.dwell_s} s per step · ${m.source} · ${(m.started || "").replace("T", " ").slice(0, 16)}`
    + (S.view !== surveyView() ? ` · “${(S.views.find(v => v.id === S.view) || {}).label || S.view}” wasn't recorded; showing power` : "") : "";
  $("#svStatus").textContent = $("#svStatus").title = run ? SV.state.message || "" : "";
  $("#svInfo").title = $("#svInfo").textContent;
  renderSurvey(); renderSamples();
}
async function renderSurvey() {
  const m = SV.meta, map = $("#svMap"); if (!m || !map) return;
  const gen = SV.gen = (SV.gen || 0) + 1;                  // only the newest render draws (images load async)
  const parts = (m.steps.length ? await surveyImage(surveyView()) : null) || [];
  // fetch the previews before touching the canvas: clearing first and drawing after the downloads
  // shows an empty frame (a flash) on every refresh
  const pvs = await Promise.all((m.previews || []).map(async pv => ({ pv, c: await surveyPreview(pv.i, pv.cols) })));
  if (!$("#svMap") || gen !== SV.gen) return;
  const dpr = devicePixelRatio || 1, W = map.clientWidth;
  map.width = W * dpr; map.height = map.clientHeight * dpr;
  const g = map.getContext("2d"), span = SV.v1 - SV.v0, X = f => (f - SV.v0) / span * map.width;
  const cs = getComputedStyle(document.documentElement), mark = cs.getPropertyValue("--mark");
  g.fillStyle = cs.getPropertyValue("--wf-bg"); g.fillRect(0, 0, map.width, map.height);
  for (const { cv: c, st } of parts) {
    const a = Math.max(st.lo, SV.v0), b = Math.min(st.hi, SV.v1); if (b <= a) continue;
    const sx = (a - st.lo) / (st.hi - st.lo) * st.cols, sw = (b - a) / (st.hi - st.lo) * st.cols;
    g.imageSmoothingEnabled = (X(b) - X(a)) / sw < 3;      // crisp columns when zoomed in
    g.drawImage(c, sx, 0, Math.max(0.5, sw), c.height, X(a), 0, X(b) - X(a), map.height);
  }
  for (const { pv, c } of pvs) {                           // steps still being recorded or processed
    const a = Math.max(pv.lo, SV.v0), b = Math.min(pv.hi, SV.v1); if (b <= a || !c) continue;
    const sx = (a - pv.lo) / (pv.hi - pv.lo) * pv.cols, sw = (b - a) / (pv.hi - pv.lo) * pv.cols;
    g.imageSmoothingEnabled = (X(b) - X(a)) / sw < 3;
    g.drawImage(c, sx, 0, Math.max(0.5, sw), c.height, X(a), 0, X(b) - X(a), map.height);
  }
  g.strokeStyle = "rgba(255,255,255,.14)"; g.lineWidth = dpr;
  for (const st of [...m.steps, ...(m.previews || [])]) { const x = X(st.lo); if (x > 0 && x < map.width) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, map.height); g.stroke(); } }
  // detector picks as boxes, filtered by Min confidence
  const pad = 1500;
  for (const t of svTracks()) {
    if (t.conf < S.minConf) continue;
    const win = m.steps[t.step].dwell_s || m.dwell_s, Y = x => Math.min(1, x / win) * map.height;
    const x0 = X(t.fmin - pad), x1 = X(t.fmax + pad); if (x1 < 0 || x0 > map.width) continue;
    const w = Math.max(3 * dpr, x1 - x0), y0 = Y(t.t0), y1 = Math.max(y0 + 3 * dpr, Y(t.t1));
    g.strokeStyle = mark; g.lineWidth = (SV.hl === t.id ? 2.5 : 1.2) * dpr; g.globalAlpha = 0.5 + 0.5 * t.conf;
    g.strokeRect(x0 + (x1 - x0 - w) / 2, y0, w, y1 - y0); g.globalAlpha = 1;
  }
  if (SV.sel != null) { g.strokeStyle = cs.getPropertyValue("--text"); g.setLineDash([4 * dpr, 4 * dpr]); g.lineWidth = dpr; g.beginPath(); g.moveTo(X(SV.sel), 0); g.lineTo(X(SV.sel), map.height); g.stroke(); g.setLineDash([]); }
  // frequency axis
  const axis = $("#svAxis"), nt = Math.max(2, Math.floor(W / 110)), stepHz = svStep(span / nt), dec = Math.max(0, Math.min(4, -Math.floor(Math.log10(stepHz / 1e6) + 1e-9)));
  axis.replaceChildren();
  for (let f = Math.ceil(SV.v0 / stepHz) * stepHz; f <= SV.v1; f += stepHz)
    axis.append(el("span", { style: `left:${(f - SV.v0) / span * 100}%` }, (f / 1e6).toFixed(dec)));
  // activity per slice of what's on screen: tracked (grey) and at Min confidence (mark); green = voice-like
  const chart = $("#svChart"); chart.width = W * dpr; chart.height = chart.clientHeight * dpr;
  const cg = chart.getContext("2d"), nb = Math.max(10, Math.floor(W / 9)), bw = span / nb, mins = m.dwell_s / 60;
  const all = new Array(nb).fill(0), shown = new Array(nb).fill(0), voice = new Array(nb).fill(false);
  for (const t of svTracks()) { const b = Math.floor((t.f - SV.v0) / bw); if (b < 0 || b >= nb) continue; all[b]++; if (t.conf >= S.minConf) shown[b]++; if ((t.vad || 0) >= (m.floor ?? 0.3)) voice[b] = true; }
  const mx = Math.max(1, ...all), bx = chart.width / nb;
  for (let b = 0; b < nb; b++) {
    const h1 = all[b] / mx * (chart.height - 5 * dpr), h2 = shown[b] / mx * (chart.height - 5 * dpr);
    cg.fillStyle = cs.getPropertyValue("--line-strong"); cg.fillRect(b * bx + dpr, chart.height - h1, bx - 2 * dpr, h1);
    cg.fillStyle = mark; cg.fillRect(b * bx + dpr, chart.height - h2, bx - 2 * dpr, h2);
    if (voice[b]) { cg.fillStyle = cs.getPropertyValue("--ok"); cg.fillRect(b * bx + dpr, 0, bx - 2 * dpr, 3 * dpr); }
  }
  SV.bins = { nb, bw, all, shown, mins };
  $("#svSel").textContent = SV.sel != null ? fmtF(SV.sel) : "";
  $("#svTune").style.display = SV.sel != null ? "" : "none";
}
async function surveyPreview(i, cols) {
  try {
    const r = await fetch("/api/survey/preview?i=" + i); if (!r.ok) return null;
    const bytes = new Uint8Array(await r.arrayBuffer()), rows = +r.headers.get("X-Rows"), w = +r.headers.get("X-Cols");
    const c = document.createElement("canvas"); c.width = w; c.height = rows;
    const g = c.getContext("2d"), img = g.createImageData(w, rows), lut = makeColLUT(), bg = getComputedStyle(document.documentElement).getPropertyValue("--wf-bg");
    for (let k = 0; k < bytes.length; k++) {
      const v = bytes[k], o = k * 4;
      if (!v) { img.data[o + 3] = 0; continue; }                // not recorded yet: leave the background
      img.data[o] = lut[v * 4]; img.data[o + 1] = lut[v * 4 + 1]; img.data[o + 2] = lut[v * 4 + 2]; img.data[o + 3] = 255;
    }
    g.putImageData(img, 0, 0); return c;
  } catch (e) { return null; }
}
// while a survey runs and its map is open: refresh every 2 s (the step being recorded fills in)
setInterval(() => { if (SV.open && !SV.file && SV.state.state === "running") openSurvey(); }, 2000);
function surveyInteract(c) {
  let drag = null;
  const fAt = e => { const r = c.getBoundingClientRect(); return SV.v0 + (e.clientX - r.left) / r.width * (SV.v1 - SV.v0); };
  c.addEventListener("wheel", e => {                         // zoom the frequency axis around the cursor
    e.preventDefault(); const m = SV.meta; if (!m) return;
    const f = fAt(e), k = Math.exp(e.deltaY * 0.0015), span = Math.min(m.hi - m.lo, Math.max(5e3, (SV.v1 - SV.v0) * k));
    const r = (f - SV.v0) / (SV.v1 - SV.v0);
    SV.v0 = Math.max(m.lo, Math.min(m.hi - span, f - r * span)); SV.v1 = SV.v0 + span; renderSurvey();
  }, { passive: false });
  const pts = new Map();                                    // touch: two fingers pinch-zoom the frequency axis
  let pinch = null;
  c.addEventListener("pointerdown", e => {
    pts.set(e.pointerId, e.clientX); c.setPointerCapture(e.pointerId);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()], r = c.getBoundingClientRect();
      pinch = { d: Math.abs(a - b) || 1, mid: SV.v0 + ((a + b) / 2 - r.left) / r.width * (SV.v1 - SV.v0), v0: SV.v0, v1: SV.v1 }; drag = null; return;
    }
    drag = { x: e.clientX, v0: SV.v0, v1: SV.v1, moved: false };
  });
  const endPt = e => { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; };
  c.addEventListener("pointercancel", endPt);
  c.addEventListener("pointermove", e => {
    const tip = $("#svTip"), m = SV.meta; if (!m) return;
    if (pts.has(e.pointerId)) pts.set(e.pointerId, e.clientX);
    if (pinch && pts.size === 2) {
      const [a, b] = [...pts.values()], k = pinch.d / (Math.abs(a - b) || 1);
      const span = Math.min(m.hi - m.lo, Math.max(5e3, (pinch.v1 - pinch.v0) * k)), r = (pinch.mid - pinch.v0) / (pinch.v1 - pinch.v0);
      SV.v0 = Math.max(m.lo, Math.min(m.hi - span, pinch.mid - r * span)); SV.v1 = SV.v0 + span; renderSurvey(); return;
    }
    if (drag) {
      const dx = e.clientX - drag.x; if (Math.abs(dx) > 3) drag.moved = true;
      if (drag.moved) { const span = drag.v1 - drag.v0, df = -dx / c.getBoundingClientRect().width * span;
        SV.v0 = Math.max(m.lo, Math.min(m.hi - span, drag.v0 + df)); SV.v1 = SV.v0 + span; renderSurvey(); }
      tip.style.display = "none"; return;
    }
    const f = fAt(e), r = c.getBoundingClientRect(), B = SV.bins, b = B ? Math.floor((f - SV.v0) / B.bw) : -1;
    tip.style.display = ""; tip.style.left = Math.max(0, Math.min(r.width - 300, e.clientX - r.left + 12)) + "px";
    const st = m.steps.find(s => f >= s.lo && f < s.hi), cal = st && st.calibration;
    tip.textContent = fmtF(f) + (B && b >= 0 && b < B.nb ? ` · ${(B.all[b] / B.mins).toFixed(1)}/min tracked · ${(B.shown[b] / B.mins).toFixed(1)}/min ≥ ${Math.round(S.minConf * 100)}%` : "")
      + (cal ? ` · step calibrated on ${cal.candidates} candidates, ${cal.voice_like} speech-like` : st ? " · step too quiet to calibrate" : "");
  });
  c.addEventListener("pointerup", e => { const was = pinch; endPt(e); if (!was && drag && !drag.moved) { SV.sel = fAt(e); renderSurvey(); } drag = null; });
  c.addEventListener("pointerleave", () => { $("#svTip").style.display = "none"; });
  c.addEventListener("dblclick", () => { SV.v0 = SV.meta.lo; SV.v1 = SV.meta.hi; renderSurvey(); });
}
function renderSamples() {
  const box = $("#svSamples"); if (!box || !SV.meta) return;
  const floor = SV.meta.floor ?? 0.3;
  const best = svTracks().filter(t => t.clips && t.clips.length && t.conf >= S.minConf)
    .sort((a, b) => b.conf * (0.5 + (b.vad || 0)) - a.conf * (0.5 + (a.vad || 0))).slice(0, 10);
  if (!best.length) { box.replaceChildren(el("div", { class: "note" }, `No recorded detections at or above ${Math.round(S.minConf * 100)}% confidence.`)); return; }
  box.replaceChildren(...best.map(t => {
    const key = "sv:" + t.id, tx = S.tx.get(t.id);
    return el("div", { class: "svrow det", "data-key": key, title: "Click for details",
      onclick: () => { SV.hl = t.id; renderSurvey(); openDetail({ survey: { file: SV.file || SV.meta.file, step: SV.meta.steps[t.step], stepIndex: t.step, track: t } }); } },
      (() => { const b = el("button", { class: "play", title: "Play", onclick: e => { e.stopPropagation(); togglePlay(key, null, { clip: t.clips[0] }); } }); b.innerHTML = ICON_PLAY; return b; })(),
      el("span", { class: "num" }, fmtF(t.f)),
      el("span", { class: "t2" }, t.text ? `“${t.text}”` : tx && tx.text ? `“${tx.text}”` : `step ${t.step + 1} · ${t.t0.toFixed(1)} s`),
      el("span", { class: "num t2", title: "ML voice score" }, t.vad != null ? `voice ${Math.round(t.vad * 100)}%` : ""),
      el("span", { class: "num", style: t.vad != null && t.vad >= floor ? "color:var(--ok)" : "" }, Math.round(t.conf * 100) + "%"));
  }));
}
function svStep(x) { const p = 10 ** Math.floor(Math.log10(x)); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x) return m * p; return 10 * p; }
// Readiness: the server says how much continuous IQ exists for this band; between polls the
// count ticks up locally once a second (the ring grows in real time), so the timer moves smoothly.
function readyBlock() {
  const bar = el("div", { class: "tbar" }, el("i", { id: "tuneBarFill" }));
  const txt = el("span", { id: "tuneReadyTxt" });
  const act = el("span", { id: "tuneReadyAct" });
  return el("div", { id: "tuneReady", style: "margin-top:8px" }, bar, el("div", { class: "note", style: "margin:4px 0 0;display:flex;gap:10px;align-items:center" }, txt, act));
}
const fmtMS = s => `${Math.floor(s / 60)}:${pad2(Math.floor(s % 60))}`;
function tickReady() {
  const r = TUNE.ready, box = $("#tuneReady"); if (!r || !box) return;
  const running = TUNE.state.state === "running";
  const have = Math.min(r.need_s, r.have_s + (r.reason === "recording" ? (Date.now() - r.at) / 1000 : 0));
  const ready = r.ready || (r.reason === "recording" && have >= r.need_s * 0.95);
  box.style.display = r.ready && !running ? "none" : "";
  $("#tuneBarFill").style.width = (100 * have / r.need_s).toFixed(1) + "%";
  const txt = $("#tuneReadyTxt"), act = $("#tuneReadyAct");
  if (r.reason === "file") { txt.textContent = "Calibration needs a live receiver: it learns from IQ recorded at the current frequency."; act.replaceChildren(); }
  else if (r.reason === "off") {
    txt.textContent = `Raw IQ recording is off (${fmtMS(have)} of ${fmtMS(r.need_s)} recorded here).`;
    if (!act.firstChild) act.replaceChildren(el("button", { class: "btn", onclick: () => { patch({ iq_ring: { enabled: true } }, 50); r.reason = "recording"; r.at = Date.now(); } }, "Start recording IQ"));
  } else {
    const left = fmtMS(Math.max(0, r.need_s * 0.95 - have));
    const recal = r.recal_in_s != null ? Math.max(0, r.recal_in_s - (Date.now() - r.at) / 1000) : null;
    txt.textContent = r.auto && r.tuned
      ? (recal > 0 ? `Calibrated. Recalibrates if nothing is caught for ${fmtMS(recal)}.` : "Nothing caught lately; recalibrating shortly.")
      : ready ? (r.auto ? "Enough recorded; calibrating within 10 s." : "Ready.")
      : `Recording since the last receiver change: ${fmtMS(have)} of ${fmtMS(r.need_s)} · ${r.auto ? "calibrates in" : "ready in"} ${left}`;
    act.replaceChildren();
  }
  if (r.auto && r.tuned) box.style.display = "";
  const go = $("#tuneGo"); if (go) go.disabled = running || !ready;
}
async function pollReady() {
  if (!$("#detector").classList.contains("active")) return;
  try {
    const r = await (await fetch(`/api/autotune/ready?minutes=${TUNE.minutes}`)).json();
    TUNE.ready = { ...r, at: Date.now() }; tickReady();
  } catch (e) {}
}
setInterval(pollReady, 5000); setInterval(tickReady, 1000);
function onTune(m) {
  const was = TUNE.state.state; TUNE.state = m;
  const rb = $("#fRecal");
  if (rb) { const run = m.state === "running"; rb.disabled = run; rb.textContent = run ? "Calibrating…" : "Recalibrate"; }
  if (FIELD.on && was === "running" && m.state !== "running")
    toast(m.state === "done" && m.result ? `Calibrated on ${m.result.candidates} candidates, ${m.result.voice_like} speech-like` : (m.message || "Calibration finished"));
  if (m.state === "done" && was === "running" && m.result) setMinConf(m.result.show_threshold);
  if (m.state === "survey" && m.show_threshold != null) setMinConf(m.show_threshold);
  if (!$("#detector").contains(document.activeElement) || m.state !== "running") buildDetector();
}

// ---------------- receiver tab (advanced) ----------------
async function buildSDR() {
  const root = $("#sdr"), c = S.cfg, sdr = c.sdr; root.replaceChildren();
  const files = await (await fetch("/api/captures")).json();
  const num = (v, step = "any") => el("input", { class: "field", type: "number", step, value: v ?? "" });
  const fmin = num(sdr.fmin_hz == null ? "" : (sdr.fmin_hz / 1e6).toFixed(4)), fmax = num(sdr.fmax_hz == null ? "" : (sdr.fmax_hz / 1e6).toFixed(4));
  const rowW = (label, input) => el("div", { class: "row wide" }, el("label", {}, label), input);
  const fileSel = el("select", { class: "sel" }, ...files.map(f => el("option", { value: f }, f.replace(/^.*\//, ""))));
  fileSel.value = sdr.file.path;
  const srcSel = el("select", { class: "sel" }, el("option", { value: "hackrf" }, "HackRF"), el("option", { value: "rtlsdr" }, "RTL-SDR"), el("option", { value: "file" }, "Replay file"));
  srcSel.value = sdr.source; srcSel.addEventListener("change", () => setSource(srcSel.value));
  const ppm = num(sdr.ppm ?? 0, 1), bin = el("select", { class: "sel" }, ...[62.5, 125, 250, 500].map(b => el("option", { value: b }, b + " Hz")));
  bin.value = c.analysis.bin_hz;
  const apply = async (p) => { await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }); await fetch("/api/restart", { method: "POST" }); };
  const usage = el("div", { class: "note mono", style: "margin-top:0" }, "measuring…");
  fetch("/api/storage").then(r => r.json()).then(u => {
    const gb = b => (b / 1e9).toFixed(b < 1e9 ? 2 : 1) + " GB";
    usage.textContent = `IQ ring ${gb(u.iq_ring)} / ${c.iq_ring.max_gb} GB · recordings ${gb(u.recordings)} · captures ${gb(u.captures)} · disk free ${gb(u.free)}${u.paused ? " · WRITING PAUSED (low disk)" : ""}`;
    if (u.paused) usage.style.color = "var(--bad)";
  });
  const themeSel = el("select", { class: "sel" }, ...THEMES.map(([k, l]) => el("option", { value: k }, l)));
  { const t = lsGet("gb_theme", "instrument"); themeSel.value = THEMES.some(([k]) => k === t) ? t : "instrument"; } themeSel.addEventListener("change", () => setTheme(themeSel.value));
  const pres = await (await fetch("/api/presets")).json();
  const presetSel = el("select", { class: "sel" }, el("option", { value: "" }, "Choose…"),
    el("optgroup", { label: "Built in" }, ...Object.entries(pres.builtin).map(([k, v]) => el("option", { value: "b:" + k }, v))),
    pres.saved.length ? el("optgroup", { label: "Saved" }, ...pres.saved.map(p => el("option", { value: "f:" + p.file }, p.label))) : null);
  presetSel.addEventListener("change", async () => {
    const v = presetSel.value; if (!v) return;
    const body = v.startsWith("f:") ? { file: v.slice(2) } : { name: v.slice(2) };
    await fetch("/api/preset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    presetSel.value = "";
  });
  const presetName = el("input", { class: "field", placeholder: "Name, e.g. basement session", maxlength: 60 });
  const saveBtn = el("button", { class: "btn", onclick: async () => {
    const r = await (await fetch("/api/presets", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: presetName.value }) })).json();
    presetName.value = ""; buildSDR();
  } }, "Save");
  const savedList = el("div", {}, ...pres.saved.map(p => el("div", { class: "row wide", style: "min-height:26px" },
    el("label", { title: p.file }, p.label),
    el("button", { class: "btn", style: "justify-self:end;height:24px", title: "Delete " + p.file, onclick: async () => {
      if (!confirm(`Delete the preset "${p.label}"?`)) return;
      await fetch("/api/presets/" + encodeURIComponent(p.file), { method: "DELETE" }); buildSDR();
    } }, "Delete"))));
  const mkScale = (label, key, init) => {
    const out = el("output", {}, String(Math.round(init)));
    const inp = el("input", { type: "range", min: 0, max: 255, step: 1, value: Math.round(init), disabled: S.autoScale }); fillRange(inp);
    inp.addEventListener("input", () => { fillRange(inp); out.textContent = inp.value;
      if (key === "min") S.smin = Math.min(+inp.value, S.smax - 1); else S.smax = Math.max(+inp.value, S.smin + 1);
      saveScale(); repaintAll(); });
    return el("div", { class: "row" }, el("label", {}, label), inp, out);
  };
  root.append(
    el("h2", {}, "Presets"),
    rowW("Load a preset", presetSel),
    el("div", { class: "row wide" }, el("label", {}, "Save current settings as"),
      el("div", { style: "display:flex;gap:6px" }, presetName, saveBtn)),
    el("div", { class: "note", style: "margin:2px 0 6px" }, "Saves the receiver, detector and playback settings. Files go in live_data/presets, named with the date, device and frequency range."),
    savedList,
    el("h2", {}, "Appearance"),
    rowW("Theme (this browser)", themeSel),
    rowW("Layout (this browser)", (() => { const l = el("select", { class: "sel" }, el("option", { value: "auto" }, "Automatic (phone on narrow screens)"), el("option", { value: "phone" }, "Phone"), el("option", { value: "desktop" }, "Desktop"));
      l.value = lsGet("gb_layout", "auto"); l.addEventListener("change", () => setLayoutPref(l.value)); return l; })()),
    el("div", { class: "row wide" }, el("label", {}, "Automatic waterfall brightness"), toggle(S.autoScale, v => {
      S.autoScale = v; lsSet("gb_autoscale", v ? "1" : "0"); if (!v) loadScale(); repaintAll(); buildSDR(); })),
    mkScale("Waterfall minimum", "min", S.smin),
    mkScale("Waterfall maximum", "max", S.smax),
    el("h2", {}, "Analysis range"),
    rowW("From (MHz, blank = whole capture)", fmin), rowW("To (MHz, blank = whole capture)", fmax),
    el("div", { style: "margin:8px 0" }, el("button", { class: "btn", onclick: () => apply({ sdr: { fmin_hz: fmin.value === "" ? null : +fmin.value * 1e6, fmax_hz: fmax.value === "" ? null : +fmax.value * 1e6 } }) }, "Apply range")),
    el("h2", {}, "Replay a recording"),
    rowW("Capture or IQ-ring segment", fileSel),
    el("div", { style: "margin:8px 0" }, el("button", { class: "btn primary", onclick: () => apply({ sdr: { source: "file", file: { path: fileSel.value }, fmin_hz: null, fmax_hz: null } }) }, "Replay this file")),
    el("div", { class: "note" }, "IQ-ring segments are the continuous raw recording (newest first). Frequency and rate come from the file name."),
    el("h2", {}, "Receiver"),
    rowW("Receiver type", srcSel),
    deviceRows(sdr),
    rowW("RTL-SDR ppm correction", ppm),
    rowW("Bin width", bin),
    el("div", { style: "margin:8px 0" }, el("button", { class: "btn", onclick: () => apply({ sdr: { ppm: +ppm.value }, analysis: { bin_hz: +bin.value } }) }, "Apply")),
    el("h2", {}, "Playback audio"),
    selectRow("Demodulator", [["auto", "Auto (follows the carrier; recovers it when there is none)"], ["sync", "Synchronous AM (needs a carrier)"], ["suppressed", "Suppressed carrier (no carrier needed)"], ["sync_sb", "Synchronous sideband (cleaner side)"], ["envelope", "Envelope"]],
      c.audio.demod || "auto", v => patch({ audio: { demod: v } }, 50)),
    el("div", { class: "row wide" }, el("label", { title: "Hard band-pass: flat inside, at least 50 dB down outside" }, "Voice band filter"), toggle(c.audio.voice_band !== false, v => patch({ audio: { voice_band: v } }, 50))),
    slider("Voice band from", ["audio", "voice_lo_hz"], 100, 800, 25, v => v + " Hz"),
    slider("Voice band to", ["audio", "voice_hi_hz"], 1800, 5000, 100, v => v + " Hz"),
    slider("Lead-in before a detection", ["audio", "preroll_s"], 0, 1.5, 0.05, v => v.toFixed(2) + " s"),
    slider("Tail after a detection", ["audio", "postroll_s"], 0, 1.2, 0.05, v => v.toFixed(2) + " s"),
    slider("Fade in / out", ["audio", "fade_s"], 0, 0.5, 0.01, v => Math.round(v * 1000) + " ms"),
    el("div", { class: "row wide" }, el("label", {}, "Noise reduction"), toggle(c.audio.denoise !== false, v => patch({ audio: { denoise: v } }))),
    slider("Noise reduction strength", ["audio", "denoise_strength"], 0, 2, 0.05, v => v.toFixed(2)),
    slider("Keep noise at", ["audio", "denoise_floor_db"], -30, -3, 1, v => v + " dB"),
    el("div", { class: "note" }, "Applies to new detections: what you hear, saved recordings and transcripts. The detector always judges the unprocessed audio. Strength 1 lowers background noise by about 10 dB; the floor keeps some atmosphere."),
    el("h2", {}, "Storage"),
    usage,
    el("div", { class: "row wide" }, el("label", {}, "Record raw IQ continuously"), toggle(c.iq_ring.enabled, v => patch({ iq_ring: { enabled: v } }))),
    slider("IQ ring size", ["iq_ring", "max_gb"], 0.5, 100, 0.5, v => v + " GB"),
    el("div", { class: "row wide" }, el("label", {}, "Keep a small IQ clip with each recording"), toggle(c.record.save_iq, v => patch({ record: { save_iq: v } }))),
    slider("Save to disk at confidence ≥", ["record", "min_save_conf"], 0, 1, 0.01, v => Math.round(v * 100) + "%"),
    slider("Detection audio kept in memory", ["record", "memory_mb"], 50, 4000, 50, v => (v >= 1000 ? (v / 1000).toFixed(1) + " GB" : v + " MB")),
    slider("Recordings size", ["record", "max_mb"], 100, 20000, 100, v => (v >= 1000 ? (v / 1000).toFixed(1) + " GB" : v + " MB")),
    slider("Always keep free on disk", ["storage", "min_free_gb"], 1, 100, 1, v => v + " GB"),
    slider("Waterfall history (memory)", ["display", "history_min"], 2, 60, 1, v => v + " min"),
    el("div", { class: "note" }, "Power change view: each frequency compared with its own recent average, so steady carriers disappear and only what rises above usual shows."),
    slider("Power change: compare with the last", ["display", "change_tau_s"], 2, 600, 1, v => v >= 60 ? (v / 60).toFixed(1) + " min" : v + " s"),
    slider("Power change: ignore rises below", ["display", "change_floor_db"], 0, 20, 0.5, v => v + " dB"),
    slider("Power change: colour range", ["display", "change_range_db"], 5, 60, 1, v => v + " dB"),
    el("div", { class: "note" }, `Rings delete their oldest files when full. Raw IQ is 2 bytes per sample: ${(2 * sdr.sample_rate / 1e6).toFixed(0)} MB/s at the current rate, so ${c.iq_ring.max_gb} GB holds about ${Math.round(c.iq_ring.max_gb * 1e9 / (2 * sdr.sample_rate))} s. If free disk space drops below the limit, the IQ ring is trimmed further and then all writing pauses (detection keeps running).`));
}

// ---------------- waterfall toolbar ----------------
function loadScale() {
  const sc = JSON.parse(lsGet("gb_scale_" + S.view, "null") || "null") || { min: 0, max: 255 };
  S.smin = sc.min; S.smax = sc.max;
}
const saveScale = () => lsSet("gb_scale_" + S.view, JSON.stringify({ min: S.smin, max: S.smax }));
$("#cmap").value = S.cmap;
$("#cmap").addEventListener("change", e => setCmap(e.target.value));
$("#minConf").value = Math.round(S.minConf * 100); fillRange($("#minConf")); $("#minConfOut").textContent = Math.round(S.minConf * 100) + "%";
let confTimer = null;
function setMinConf(v) {
  S.minConf = v; lsSet("gb_minconf", v);
  $("#minConf").value = Math.round(v * 100); fillRange($("#minConf")); $("#minConfOut").textContent = Math.round(v * 100) + "%";
  renderLive(); clearTimeout(confTimer); confTimer = setTimeout(() => loadPage(true), 250);
  if (SV.open) { renderSurvey(); renderSamples(); }
  if ($("#fMin")) syncFieldMin();
}
$("#minConf").addEventListener("input", e => setMinConf(+e.target.value / 100));
$("#caps").checked = lsGet("gb_caps", "1") === "1";

$("#caps").addEventListener("change", e => lsSet("gb_caps", e.target.checked ? "1" : "0"));
let WS = null;
function setView(mode) {
  S.view = mode; lsSet("gb_view", mode); $("#view").value = mode;
  S.chunks.clear(); loadScale(); LUT = makeLUT();
  if (SV.open) drawSurvey();
  if (WS && WS.readyState === 1) WS.send(JSON.stringify({ type: "view", mode }));
  const top = topTime(); if (top !== null) fetchHistory(top - cv.clientHeight / S.pps - 30, top + 5);
}
$("#view").addEventListener("change", e => setView(e.target.value));

// Waterfall views: checks at weight 0 are not computed, so only those are listed.
function availableViews() {
  const sv = SV.open && SV.meta ? new Set(SV.meta.views) : null;
  return (S.views || []).filter(v => !v.check || (S.cfg && S.cfg.checks[v.id] && S.cfg.checks[v.id].weight > 0) || (sv && sv.has(v.id)));
}
function buildViewList() {
  const vs = availableViews(), sel = $("#view");
  const opts = [...vs.filter(v => !v.check).map(v => el("option", { value: v.id }, v.label))];
  const checks = vs.filter(v => v.check);
  if (checks.length) opts.push(el("optgroup", { label: "Detector maps (checks above 0)" }, ...checks.map(v => el("option", { value: v.id }, v.label))));
  sel.replaceChildren(...opts);
  if (!vs.some(v => v.id === S.view)) { if (S.view !== "power") setView("power"); }
  else sel.value = S.view;
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  if (name === "sdr") buildSDR();
  if (name === "detector") pollReady();
}
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// ---------------- status ----------------
function setStatus(s) {
  S.status = s; syncClock(s.t);
  if (s.fmin && s.fmax) { S.fmin = s.fmin; S.fmax = s.fmax; }
  const ok = s.state === "running";
  $("#dot").className = ok ? "ok" : (s.state === "error" || s.state === "no data") ? "bad" : "";
  const st = $("#status");
  st.className = ok ? "" : "bad";
  const w = s.whisper || {};
  const behind = ok && s.load > 1;
  st.className = ok && !behind ? "" : "bad";
  const paused = ok && s.storage && s.storage.paused;
  if (paused) st.className = "bad";
  st.textContent = ok ? `${paused ? `DISK LOW (${s.storage.free_gb} GB free) — recording paused · ` : ""}CPU ${(s.load * 100).toFixed(0)}%${behind ? " — can't keep up: choose a smaller span (top bar), narrow the analysis range (Settings), or raise the detector thresholds" : ""}${s.dropped ? ` · dropped ${s.dropped}` : ""}${w.queue || w.busy ? ` · transcribing ${w.queue} queued${w.rate ? `, ${w.rate}/s` : ""}` : ""}${s.error ? " · " + s.error : ""}`
    : `${s.state}${s.error ? ": " + s.error : ""}`;
  st.title = st.textContent;
}

// ---------------- websocket ----------------
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  WS = ws;
  ws.onopen = () => { if (S.view !== "power") ws.send(JSON.stringify({ type: "view", mode: S.view })); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello": {
        S.cfg = m.config; S.checks = m.checks; S.groups = m.check_groups || []; S.presets = m.presets; S.sens = m.sensitivity; S.views = m.views; S.epoch = m.epoch; S.asrModels = m.asr_models; S.root = m.root;
        fetch("/api/marks").then(r => r.json()).then(x => { S.marks = x; }).catch(() => {});
        S.hop = m.config.analysis.hop_s; S.lag = m.config.analysis.settle_s + m.config.analysis.hop_s + 0.25;
        S.delay = Math.max(S.lag, S.delay); S.lineY = S.delay - S.lag;
        if (m.reset && m.reset.fmax) { S.fmin = m.reset.fmin; S.fmax = m.reset.fmax; }
        buildViewList();
        if (!availableViews().some(v => v.id === S.view)) S.view = "power";
        $("#view").value = S.view; loadScale(); LUT = makeLUT();

        setStatus(m.status || {});
        for (const b of m.detections) S.dets.set(b.id, { box: b, ended: false, playing: [], pcm: [] });
        buildTop(); buildDetector(); loadPage(true);
        Promise.all([fetch("/api/autotune").then(r => r.json()).then(t => { TUNE.state = t; }), loadSurveyList()]).then(buildDetector);
        const t = m.status && m.status.t; if (t) fetchHistory(t - 240, t + 5);
        break;
      }
      case "config": {
        S.cfg = m.config; buildViewList(); S.lag = m.config.analysis.settle_s + m.config.analysis.hop_s + 0.25;
        buildTop(); if (!$("#detector").contains(document.activeElement)) buildDetector(); break;
      }
      case "reset": if (m.fmax) { S.fmin = m.fmin; S.fmax = m.fmax; } break;
      case "status": setStatus(m); break;
      case "wf": addChunk(m); break;
      case "det": S.dets.set(m.box.id, { box: m.box, ended: false, playing: [], pcm: [] }); renderLive(); break;
      case "det_update": { const d = S.dets.get(m.box.id); if (d) { d.box = m.box; d.ended = false; } else S.dets.set(m.box.id, { box: m.box, ended: false, playing: [], pcm: [] }); renderLive(); break; }
      case "det_end": { const d = S.dets.get(m.box.id); if (d) { d.box = m.box; d.ended = true; } renderLive(); break; }
      case "audio": onAudio(m); break;
      case "recording": onRecording(m.meta); break;
      case "saved": onSaved(m); break;
      case "autotune": onTune(m); break;
      case "survey": onSurvey(m); break;
      case "mark": (S.marks = S.marks || []).push(m); break;
      case "transcript": onTranscript(m); break;
    }
  };
  ws.onclose = () => { setStatus({ state: "disconnected" }); setTimeout(connect, 1500); };
}
fillRange($("#vol"));
connect();
requestAnimationFrame(draw);
$("#mapBtn").addEventListener("click", openMap);

// ---------------- detection detail dialog ----------------
// Opens from a detection row or a survey sample: full transcript, playback with waveform and
// spectrogram (both downloadable), downloads (audio WAV, narrowband IQ), and everything measured.
var DT = { open: false, buf: null, src: null, t0: 0, off: 0, raf: 0, spec: null };
function fftInPlace(re, im) {                             // iterative radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k], vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci, vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi; re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      } }
  }
}
function specOf(x, rate) {                                // dB STFT, 0-4 kHz: [cols][bins]
  const N = 512, hop = 128, win = new Float32Array(N).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  const bins = Math.min(N / 2, Math.round(4000 / (rate / N))), cols = [];
  for (let s = 0; s + N <= x.length; s += hop) {
    const re = new Float32Array(N), im = new Float32Array(N);
    for (let i = 0; i < N; i++) re[i] = x[s + i] * win[i];
    fftInPlace(re, im);
    const c = new Float32Array(bins); for (let k = 0; k < bins; k++) c[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12);
    cols.push(c);
  }
  const all = []; for (let i = 0; i < cols.length; i += 3) for (let k = 0; k < bins; k += 3) all.push(cols[i][k]);
  all.sort((a, b) => a - b);
  return { cols, bins, lo: all[Math.floor(all.length * 0.2)] || -100, hi: all[Math.floor(all.length * 0.998)] || 0, hop, rate };
}
const labelOf = id => ((S.checks || []).find(c => c.id === id) || {}).label || id;
function closeDetail() {
  dtStop(); DT.open = false; const m = $("#detail"); if (m) m.remove();
  document.removeEventListener("keydown", dtKey, true);
}
function dtKey(e) { if (e.key === "Escape") { e.stopPropagation(); closeDetail(); } else if (e.key === " " && DT.open) { e.preventDefault(); e.stopPropagation(); dtToggle(); } }
// opts: { key } for a detection (memory or saved), { live: d } for one still running, { survey: { file, step, track } }
async function openDetail(opts) {
  closeDetail(); ensureAudio();
  DT.open = true; DT.buf = null; DT.off = 0;
  const box = el("div", { class: "dlg", role: "dialog", "aria-modal": "true" });
  const wrap = el("div", { id: "detail", onclick: e => { if (e.target === wrap) closeDetail(); } }, box);
  document.body.append(wrap); document.addEventListener("keydown", dtKey, true);
  box.append(el("div", { class: "dlghead" }, el("strong", {}, "Loading…"), el("span", { style: "flex:1" }), el("button", { class: "btn", onclick: closeDetail }, "Close")));
  // ---- gather what we know
  let info = {}, clip = null, iqUrl = null, jump = null, tx = null;
  if (opts.survey) {
    const { file, step: st, track: t } = opts.survey;
    info = { f: t.f, fmin: t.fmin, fmax: t.fmax, dur: t.dur ?? (t.t1 - t.t0), conf: t.conf, scores: t.scores || { vad: t.vad }, info: t.info || {},
      when: `Survey ${file ? file.replace(/\.json$/, "").replace(/_/g, " ") : ""} · step ${opts.survey.stepIndex + 1} at ${t.t0.toFixed(1)} s`,
      extra: st.calibration ? [["Step calibration", `${st.calibration.candidates} candidates, ${st.calibration.voice_like} speech-like${st.calibration.kept_checks ? " (kept current checks)" : ""}`]] : [],
      rx: { center: st.fc, rate: st.fs }, mode: t.mode };
    tx = t.text ? { text: t.text } : null;
    clip = t.clips && t.clips[0];
    iqUrl = `/api/iq?survey=${encodeURIComponent(file || SV.meta.file)}&track=${t.id}`;
  } else {
    const key = opts.key || (opts.live && (opts.live.box.clips || [])[0] || "").split("key=")[1];
    let meta = null;
    if (key) { try { const r = await fetch("/api/detection?key=" + encodeURIComponent(decodeURIComponent(key))); if (r.ok) meta = await r.json(); } catch (e) {} }
    const b = opts.live ? opts.live.box : null;
    if (meta) {
      info = { f: meta.freq_hz.mean, fmin: meta.freq_hz.min, fmax: meta.freq_hz.max, dur: meta.duration_s, conf: meta.confidence.max, confFinal: meta.confidence.final,
        scores: meta.scores || {}, info: meta.info || {}, when: new Date(meta.utc_start).toLocaleString(), utc: meta.utc_start,
        hist: meta.confidence.history, rx: { center: meta.fc, rate: meta.fs, gain: meta.sdr && meta.sdr.gain_db, amp: meta.sdr && meta.sdr.amp, source: meta.sdr && meta.sdr.source },
        extra: [["Chain / segment", `${meta.chain} / ${meta.segment}`], ["Saved", meta.saved === false ? "In memory only" : "On disk"]], mode: meta.mode };
      tx = meta.transcript || S.tx.get(meta.id) || null;
      clip = meta.clip || (meta.files && meta.files.wav ? "/rec/" + meta.files.wav : null);
      iqUrl = "/api/iq?key=" + encodeURIComponent(meta.key);
      if (meta.epoch && Math.abs(meta.epoch - S.epoch) < 1e-3) jump = meta.stream_t1;
    } else if (b) {
      info = { f: (b.fmin + b.fmax) / 2, fmin: b.fmin, fmax: b.fmax, dur: b.t1 - b.t0, conf: b.conf, scores: b.scores || {}, info: b.info || {}, when: clockTime(b.t0) + " (still live)" };
      tx = S.tx.get(b.id) || null; jump = b.t1;
    }
  }
  if (!DT.open) return;
  const renderBase = iqUrl ? iqUrl.split("?")[1] : null;
  DT.clip = clip;
  // ---- header, transcript
  box.replaceChildren();
  box.append(el("div", { class: "dlghead" },
    el("strong", { class: "num" }, fmtF(info.f || 0)), el("span", { class: "t2" }, info.when || ""),
    el("span", { class: "t2 num" }, `${(info.dur || 0).toFixed(1)} s · ${Math.round((info.conf || 0) * 100)}%`),
    el("span", { style: "flex:1" }), el("button", { class: "btn", title: "Close (Esc)", onclick: closeDetail }, "Close")));
  const txText = tx && tx.text ? tx.text : null;
  box.append(el("div", { class: "dlgtx" + (txText ? "" : " t3") }, txText ? `“${txText}”` : tx ? "No words" : "No transcript"),
    tx && tx.filtered && tx.filtered !== tx.text && /\p{L}/u.test(tx.filtered) ? el("div", { class: "note", style: "margin:2px 0 0" }, `Before filtering: “${tx.filtered}”`) : "");
  // ---- player: waveform + spectrogram with a playhead; click to seek
  const wave = el("canvas", { class: "dlgwave" }), spec = el("canvas", { class: "dlgspec" }), head = el("div", { class: "dlgplayhead" });
  const playBtn = el("button", { class: "btn primary", id: "dtPlay", onclick: dtToggle, disabled: !clip }, "Play");
  const time = el("span", { class: "num t2", id: "dtTime" }, "");
  const pl = el("div", { class: "dlgplot" }, wave, spec, head);
  pl.addEventListener("click", e => { if (!DT.buf) return; const r = pl.getBoundingClientRect(); dtSeek((e.clientX - r.left) / r.width * DT.buf.duration); });
  box.append(el("div", { class: "dlgplayer" }, el("div", { style: "display:flex;gap:10px;align-items:center;margin-bottom:6px" }, playBtn, time,
    el("span", { style: "flex:1" }), el("span", { class: "t3 kbdhint", style: "font-size:12px" }, "Space plays · click the plot to seek")), pl,
    renderBase ? abControls(renderBase, clip) : el("div", { class: "note" }, "A/B listening needs this detection's IQ, which isn't available.")));
  // ---- downloads and actions
  const fname = `${(info.f / 1e6).toFixed(4)}MHz_${(info.utc || "survey").slice(0, 19).replace(/[-:]/g, "").replace("T", "-")}`;
  const dl = (label, href, name, title) => el("a", { class: "btn", href, download: name, title }, label);
  const pngBtn = (label, cv, suffix) => el("button", { class: "btn", onclick: () => cv.toBlob(b => { const a = el("a", { href: URL.createObjectURL(b), download: `${fname}_${suffix}.png` }); a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }) }, label);
  box.append(el("div", { class: "dlgacts" },
    clip ? dl("Audio WAV", clip, `${fname}_audio.wav`, "Demodulated audio, 16 kHz") : null,
    iqUrl ? dl("IQ WAV", iqUrl, `${fname}_iq48k.wav`, "Raw IQ around this detection (±1 s), shifted to its frequency, 48 kHz stereo I/Q — opens in SDR#, SDR++ and similar") : null,
    pngBtn("Spectrogram PNG", spec, "spectrogram"), pngBtn("Waveform PNG", wave, "waveform"),
    el("span", { style: "flex:1" }),
    jump != null ? el("button", { class: "btn", onclick: () => { closeDetail(); if (SV.open) closeSurvey(); jumpTo(jump); } }, "Show on waterfall") : null,
    opts.survey ? el("button", { class: "btn", onclick: () => { const t = opts.survey.track; closeDetail(); SV.hl = t.id; SV.sel = t.f; SV.v0 = Math.max(SV.meta.lo, t.f - 60e3); SV.v1 = Math.min(SV.meta.hi, t.f + 60e3); renderSurvey(); } }, "Show on map") : null,
    el("button", { class: "btn", title: "Move the receiver here", onclick: () => { closeDetail(); if (SV.open) surveyTune(info.f); else { const step = S.cfg.sdr.sample_rate * 0.8; patch({ sdr: { center_hz: Math.round(info.f), fmin_hz: Math.round(info.f - step / 2), fmax_hz: Math.round(info.f + step / 2) } }, 50); } } }, "Tune here")));
  // ---- the numbers
  const kv = [];
  const fmtMode = m => m === 1 ? "USB" : m === -1 ? "LSB" : m === 0 ? "AM" : null;
  kv.push(["Frequency", `${fmtF(info.f)}${info.fmax - info.fmin > 200 ? ` (${fmtF(info.fmin)} – ${fmtF(info.fmax)}, drift ${((info.fmax - info.fmin) / 1e3).toFixed(1)} kHz)` : ""}`]);
  if (info.confFinal != null) kv.push(["Confidence", `max ${Math.round(info.conf * 100)}%, final ${Math.round(info.confFinal * 100)}%`]);
  const I = { ...(info.info || {}), ...(info.mode != null ? { mode: info.mode } : {}) };
  if (fmtMode(I.mode) || I.mode_name) kv.push(["Modulation", I.mode_name || fmtMode(I.mode)]);
  if (I.f0_median_hz) kv.push(["Pitch", `${Math.round(I.f0_median_hz)} Hz`]);
  if (I.voiced_frac != null) kv.push(["Voiced", `${Math.round(I.voiced_frac * 100)}% of frames`]);
  if (I.am_phase_z != null) kv.push(["AM cross-term", `z = ${I.am_phase_z.toFixed(1)}`]);
  if (info.rx) kv.push(["Receiver", [info.rx.source, info.rx.center ? `${fmtF(info.rx.center)} centre` : "", info.rx.rate ? `${(info.rx.rate / 1e6).toFixed(1)} MS/s` : "", info.rx.gain != null ? `gain ${info.rx.gain} dB` : "", info.rx.amp ? "amp on" : ""].filter(Boolean).join(" · ")]);
  for (const e of info.extra || []) kv.push(e);
  const scores = Object.entries(info.scores || {}).filter(([, v]) => v != null).sort((a, b) => b[1] - a[1]);
  box.append(el("div", { class: "dlggrid" },
    el("div", {}, el("h3", {}, "Details"), ...kv.map(([k, v]) => el("div", { class: "kv" }, el("span", { class: "t3" }, k), el("span", {}, v)))),
    el("div", {}, el("h3", {}, "Check scores"), ...(scores.length ? scores.map(([k, v]) => el("div", { class: "kv sc" }, el("span", { class: "t2" }, k === "vad" ? "ML voice detector" : labelOf(k)),
      el("span", { class: "bar" }, el("i", { style: `width:${Math.round(v * 100)}%` })), el("span", { class: "num" }, Math.round(v * 100)))) : [el("div", { class: "note" }, "None recorded.")]),
      info.hist && info.hist.length > 2 ? el("h3", { style: "margin-top:12px" }, "Confidence over time") : "", info.hist && info.hist.length > 2 ? confSpark(info.hist) : "")));
  // ---- load and draw the audio
  if (clip) {
    try {
      const r = await fetch(clip); if (!r.ok) throw new Error("gone");
      DT.buf = await AC.decodeAudioData(await r.arrayBuffer());
    } catch (e) { playBtn.disabled = true; time.textContent = "Audio no longer available"; }
  } else time.textContent = "No audio recorded";
  if (!DT.open) return;
  if (DT.buf) {
    DT.spec = specOf(DT.buf.getChannelData(0), DT.buf.sampleRate);
    time.textContent = `0.0 / ${DT.buf.duration.toFixed(1)} s`;
  }
  requestAnimationFrame(() => { drawDetailWave(); drawDetailSpec(); });
}
function confSpark(hist) {
  const c = el("canvas", { class: "dlgspark" });
  requestAnimationFrame(() => {
    const dpr = devicePixelRatio || 1, W = c.clientWidth * dpr, H = c.clientHeight * dpr; c.width = W; c.height = H;
    const g = c.getContext("2d"), t0 = hist[0][0], t1 = hist[hist.length - 1][0] || t0 + 1;
    g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--mark"); g.lineWidth = 1.5 * dpr; g.beginPath();
    hist.forEach(([t, v], i) => { const x = (t - t0) / Math.max(1e-6, t1 - t0) * W, y = H - v * (H - 2 * dpr) - dpr; i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
  });
  return c;
}
function drawDetailWave() {
  const c = $("#detail .dlgwave"); if (!c) return;
  const dpr = devicePixelRatio || 1, W = c.clientWidth * dpr, H = c.clientHeight * dpr; c.width = W; c.height = H;
  const g = c.getContext("2d"), cs = getComputedStyle(document.documentElement);
  g.fillStyle = cs.getPropertyValue("--wf-bg"); g.fillRect(0, 0, W, H);
  if (!DT.buf) return;
  const x = DT.buf.getChannelData(0), per = Math.max(1, Math.floor(x.length / W));
  g.fillStyle = cs.getPropertyValue("--mark");
  for (let i = 0; i < W; i++) {
    let lo = 1, hi = -1; for (let k = i * per; k < Math.min(x.length, (i + 1) * per); k++) { if (x[k] < lo) lo = x[k]; if (x[k] > hi) hi = x[k]; }
    if (hi < lo) continue; g.fillRect(i, H / 2 - hi * H * 0.46, 1, Math.max(1, (hi - lo) * H * 0.46));
  }
}
function drawDetailSpec() {
  const c = $("#detail .dlgspec"); if (!c) return;
  const dpr = devicePixelRatio || 1, W = c.clientWidth * dpr, H = c.clientHeight * dpr; c.width = W; c.height = H;
  const g = c.getContext("2d");
  if (!DT.spec) { g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--wf-bg"); g.fillRect(0, 0, W, H); return; }
  const { cols, bins, lo, hi } = DT.spec, lut = makeColLUT(), img = g.createImageData(cols.length, bins), span = Math.max(10, hi - lo);
  for (let x = 0; x < cols.length; x++) for (let k = 0; k < bins; k++) {
    const v = Math.max(0, Math.min(255, Math.round((cols[x][k] - lo) / span * 255))) * 4, o = ((bins - 1 - k) * cols.length + x) * 4;
    img.data[o] = lut[v]; img.data[o + 1] = lut[v + 1]; img.data[o + 2] = lut[v + 2]; img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = cols.length; off.height = bins; off.getContext("2d").putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, W, H);
  g.fillStyle = "rgba(255,255,255,.6)"; g.font = `${10 * dpr}px ${TH.mono}`;
  for (const f of [1000, 2000, 3000]) { const y = H * (1 - f / 4000); g.fillText(f / 1000 + "k", 4 * dpr, y + 3 * dpr); }
}
function dtStop() {
  if (DT.src) { try { DT.src.onended = null; DT.src.stop(); } catch (e) {} DT.off = Math.min(DT.buf ? DT.buf.duration : 0, DT.off + AC.currentTime - DT.t0); }
  DT.src = null; cancelAnimationFrame(DT.raf); if (PB.gain) PB.gain.gain.value = 1;
  const b = $("#dtPlay"); if (b) b.textContent = "Play";
}
function dtToggle() {
  if (!DT.buf) return;
  if (DT.src) { dtStop(); return; }
  if (S.playing) togglePlay(S.playing.key);                // one clip at a time
  if (DT.off >= DT.buf.duration - 0.02) DT.off = 0;
  const src = AC.createBufferSource(); src.buffer = DT.buf; src.connect(master); src.start(0, DT.off);
  DT.src = src; DT.t0 = AC.currentTime; if (PB.gain) PB.gain.gain.value = 0;   // the clip has the floor
  src.onended = () => { if (DT.src === src) { DT.src = null; DT.off = DT.buf.duration; dtTick(); const b = $("#dtPlay"); if (b) b.textContent = "Play"; if (PB.gain) PB.gain.gain.value = 1; } };
  $("#dtPlay").textContent = "Pause";
  const loop = () => { dtTick(); if (DT.src) DT.raf = requestAnimationFrame(loop); }; loop();
}
function dtSeek(t) { const was = !!DT.src; if (was) dtStop(); DT.off = Math.max(0, Math.min(DT.buf.duration, t)); dtTick(); if (was) dtToggle(); }
function dtTick() {
  if (!DT.buf) return;
  const t = DT.src ? DT.off + AC.currentTime - DT.t0 : DT.off, h = $("#detail .dlgplayhead"), tm = $("#dtTime");
  if (h) h.style.left = (Math.min(1, t / DT.buf.duration) * 100) + "%";
  if (tm) tm.textContent = `${t.toFixed(1)} / ${DT.buf.duration.toFixed(1)} s`;
}
// A/B: the same clip re-rendered from its IQ with another demodulator, noise reduction or voice band.
function abControls(base, clip) {
  const au = S.cfg.audio;
  DT.ab = { demod: "rec", denoise: au.denoise !== false, band: au.voice_band !== false }; DT.tunedUrl = null;
  const seg = (items, key) => el("div", { class: "abseg" }, ...items.map(([v, l]) => el("button", { class: "btn" + (DT.ab[key] === v ? " on" : ""), "data-v": String(v),
    onclick: e => { DT.ab[key] = v; e.target.parentNode.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === e.target)); abLoad(base, clip); } }, l)));
  const note = el("span", { id: "abNote", class: "t3", style: "font-size:12px" });
  const tuneOut = el("div", { id: "tuneOut", class: "tuneout" });
  const tuneBtn = el("button", { class: "btn primary", onclick: () => tuneThis(base, clip, tuneBtn) }, "Tune to this detection");
  return el("div", {}, el("div", { class: "abrow" },
    el("span", { class: "t2" }, "Listen as"),
    seg([["rec", "Recorded"], ["auto", "Auto"], ["envelope", "Envelope"], ["sync", "Sync AM"], ["suppressed", "No carrier"], ["sync_sb", "Sync sideband"]], "demod"),
    el("span", { class: "t2" }, "Noise reduction"), seg([[true, "On"], [false, "Off"]], "denoise"),
    el("span", { class: "t2" }, "Voice band"), seg([[true, "On"], [false, "Off"]], "band"), note),
    el("div", { style: "display:flex;gap:10px;align-items:center;margin-top:10px" }, tuneBtn,
      el("span", { class: "t3", style: "font-size:12px" }, "Finds the demodulation and filtering that make this one signal most voice-like.")),
    tuneOut);
}
async function tuneThis(base, clip, btn) {
  const out = $("#tuneOut"); btn.disabled = true; btn.textContent = "Tuning…";
  out.replaceChildren(el("div", { class: "note" }, "Trying detectors, carrier fine-tuning, voice bands and noise reduction on this clip…"));
  try {
    const q = Object.fromEntries(new URLSearchParams(base));
    const r = await fetch("/api/tune_audio", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(q) });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json(), b = res.best, pct = v => v < 0.01 ? "under 1%" : (v * 100).toFixed(v > 0.99 ? 2 : 0) + "%";
    const desc = [b.label + (b.mode && b.offset ? `, carrier ${b.offset > 0 ? "+" : ""}${Math.round(b.offset)} Hz` : ""),
      b.lo ? `voice band ${b.lo}–${b.hi} Hz` : "no band filter",
      b.denoise ? `noise reduction ${b.strength <= 0.5 ? "gentle" : b.strength >= 1.5 ? "strong" : "normal"}` : "no noise reduction"].join(" · ");
    DT.tunedUrl = `/api/render?${base}&demod=${b.demod}&mode=${b.mode}&offset=${b.offset}&band=${b.lo ? 1 : 0}` + (b.lo ? `&lo=${b.lo}&hi=${b.hi}` : "")
      + `&denoise=${b.denoise ? 1 : 0}&strength=${b.strength}&floor=${b.floor}`;
    DT.tunedApply = res.apply;
    const onlyHere = b.mode !== 0;
    out.replaceChildren(
      el("div", { class: "tunebest" }, el("strong", {}, desc)),
      el("div", { class: "note", style: "margin:2px 0 8px" }, `Voice score ${pct(res.score)} (your current settings: ${pct(res.baseline)}).`
        + (res.score < 0.05 ? " Nothing here sounds like speech to the voice detector under any setting, so this choice is close to arbitrary." : "")
        + (onlyHere ? ` It sounds best as ${b.mode > 0 ? "upper" : "lower"} sideband with its carrier fine-tuned, which only applies to this signal; "Use these settings" takes the filtering.` : "")),
      el("div", { style: "display:flex;gap:8px" },
        el("button", { class: "btn", onclick: () => { DT.ab.demod = "tuned"; document.querySelectorAll(".abseg button").forEach(x => x.classList.toggle("on", x.dataset.v === "tuned")); abLoad(base, clip); } }, "Listen"),
        el("button", { class: "btn", onclick: e => { patch({ audio: DT.tunedApply }, 50); e.target.textContent = "Saved as your playback settings"; e.target.disabled = true; } }, "Use these settings")),
      el("details", { style: "margin-top:8px" }, el("summary", { class: "t2" }, `All ${res.trials.length} trials`),
        ...res.trials.map(t => el("div", { class: "kv sc" }, el("span", { class: "t2" }, `${t.stage}: ${t.label}`),
          el("span", { class: "bar" }, el("i", { style: `width:${Math.round(t.score * 100)}%` })), el("span", { class: "num" }, (t.score * 100).toFixed(1))))));
    const seg_ = document.querySelector(".abseg");                    // add "Tuned" to the A/B row
    if (seg_ && !seg_.querySelector('[data-v="tuned"]')) seg_.append(el("button", { class: "btn", "data-v": "tuned", onclick: e => { DT.ab.demod = "tuned"; seg_.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === e.target)); abLoad(base, clip); } }, "Tuned"));
  } catch (e) { out.replaceChildren(el("div", { class: "err" }, String(e.message || e).slice(0, 160))); }
  btn.disabled = false; btn.textContent = "Tune again";
}
async function abLoad(base, clip) {
  const a = DT.ab, note = $("#abNote");
  const url = a.demod === "tuned" && DT.tunedUrl ? DT.tunedUrl
    : a.demod === "rec" && a.denoise === (S.cfg.audio.denoise !== false) && a.band === (S.cfg.audio.voice_band !== false) && clip ? clip
    : `/api/render?${base}&demod=${a.demod === "rec" ? "" : a.demod}&denoise=${a.denoise ? 1 : 0}&band=${a.band ? 1 : 0}`;
  if (note) note.textContent = "Rendering…";
  const gen = DT.abGen = (DT.abGen || 0) + 1;
  try {
    const r = await fetch(url); if (!r.ok) throw new Error(await r.text());
    const buf = await AC.decodeAudioData(await r.arrayBuffer());
    if (gen !== DT.abGen || !DT.open) return;
    if (DT.src) dtStop();
    DT.buf = buf; DT.spec = specOf(buf.getChannelData(0), buf.sampleRate); DT.off = 0;   // a new setting: from the top
    drawDetailWave(); drawDetailSpec(); dtTick();
    dtToggle();
    if (note) note.textContent = "";
  } catch (e) { if (note && gen === DT.abGen) note.textContent = String(e.message || e).slice(0, 80); }
}

// Receiver: which device (when several are connected) and a quick "is it working?" test.
function deviceRows(sdr) {
  const sel = el("select", { class: "sel" }, el("option", { value: "" }, "First one found"));
  const out = el("div", { class: "note", style: "margin:4px 0 0" });
  const load = async () => {
    sel.disabled = true;
    try {
      const devs = (await (await fetch("/api/devices")).json()).filter(d => d.kind === sdr.source);
      sel.replaceChildren(el("option", { value: "" }, devs.length ? "First one found" : "None detected"), ...devs.map(d => el("option", { value: d.id }, d.label)));
      if (sdr.device && !devs.some(d => d.id === sdr.device)) sel.append(el("option", { value: sdr.device }, `${sdr.device} (not detected now)`));
      sel.value = sdr.device || "";
    } catch (e) {}
    sel.disabled = false;
  };
  sel.addEventListener("change", () => patch({ sdr: { device: sel.value } }, 50));   // a receiver change restarts it
  const test = el("button", { class: "btn", onclick: async () => {
    test.disabled = true; out.className = "note"; out.textContent = "Testing…";
    try {
      const r = await (await fetch("/api/sdr/test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: sdr.source, device: sel.value }) })).json();
      out.className = r.ok ? "note" : "err";
      out.textContent = r.ok ? `${r.message || "Working"}: level ${r.level_dbfs} dBFS, clipping ${r.clipping_pct}% — ${r.verdict}.` : r.message;
      if (r.ok) out.style.color = "var(--ok)";
    } catch (e) { out.className = "err"; out.textContent = "Test failed."; }
    test.disabled = false;
  } }, "Test");
  if (sdr.source !== "file") load();
  return sdr.source === "file" ? el("span") : el("div", {},
    el("div", { class: "row wide" }, el("label", {}, "Device"), sel),
    el("div", { style: "display:flex;gap:6px;justify-content:flex-end;margin:4px 0 0" },
      el("button", { class: "btn", title: "Look for connected devices again", onclick: load }, "Refresh"), test), out);
}

// ---------------- help ----------------
function openHelp() {
  const c = S.cfg || {}, root = (S.root || "the ghostbox folder").replace(/\/$/, "");
  const p = (...k) => el("p", {}, ...k), b = t => el("strong", {}, t), code = t => el("code", {}, t);
  const step = (n, title, ...body) => el("div", { class: "helpstep" }, el("div", { class: "helpn" }, String(n)), el("div", {}, el("h3", {}, title), ...body));
  const close = () => { wrap.remove(); document.removeEventListener("keydown", key, true); };
  const key = e => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  const box = el("div", { class: "dlg help" },
    el("div", { class: "dlghead" }, el("strong", {}, "Getting started"), el("span", { style: "flex:1" }), el("button", { class: "btn", onclick: close }, "Close")),
    p("This listens to a whole slice of the radio spectrum at once and picks out anything that sounds like a human voice. The quickest way to use it:"),
    step(1, "Connect your receiver",
      p("Plug in your HackRF or RTL-SDR, then open ", b("Settings › Receiver"), ". Choose the ", b("Receiver type"), ". If you have more than one device, pick it under ", b("Device"), ". Click ", b("Test"),
        ": it should say it's receiving at a good level. If it says ", b("clipping"), ", lower the Gain at the top; if it says ", b("very quiet"), ", raise it or check the antenna.")),
    step(2, "Choose how much to listen to",
      p("The ", b("span"), " selector at the top sets how wide a slice you hear at once (a HackRF goes up to 20 MHz, an RTL-SDR to about 2.4 MHz). Wider covers more but needs more computer; 4 MHz is a good start. Watch ", b("CPU"),
        " at the top right: it should stay below about 70%. If it says the computer can't keep up, pick a narrower span.")),
    step(3, "Let it calibrate itself",
      p("Open the ", b("Detector"), " tab and make sure ", b("Recalibrate automatically"), " is on. After a couple of minutes on a band it studies what it heard, works out which of its checks pick out voices best there, and sets its thresholds. It does this again whenever you retune, and whenever it hasn't caught anything for a while.")),
    step(4, "Listen to what it finds",
      p("The ", b("Detections"), " tab lists everything that sounded voice-like, newest first, with a confidence. Audio plays a few seconds behind live as each detection happens; click anywhere once to switch sound on. ",
        b("Pause"), " (or the space bar) stops live playback, and the slider next to it is the volume. The bottom of the screen shows each voice's waveform and the sound's spectrum."),
      p("Click a detection for everything about it: the full transcript, playback with a spectrogram, different ways to demodulate it, and downloads. If one sounds like the real thing, ", b("Tune to this detection"), " finds the clearest settings for it.")),
    step(5, "Survey a wide range",
      p("To find where the activity is, click ", b("Map"), ", then ", b("New"), ". Enter a range (for example 0.5–30 MHz), choose ", b("1 min"), " per step, and start it. It steps across the range, calibrates on each step, and transcribes what it finds; the map fills in as it goes. With a 4 MHz span, 0.5–30 MHz is 10 steps, about 12 minutes; give it longer if your computer is slow. Zoom with the scroll wheel, click a box or a sample for details, and use ", b("Tune here"), " to listen live.")),
    step(6, "Keep what you found",
      p("Everything is under ", code(root), ":"),
      el("ul", {},
        el("li", {}, code("recordings/"), " — saved detections: audio (WAV) and details (JSON). Detections at least ", b(`${Math.round(((c.record || {}).min_save_conf ?? 0.7) * 100)}%`), " confident are saved automatically; for others, click the save icon in the list before closing the app (the rest live in memory only)."),
        el("li", {}, code("live_data/surveys/"), " — your surveys, with their clips."),
        el("li", {}, code(((c.iq_ring || {}).dir || "iq_ring") + "/"), " — the raw radio recording. It's a rolling buffer: the oldest is deleted to make room, so download the ", b("IQ WAV"), " of anything important from its detail window."),
        el("li", {}, code("live_data/presets/"), " — settings you saved.")),
      p("To keep an investigation, copy those folders somewhere safe once you're done.")));
  const wrap = el("div", { id: "helpWrap", class: "modalwrap", onclick: e => { if (e.target === wrap) close(); } }, box);
  document.body.append(wrap); document.addEventListener("keydown", key, true);
}
$("#helpBtn").addEventListener("click", openHelp);

// ---------------- phone ("field") layout ----------------
// Same app, rearranged: waterfall, a "now" panel (K2-style confidence LEDs, frequency, live
// transcript), big Pause / Mark / volume, and the detections list. Everything else lives in a menu
// that borrows the real controls (so they behave exactly as on the desktop) and in full-screen sheets.
var FIELD = { on: false };
function wantField() { const p = lsGet("gb_layout", "auto"); return p === "phone" || (p === "auto" && innerWidth <= 760); }
function applyLayout() {
  const on = wantField(); if (on === FIELD.on) return;
  FIELD.on = on; document.body.classList.toggle("field", on);
  if (!on) { closeSheet(); closeFieldMenu(); }
  if (on && !$("#leds").children.length)
    for (let i = 0; i < 12; i++) $("#leds").append(el("i", { style: `background:var(${i < 5 ? "--ok" : i < 9 ? "--mark" : "--alarm"})` }));
}
addEventListener("resize", applyLayout);
function setLayoutPref(v) { lsSet("gb_layout", v); applyLayout(); }
// the "now" panel
setInterval(() => {
  if (!FIELD.on) return;
  const voices = playingVoices();
  let d = voices[0];
  if (!d) for (const x of S.dets.values()) if (!x.ended && audible(x) && (!d || x.box.conf > d.box.conf)) d = x;
  const conf = d ? d.box.conf : 0, lit = Math.round(conf * 12);
  [...$("#leds").children].forEach((e, i) => e.classList.toggle("on", i < lit));
  $("#nowFreq").textContent = d ? fmtF((d.box.fmin + d.box.fmax) / 2) : S.paused ? "Paused" : (!AC || AC.state !== "running") ? "Tap to start audio" : "Listening…";
  $("#nowConf").textContent = d ? Math.round(conf * 100) + "%" : "";
  const log = (S.txLog || []).slice(-3), box = $("#nowText");
  const key = log.map(x => x.at).join();
  if (box.dataset.k !== key) {
    box.dataset.k = key;
    box.replaceChildren(...log.slice(0, -1).map(x => el("span", { class: "prev" }, `“${x.text}”`)),
      ...log.slice(-1).map(x => el("span", { class: x.doubt ? "doubt" : "" }, `“${x.text}”`)));
  }
}, 150);
function toast(text) { const t = $("#toast") || document.body.appendChild(el("div", { id: "toast" })); t.textContent = text; t.style.display = ""; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", 2600); }
async function markNow() {
  const t = S.offset === null ? undefined : playheadTime();
  try { const m = await (await fetch("/api/mark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ t }) })).json(); toast(`Marked ${clockTime(m.t)}`); } catch (e) { toast("Couldn't save the mark"); }
}
// min confidence, right in the panel
function syncFieldMin() { const v = Math.round(S.minConf * 100); $("#fMin").value = v; fillRange($("#fMin")); $("#fMinOut").textContent = v + "%"; }
$("#fMin").addEventListener("input", e => { setMinConf(+e.target.value / 100); syncFieldMin(); });
syncFieldMin();
// recalibrate on demand (the same as Detector > Recalibrate now)
$("#fRecal").addEventListener("click", async () => {
  const at = S.cfg.autotune, b = $("#fRecal"); b.disabled = true; b.textContent = "Calibrating…";
  const r = await (await fetch("/api/autotune", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ minutes: at.minutes, per_min: at.per_min }) })).json().catch(() => ({}));
  if (r.error) { toast(r.error); b.disabled = false; b.textContent = "Recalibrate"; }
  FIELD.recal = true;
});
// touch on the waterfall: one finger scrolls through time, two fingers zoom it; taps behave as clicks
(() => {
  let t0 = null, pinch = null;
  cv.addEventListener("touchstart", e => {
    if (e.touches.length === 2) {
      const d = Math.abs(e.touches[0].clientY - e.touches[1].clientY) || 1; pinch = { d, pps: S.pps }; t0 = null; return;
    }
    const top = topTime(); if (top === null) return;
    t0 = { y: e.touches[0].clientY, top: S.follow ? top : S.viewTop, moved: false };
  }, { passive: true });
  cv.addEventListener("touchmove", e => {
    if (pinch && e.touches.length === 2) {
      e.preventDefault();
      const d = Math.abs(e.touches[0].clientY - e.touches[1].clientY) || 1;
      S.pps = Math.max(3, Math.min(160, pinch.pps * d / pinch.d)); lsSet("gb_pps", S.pps); return;
    }
    if (!t0) return;
    const dy = e.touches[0].clientY - t0.y;
    if (!t0.moved && Math.abs(dy) < 8) return;
    e.preventDefault(); t0.moved = true; downAt = null;     // a swipe, not a tap
    const live = liveTop(); if (live === null) return;
    S.follow = false; S.viewTop = Math.min(live, t0.top + dy / S.pps); updateNav();
  }, { passive: false });
  cv.addEventListener("touchend", e => { if (e.touches.length < 2) pinch = null; if (!e.touches.length) t0 = null; });
})();
// full-screen sheets for the Detector and Settings tabs
function openSheet(tab, title) { closeFieldMenu(); document.body.classList.add("sheet"); $("#sheetTitle").textContent = title; switchTab(tab); }
function closeSheet() { if (!document.body.classList.contains("sheet")) return; document.body.classList.remove("sheet"); switchTab("dets"); }
$("#sheetClose").addEventListener("click", closeSheet);
// the menu: borrows the header's span / gain / map controls and the waterfall toolbar
const BORROW = ["#grpRate", "#grpGain", "#wfbar", "#vol"];
function closeFieldMenu() {
  const m = $("#fieldMenu"); if (!m) return;
  for (const b of FIELD.borrowed || []) b.parent.insertBefore(b.node, b.next);
  FIELD.borrowed = []; m.remove();
}
function openFieldMenu() {
  if ($("#fieldMenu")) return closeFieldMenu();
  FIELD.borrowed = BORROW.map(q => { const n = $(q); return { node: n, parent: n.parentNode, next: n.nextSibling }; });
  const [rate, gain, bar, vol] = FIELD.borrowed.map(b => b.node);
  const go = (label, fn) => el("button", { class: "btn", onclick: fn }, label);
  const box = el("div", { class: "dlg" },
    el("div", { class: "dlghead" }, el("strong", {}, "Ghostbox"), el("span", { class: "t2 num", style: "font-size:12px" }, $("#status").textContent), go("Close", closeFieldMenu)),
    el("h3", {}, "Receiver"), el("div", { class: "mrow" }, rate), el("div", { class: "mrow" }, gain),
    el("h3", {}, "Display"), el("div", { class: "mrow" }, bar),
    el("h3", {}, "Volume"), el("div", { class: "mrow" }, vol),
    el("div", { class: "mbtns" },
      go("Mark this moment", () => { closeFieldMenu(); markNow(); }),
      go("Map", () => { closeFieldMenu(); openMap(); }),
      go("Detector", () => openSheet("detector", "Detector")),
      go("Settings", () => openSheet("sdr", "Settings")),
      go("Help", () => { closeFieldMenu(); openHelp(); }),
      go("Desktop layout", () => { closeFieldMenu(); setLayoutPref("desktop"); })));
  const wrap = el("div", { id: "fieldMenu", onclick: e => { if (e.target === wrap) closeFieldMenu(); } }, box);
  document.body.append(wrap);
}
$("#menuBtn").addEventListener("click", openFieldMenu);
applyLayout();

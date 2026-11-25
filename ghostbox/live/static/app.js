// Ghostbox Live client: waterfall + detection overlays + delayed audio playback + config UI.
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
};
const fmtF = hz => hz >= 1e6 ? (hz / 1e6).toFixed(4) + " MHz" : (hz / 1e3).toFixed(1) + " kHz";
const nowS = () => performance.now() / 1000;
let LS = {}; try { LS = window.localStorage || {}; } catch (e) {}
const lsGet = (k, d) => { try { const v = LS.getItem && LS.getItem(k); return v === null || v === undefined ? d : v; } catch (e) { return d; } };
const lsSet = (k, v) => { try { LS.setItem && LS.setItem(k, v); } catch (e) {} };

// ---------------- state ----------------
const S = {
  cfg: null, checks: [], presets: {}, status: {},
  fr: 25, fmin: 0, fmax: 1, wfW: 0, T0: null, N: 0,
  offsets: [], offset: null,          // client time - stream time of newest received row
  dets: new Map(),                     // id -> {box, playing:[[a,b]], ended}
  hop: 1.0, delay: 4, pps: 25,
  view: "power", views: [], recs: [], tx: new Map(), chainTx: new Map(),
  smin: 0, smax: 255, cmap: "magma",
};
const RING_H = 3000;
const ring = document.createElement("canvas");
const rctx = ring.getContext("2d");
const cv = $("#wf"), ctx = cv.getContext("2d");

// colour maps (256-entry RGBA LUTs) and the scale (min/max) applied on top
const CMAPS = {
  magma: [[0, 4, 4, 16], [0.25, 50, 18, 92], [0.5, 150, 40, 110], [0.75, 245, 110, 60], [1, 252, 250, 180]],
  viridis: [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140], [0.75, 94, 201, 98], [1, 253, 231, 37]],
  ice: [[0, 0, 0, 8], [0.35, 10, 40, 120], [0.65, 40, 150, 220], [0.85, 160, 230, 250], [1, 255, 255, 255]],
  gray: [[0, 0, 0, 0], [1, 255, 255, 255]],
};
function makeLUT(name) {
  const stops = CMAPS[name] || CMAPS.magma;
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255; let j = 0; while (j < stops.length - 2 && x > stops[j + 1][0]) j++;
    const [a, b] = [stops[j], stops[j + 1]]; const t = (x - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) lut[i * 4 + c] = a[c + 1] + (b[c + 1] - a[c + 1]) * t;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}
let LUT = makeLUT("magma");
function scaledLUT() {                        // value 0..255 -> colour, after min/max stretch
  const out = new Uint8ClampedArray(256 * 4), span = Math.max(1, S.smax - S.smin);
  for (let v = 0; v < 256; v++) {
    const i = Math.max(0, Math.min(255, Math.round((v - S.smin) / span * 255)));
    out.set(LUT.subarray(i * 4, i * 4 + 4), v * 4);
  }
  return out;
}
let SLUT = scaledLUT();
let VALS = new Uint8Array(0);                 // raw row values, same layout as the ring canvas
function paintRow(ringY) {
  const w = ring.width, img = rctx.createImageData(w, 1);
  for (let x = 0; x < w; x++) { const v = VALS[ringY * w + x] * 4; img.data[x * 4] = SLUT[v]; img.data[x * 4 + 1] = SLUT[v + 1]; img.data[x * 4 + 2] = SLUT[v + 2]; img.data[x * 4 + 3] = 255; }
  rctx.putImageData(img, 0, ringY);
}
function repaintRing() {
  SLUT = scaledLUT();
  const w = ring.width, img = rctx.createImageData(w, RING_H);
  for (let i = 0; i < VALS.length; i++) { const v = VALS[i] * 4; img.data[i * 4] = SLUT[v]; img.data[i * 4 + 1] = SLUT[v + 1]; img.data[i * 4 + 2] = SLUT[v + 2]; img.data[i * 4 + 3] = 255; }
  rctx.putImageData(img, 0, 0);
}
function clearRing(w) {
  ring.width = Math.max(1, w); ring.height = RING_H; VALS = new Uint8Array(ring.width * RING_H);
  repaintRing();
}

function resetWaterfall(r) {
  S.fmin = r.fmin; S.fmax = r.fmax; S.fr = r.frame_rate || 25;
  S.T0 = null; S.N = 0; S.offsets = []; S.offset = null; S.dets.clear();
  clearRing(S.wfW || 1200);
  renderDetList();
}

function onRows(m) {
  const bytes = Uint8Array.from(atob(m.rows), c => c.charCodeAt(0));
  const w = m.w, n = bytes.length / w;
  if ((m.mode || "power") !== S.view) return;
  if (w !== ring.width || m.fmin !== S.fmin || m.fmax !== S.fmax) {
    S.wfW = w; S.fmin = m.fmin; S.fmax = m.fmax; clearRing(w); S.T0 = null; S.N = 0;
  }
  if (S.T0 === null) { S.T0 = m.t0; S.N = 0; }
  const k0 = Math.round((m.t0 - S.T0) / m.dt);
  if (k0 > S.N + 2 || k0 < S.N - 2) { S.N = k0; }
  for (let r = 0; r < n; r++) {
    const ringY = RING_H - 1 - ((S.N + r) % RING_H);
    VALS.set(bytes.subarray(r * w, (r + 1) * w), ringY * w);
    paintRow(ringY);
  }
  S.N += n;
}
// clock sync: status carries the live edge (stream time of the newest samples)
function syncClock(t) {
  if (typeof t !== "number") return;
  S.offsets.push(nowS() - t); if (S.offsets.length > 20) S.offsets.shift();
  S.offset = Math.min(...S.offsets);
}

// ---------------- drawing ----------------
function topTime() {
  if (S.offset === null || S.T0 === null) return null;
  const t = nowS() - S.offset - S.hop - 0.05;
  return Math.min(t, S.T0 + (S.N - 1) / S.fr);
}
const xOf = f => (f - S.fmin) / (S.fmax - S.fmin) * cv.width;

function draw() {
  requestAnimationFrame(draw);
  const W = cv.clientWidth * devicePixelRatio, H = cv.clientHeight * devicePixelRatio;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  const top = topTime();
  if (top === null) return;
  $("#banner").textContent = "";
  const dpr = devicePixelRatio, pps = S.pps * dpr;
  const rowPx = pps / S.fr;
  const topRow = (top - S.T0) * S.fr;
  const kTop = Math.floor(topRow), frac = topRow - kTop;
  const visRows = Math.min(Math.ceil(H / rowPx) + 1, RING_H, kTop + 1);
  const yTop = RING_H - 1 - (((kTop % RING_H) + RING_H) % RING_H);
  const h1 = Math.min(RING_H - yTop, visRows);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(ring, 0, yTop, ring.width, h1, 0, frac * rowPx, W, h1 * rowPx);
  if (visRows > h1) ctx.drawImage(ring, 0, 0, ring.width, visRows - h1, 0, (frac + h1) * rowPx, W, (visRows - h1) * rowPx);

  // time grid (seconds ago, relative to the top row)
  ctx.font = `${11 * dpr}px ui-monospace,Menlo,monospace`;
  ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.strokeStyle = "rgba(255,255,255,.08)";
  for (let s = 5; s * pps < H; s += 5) {
    const y = s * pps; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillText(`-${s}s`, 4 * dpr, y - 3 * dpr);
  }
  // frequency axis
  const span = S.fmax - S.fmin; const step = niceStep(span / 8);
  for (let f = Math.ceil(S.fmin / step) * step; f <= S.fmax; f += step) {
    const x = xOf(f); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillText(fmtF(f), x + 3 * dpr, 13 * dpr);
  }
  // playback line: audio at stream time t plays when it is (delay - hop) below the top
  const yPlay = (top - (nowS() - S.offset - S.delay)) * pps;
  ctx.save(); ctx.setLineDash([6 * dpr, 6 * dpr]); ctx.strokeStyle = "rgba(245,197,66,.7)"; ctx.lineWidth = dpr;
  ctx.beginPath(); ctx.moveTo(0, yPlay); ctx.lineTo(W, yPlay); ctx.stroke(); ctx.restore();
  ctx.fillStyle = "rgba(245,197,66,.85)"; ctx.fillText(`▶ playback (${S.delay.toFixed(2)} s behind live)`, W - 230 * dpr, yPlay - 4 * dpr);

  // detections
  const now = nowS();
  const chains = new Map();
  for (const d of S.dets.values()) {
    const b = d.box; const y0 = (top - b.t1) * pps, y1 = (top - b.t0) * pps;
    if (y1 < 0 || y0 > H) continue;
    const pad = Math.max(5 * dpr, (xOf(b.fmin + 3400) - xOf(b.fmin)));
    let x0 = xOf(b.fmin) - pad, x1 = xOf(b.fmax) + pad;
    const playing = d.playing.some(([a, z]) => now >= a && now <= z);
    d.isPlaying = playing;
    const hue = 50 - 50 * Math.min(1, Math.max(0, (b.conf - 0.3) / 0.6));
    ctx.save();
    if (playing) { ctx.shadowColor = `hsl(${hue},100%,60%)`; ctx.shadowBlur = 18 * dpr; ctx.fillStyle = `hsla(${hue},100%,60%,.22)`; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
    ctx.strokeStyle = `hsla(${hue},100%,${playing ? 70 : 55}%,${playing ? 1 : .85})`;
    ctx.lineWidth = (playing ? 2.5 : 1.3) * dpr;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.restore();
    if (b.path && b.path.length > 1 && b.fmax - b.fmin > 500) {
      ctx.strokeStyle = `hsla(${hue},100%,75%,.9)`; ctx.lineWidth = 1.5 * dpr; ctx.beginPath();
      b.path.forEach(([t, f], i) => { const x = xOf(f), y = (top - t) * pps; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.lineTo(xOf(b.path[b.path.length - 1][1]), y0); ctx.stroke();
    }
    ctx.fillStyle = `hsla(${hue},100%,75%,1)`;
    const ly = Math.max(12 * dpr, y0 + 11 * dpr);
    ctx.fillText(`#${b.id} ${fmtF((b.fmin + b.fmax) / 2)} ${(b.conf * 100).toFixed(0)}%`, x1 + 4 * dpr, ly);
    const tx = S.tx.get(b.id);
    if (tx && tx.text && $("#caps").checked) caption(`“${tx.text}”`, x1 + 4 * dpr, ly + 15 * dpr, tx.no_speech_prob > 0.6, playing);
    if (!chains.has(b.chain)) chains.set(b.chain, []);
    chains.get(b.chain).push(b);
  }
  // hop links: dashed line from the end of one track to the start of the next in the same chain
  ctx.save(); ctx.setLineDash([3 * dpr, 4 * dpr]); ctx.strokeStyle = "rgba(120,220,255,.8)"; ctx.lineWidth = 1.2 * dpr;
  for (const list of chains.values()) {
    list.sort((a, b) => a.t0 - b.t0);
    for (let i = 1; i < list.length; i++) {
      const a = list[i - 1], b = list[i];
      ctx.beginPath(); ctx.moveTo(xOf(a.path[a.path.length - 1][1]), (top - a.t1) * pps);
      ctx.lineTo(xOf(b.path[0][1]), (top - b.t0) * pps); ctx.stroke();
    }
  }
  ctx.restore();
  // whole-chain transcript, under the chain's newest box
  if ($("#caps").checked) for (const [cid, list] of chains) {
    const tx = S.chainTx.get(cid); if (!tx || !tx.text || list.length < 2) continue;
    const b = list[list.length - 1];
    caption(`⛓ “${tx.text}”`, xOf(b.fmax) + 8 * dpr, (top - b.t0) * pps + 14 * dpr, tx.no_speech_prob > 0.6, false, "rgba(120,220,255,1)");
  }
  // drop boxes that scrolled off
  for (const [id, d] of S.dets) if ((top - d.box.t1) * pps > H * 1.5 && d.ended) S.dets.delete(id);
}
function caption(text, x, y, doubtful, playing, color) {
  const dpr = devicePixelRatio;
  if (text.length > 90) text = text.slice(0, 88) + "…”";
  ctx.save();
  ctx.font = `italic ${(playing ? 13 : 12) * dpr}px system-ui,-apple-system,sans-serif`;
  const w = ctx.measureText(text).width;
  if (x + w > cv.width - 4 * dpr) x = Math.max(4 * dpr, cv.width - w - 8 * dpr);
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(x - 3 * dpr, y - 12 * dpr, w + 6 * dpr, 16 * dpr);
  ctx.fillStyle = doubtful ? "rgba(170,170,170,.8)" : (color || (playing ? "#fff6c8" : "#ffffff"));
  ctx.fillText(text, x, y);
  ctx.restore();
}
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x))); const m = x / p; return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p; }

cv.addEventListener("mousemove", e => {
  const r = cv.getBoundingClientRect(); const x = e.clientX - r.left, y = e.clientY - r.top;
  const f = S.fmin + x / r.width * (S.fmax - S.fmin); const tip = $("#tip");
  tip.style.display = "block"; tip.style.left = (x + 12) + "px"; tip.style.top = (y + 12) + "px";
  tip.textContent = `${fmtF(f)}  -${(y / S.pps).toFixed(1)} s`;
});
cv.addEventListener("mouseleave", () => $("#tip").style.display = "none");
cv.addEventListener("click", e => {
  const r = cv.getBoundingClientRect(), top = topTime(); if (top === null) return;
  const f = S.fmin + (e.clientX - r.left) / r.width * (S.fmax - S.fmin), t = top - (e.clientY - r.top) / S.pps;
  let best = null, bd = 1e18;
  for (const d of S.dets.values()) {
    const b = d.box; const df = Math.max(0, b.fmin - f, f - b.fmax), dt = Math.max(0, b.t0 - t, t - b.t1);
    const dist = df / (S.fmax - S.fmin) * r.width + dt * S.pps;
    if (dist < bd) { bd = dist; best = d; }
  }
  if (best && bd < 30) { switchTab("dets"); const c = document.getElementById("det" + best.box.id); if (c) c.scrollIntoView({ block: "center" }); }
});

// ---------------- audio ----------------
let AC = null, master = null;
$("#audioBtn").addEventListener("click", async () => {
  if (!AC) { AC = new (window.AudioContext || window.webkitAudioContext)(); master = AC.createGain(); master.connect(AC.destination); }
  await AC.resume(); master.gain.value = +$("#vol").value;
  $("#audioBtn").textContent = "Audio on"; $("#audioBtn").classList.remove("primary"); $("#audioBtn").classList.add("on");
});
$("#vol").addEventListener("input", e => { if (master) master.gain.value = +e.target.value; });

function onAudio(m) {
  const d = S.dets.get(m.id); if (!d || S.offset === null) return;
  const bytes = Uint8Array.from(atob(m.pcm), c => c.charCodeAt(0));
  const pcm = new Int16Array(bytes.buffer);
  const dur = pcm.length / m.rate;
  const when = m.t + S.offset + S.delay;          // client clock time to start playing
  const now = nowS();
  let skip = Math.max(0, now - when);
  if (skip >= dur) return;
  d.playing.push([Math.max(when, now), when + dur]); if (d.playing.length > 200) d.playing.shift();
  if (!AC || d.muted) return;
  const buf = AC.createBuffer(1, pcm.length, m.rate); const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = AC.createBufferSource(); src.buffer = buf; src.connect(master);
  src.start(AC.currentTime + Math.max(0, when - now), skip);
}

// ---------------- detections list ----------------
function checkLabel(id) { const c = S.checks.find(c => c.id === id); return c ? c.label : id; }
function scoresGrid(scores) {
  const g = el("div", { class: "scores" });
  for (const [k, v] of Object.entries(scores || {})) {
    g.append(el("span", { class: "muted" }, checkLabel(k)), el("div", { class: "bar" }, el("i", { style: `width:${(v * 100).toFixed(0)}%` })), el("span", { class: "mono" }, v.toFixed(2)));
  }
  return g;
}
function txBlock(tx, label) {
  if (!tx) return "";
  const doubt = !tx.text || tx.no_speech_prob > 0.6;
  return el("div", { class: "transcript" + (doubt ? " doubt" : "") },
    tx.text ? `${label ? label + " " : ""}“${tx.text}”` : `${label ? label + " " : ""}(no words)`,
    el("div", { class: "tmeta" }, `whisper ${tx.model} · ${tx.language} · no-speech ${(tx.no_speech_prob * 100).toFixed(0)}%${doubt ? " · probably not real speech" : ""}`));
}
function renderDet(d) {
  const b = d.box; const id = "det" + b.id;
  const info = b.info || {};
  const extra = [info.f0_median_hz ? `pitch ${info.f0_median_hz.toFixed(0)} Hz` : null, info.voiced_frac !== undefined ? `voiced ${(info.voiced_frac * 100).toFixed(0)}%` : null,
    b.chain !== b.id ? `hop-chain #${b.chain}` : null].filter(Boolean).join(" · ");
  const card = el("div", { class: "card" + (d.ended ? "" : " live") + (d.isPlaying ? " playing" : ""), id },
    el("h3", {}, el("span", {}, `#${b.id} · ${fmtF((b.fmin + b.fmax) / 2)}`),
      el("button", { onclick: () => { d.muted = !d.muted; renderDetList(); } }, d.muted ? "unmute" : "mute")),
    el("div", { class: "muted mono" }, `${(b.t1 - b.t0).toFixed(1)} s · conf ${(b.conf * 100).toFixed(0)}% ${d.ended ? "· ended" : "· live"}${b.fmax - b.fmin > 500 ? ` · slid ${((b.fmax - b.fmin) / 1e3).toFixed(1)} kHz` : ""}`),
    el("div", { class: "bar" }, el("i", { style: `width:${(b.conf * 100).toFixed(0)}%` })),
    extra ? el("div", { class: "muted", style: "font-size:11px" }, extra) : "",
    txBlock(S.tx.get(b.id)),
    scoresGrid(b.scores));
  return card;
}
let detTimer = null;
function renderDetList() {
  if (detTimer) return;
  detTimer = setTimeout(() => {
    detTimer = null;
    const list = $("#detList"); list.replaceChildren();
    const ds = [...S.dets.values()].filter(d => !d.ended).sort((a, b) => b.box.t0 - a.box.t0).slice(0, 30);
    $("#detsEmpty").style.display = ds.length ? "none" : "block";
    for (const d of ds) list.append(renderDet(d));
  }, 250);
}
setInterval(() => { for (const d of S.dets.values()) { const p = d.playing.some(([a, z]) => nowS() >= a && nowS() <= z); if (p !== d.wasPlaying) { d.wasPlaying = p; renderDetList(); } } }, 200);

// ---------------- config UI ----------------
let pending = {}, pendTimer = null;
function patch(obj) {
  const merge = (a, b) => { for (const [k, v] of Object.entries(b)) { if (v && typeof v === "object" && !Array.isArray(v)) { a[k] = a[k] || {}; merge(a[k], v); } else a[k] = v; } };
  merge(pending, obj);
  clearTimeout(pendTimer);
  pendTimer = setTimeout(async () => { const p = pending; pending = {}; await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) }); }, 300);
}
const setPath = (path, v) => { const o = {}; let cur = o; path.forEach((k, i) => { if (i === path.length - 1) cur[k] = v; else cur = cur[k] = {}; }); return o; };
const getPath = (obj, path) => path.reduce((o, k) => (o || {})[k], obj);

function slider(label, path, min, max, step, fmt = v => v) {
  const v = getPath(S.cfg, path);
  const out = el("output", {}, fmt(+v));
  const inp = el("input", { type: "range", min, max, step, value: v });
  inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); patch(setPath(path, +inp.value)); });
  return el("div", { class: "row" }, el("label", {}, label), inp, out);
}
function buildDetector() {
  const root = $("#detector"); root.replaceChildren();
  const c = S.cfg;
  root.append(el("h2", {}, "Thresholds & tracking"),
    slider("candidate threshold (stage 1)", ["detect", "candidate_threshold"], 0, 1, 0.01, v => v.toFixed(2)),
    slider("detection threshold (confidence)", ["detect", "threshold"], 0, 1, 0.01, v => v.toFixed(2)),
    slider("novelty strength", ["detect", "novelty"], 0, 1.5, 0.05, v => v.toFixed(2)),
    slider("novelty memory (s)", ["detect", "novelty_tau_s"], 2, 120, 1, v => v + " s"),
    slider("prominence strength", ["detect", "prominence"], 0, 1.5, 0.05, v => v.toFixed(2)),
    slider("prominence width (kHz)", ["detect", "prominence_hz"], 5000, 200000, 5000, v => (v / 1e3) + " kHz"),
    slider("min duration", ["detect", "min_duration_s"], 0.2, 5, 0.1, v => v.toFixed(1) + " s"),
    slider("candidates per frame", ["detect", "max_candidates"], 1, 30, 1),
    slider("max simultaneous tracks", ["detect", "max_tracks"], 1, 40, 1),
    slider("max slide rate (kHz/s)", ["detect", "max_slide_hz_s"], 0, 500000, 5000, v => (v / 1e3) + " kHz/s"),
    slider("track gap tolerance", ["detect", "track_gap_s"], 0.05, 2, 0.05, v => v.toFixed(2) + " s"),
    slider("hop link window", ["detect", "link_hops_s"], 0, 2, 0.05, v => v.toFixed(2) + " s"),
    slider("min separation (kHz)", ["detect", "min_sep_hz"], 1000, 50000, 500, v => (v / 1e3) + " kHz"),
    slider("audio bandwidth (Hz)", ["audio", "bandwidth_hz"], 1500, 7500, 100, v => v + " Hz"),
  );
  const w = c.whisper;
  const wTog = el("input", { type: "checkbox" }); wTog.checked = w.enabled;
  wTog.addEventListener("change", () => patch({ whisper: { enabled: wTog.checked } }));
  const cTog = el("input", { type: "checkbox" }); cTog.checked = w.chains;
  cTog.addEventListener("change", () => patch({ whisper: { chains: cTog.checked } }));
  const sel = (opts, val, key) => { const e = el("select", {}, ...opts.map(o => el("option", { value: o }, o))); e.value = val; e.addEventListener("change", () => patch({ whisper: { [key]: e.value } })); return e; };
  const ws = S.status.whisper || {};
  root.append(el("h2", {}, "Transcription (Whisper, local)"),
    el("div", { class: "card check" },
      el("h3", {}, el("label", {}, wTog, "Transcribe detections"), el("span", { class: "stage" }, ws.model ? `loaded: ${ws.model}` : "loads on first use")),
      el("div", { class: "desc" }, "Runs Whisper on each finished detection. Whisper invents words from noise, so each transcript shows its own no-speech probability."),
      el("div", { class: "row" }, el("label", {}, "model"), sel(["tiny", "base", "small", "medium"], w.model, "model")),
      el("div", { class: "row" }, el("label", {}, "language"), sel(["auto", "en", "es", "fr", "de", "it", "pt", "ja", "zh"], w.language, "language")),
      slider("only if confidence reached", ["whisper", "min_confidence"], 0, 1, 0.01, v => v.toFixed(2)),
      el("div", { class: "row" }, el("label", {}, cTog, " also transcribe whole hop chains")),
      ws.error ? el("div", { style: "color:var(--bad)" }, ws.error) : ""));
  for (const stage of [1, 2]) {
    root.append(el("h2", {}, stage === 1 ? "Stage 1 — whole band, every bin (IQ domain)" : "Stage 2 — per candidate (demodulated)"));
    for (const ch of S.checks.filter(x => x.stage === stage)) {
      const cc = c.checks[ch.id];
      const tog = el("input", { type: "checkbox" }); tog.checked = cc.enabled;
      tog.addEventListener("change", () => patch(setPath(["checks", ch.id, "enabled"], tog.checked)));
      const card = el("div", { class: "card check" },
        el("h3", {}, el("label", {}, tog, ch.label), el("span", { class: "stage" }, `stage ${stage}`)),
        el("div", { class: "desc" }, ch.desc),
        slider("weight", ["checks", ch.id, "weight"], 0, 3, 0.05, v => v.toFixed(2)));
      for (const p of ch.params) card.append(slider(p.label, ["checks", ch.id, p.key], p.min, p.max, p.step));
      if (ch.id === "vad" && S.status.vad === false) card.append(el("div", { class: "muted" }, "Silero VAD not available on the server."));
      root.append(card);
    }
  }
  root.append(el("div", { class: "row" }, el("button", { onclick: async () => { if (confirm("Reset all settings to defaults?")) await fetch("/api/reset", { method: "POST" }); } }, "Reset all to defaults")));
}

async function buildSDR() {
  const root = $("#sdr"); root.replaceChildren();
  const c = S.cfg, sdr = c.sdr;
  const presetSel = el("select", {}, el("option", { value: "" }, "— choose a preset —"), ...Object.entries(S.presets).map(([k, v]) => el("option", { value: k }, v)));
  root.append(el("h2", {}, "Presets"), el("div", { class: "row" }, presetSel,
    el("button", { class: "primary", onclick: async () => { if (presetSel.value) await fetch("/api/preset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: presetSel.value }) }); } }, "Apply preset")));

  const form = {};
  const field = (label, key, input) => { form[key] = input; return el("div", { class: "row" }, el("label", {}, label), input); };
  const num = (v, step = "any") => el("input", { type: "number", value: v ?? "", step });
  const src = el("select", {}, ...["hackrf", "rtlsdr", "file"].map(s => el("option", { value: s }, s)));
  src.value = sdr.source;
  const files = await (await fetch("/api/captures")).json();
  const fileSel = el("select", {}, ...files.map(f => el("option", { value: f }, f))); fileSel.value = sdr.file.path;
  const amp = el("input", { type: "checkbox" }); amp.checked = !!sdr.hackrf.amp;
  root.append(el("h2", {}, "Receiver"),
    field("source", "source", src),
    field("center frequency (MHz)", "center", num(sdr.center_hz / 1e6)),
    field("sample rate (MS/s)", "rate", num(sdr.sample_rate / 1e6)),
    field("analyse from (MHz, blank = all)", "fmin", num(sdr.fmin_hz === null ? "" : sdr.fmin_hz / 1e6)),
    field("analyse to (MHz, blank = all)", "fmax", num(sdr.fmax_hz === null ? "" : sdr.fmax_hz / 1e6)),
    el("h2", {}, "HackRF"),
    field("RF amp (+14 dB)", "amp", amp),
    field("LNA gain (0-40, step 8)", "lna", num(sdr.hackrf.lna, 8)),
    field("VGA gain (0-62, step 2)", "vga", num(sdr.hackrf.vga, 2)),
    el("h2", {}, "RTL-SDR"),
    field("gain (dB, 0 = AGC)", "rgain", num(sdr.rtlsdr.gain, 0.1)),
    field("ppm correction", "ppm", num(sdr.rtlsdr.ppm, 1)),
    el("h2", {}, "File replay"),
    field("capture file", "file", fileSel),
    el("h2", {}, "Analysis"),
    field("bin width (Hz)", "bin", el("select", {}, ...[62.5, 125, 250, 500].map(b => el("option", { value: b }, b)))),
    field("analysis window (s)", "win", num(c.analysis.window_s, 0.5)),
    field("settle (s behind live)", "settle", num(c.analysis.settle_s, 0.1)),
  );
  form.bin.value = c.analysis.bin_hz;
  root.append(el("div", { class: "row" }, el("button", {
    class: "primary", onclick: async () => {
      const opt = v => v.value === "" ? null : +v.value * 1e6;
      const p = {
        sdr: {
          source: form.source.value, center_hz: +form.center.value * 1e6, sample_rate: +form.rate.value * 1e6,
          fmin_hz: opt(form.fmin), fmax_hz: opt(form.fmax),
          hackrf: { amp: form.amp.checked, lna: +form.lna.value, vga: +form.vga.value },
          rtlsdr: { gain: +form.rgain.value, ppm: +form.ppm.value }, file: { path: form.file.value },
        },
        analysis: { bin_hz: +form.bin.value, window_s: +form.win.value, settle_s: +form.settle.value },
      };
      await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p) });
      await fetch("/api/restart", { method: "POST" });
    }
  }, "Apply & restart receiver")),
    el("div", { class: "muted", style: "font-size:11.5px" }, "HackRF tunes ~1 MHz–6 GHz at 2–20 MS/s; RTL-SDR ~25 MHz–1.75 GHz at up to 2.4 MS/s. Sample rate / bin width must divide evenly into 25 frames per second. Wider bands cost more CPU (watch “load”)."));
}

async function loadRecs() {
  S.recs = await (await fetch("/api/recordings")).json();
  renderRecs();
}
let recTimer = null;
function renderRecs() {
  if (recTimer) return;
  recTimer = setTimeout(() => {
    recTimer = null;
    const list = $("#recList"); list.replaceChildren(); list.classList.remove("muted");
    $("#recCount").textContent = S.recs.length ? `(${S.recs.length})` : "";
    if (!S.recs.length) { list.append(el("div", { class: "muted" }, "No recordings yet.")); return; }
    for (const m of S.recs.slice(0, 150)) list.append(recCard(m));
  }, 300);
}
function recCard(m) {
  const f = m.freq_hz || {};
  return el("div", { class: "card" },
    el("h3", {}, el("span", {}, `#${m.id}${m.segment ? "." + m.segment : ""} · ${fmtF(f.mean || 0)}`, m.demo ? el("span", { class: "stage", title: m.demo, style: "margin-left:6px" }, "demo") : ""), el("span", { class: "mono muted" }, (m.utc_start || "").replace("T", " ").slice(0, 19))),
    el("div", { class: "muted mono" }, `${m.duration_s}s · conf max ${(m.confidence.max * 100).toFixed(0)}% · final ${(m.confidence.final * 100).toFixed(0)}%${f.max - f.min > 500 ? ` · slid ${((f.max - f.min) / 1e3).toFixed(1)} kHz` : ""}${m.chain !== m.id ? ` · hop-chain #${m.chain}` : ""}`),
    txBlock(m.transcript), txBlock(m.chain_transcript, "⛓ whole hop chain:"),
    el("audio", { controls: "", preload: "none", src: "/rec/" + m.files.wav }),
    el("details", {}, el("summary", {}, "scores & files"),
      el("div", { class: "row" }, el("a", { href: "/rec/" + m.files.wav, download: "" }, "wav"), " ",
        el("a", { href: "/rec/" + m.files.wav.replace(".wav", ".json"), target: "_blank" }, "metadata"), " ",
        m.files.iq ? el("a", { href: "/rec/" + m.files.iq, download: "" }, "IQ (16 kHz complex .npy)") : ""),
      scoresGrid(m.scores)));
}
function onTranscript(m) {
  const res = { text: m.text, no_speech_prob: m.no_speech_prob, avg_logprob: m.avg_logprob, language: m.language, model: m.model };
  if (m.kind === "track") {
    S.tx.set(m.id, res);
    const r = S.recs.find(r => r.files && r.files.wav === m.wav); if (r) r.transcript = res;
  } else {
    S.chainTx.set(m.chain, res);
    for (const r of S.recs) if ((m.members || []).includes(r.id)) r.chain_transcript = res;
  }
  renderRecs(); renderDetList();
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  if (name === "sdr") buildSDR();
}
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// header controls
S.delay = +lsGet("gb_delay", 4); $("#delay").value = S.delay; $("#delayOut").textContent = S.delay.toFixed(2) + " s";
$("#delay").addEventListener("input", e => { S.delay = +e.target.value; $("#delayOut").textContent = S.delay.toFixed(2) + " s"; lsSet("gb_delay", S.delay); });
S.pps = +lsGet("gb_pps", 25); $("#pps").value = S.pps;
$("#pps").addEventListener("input", e => { S.pps = +e.target.value; lsSet("gb_pps", S.pps); });
function loadScale() {
  const sc = JSON.parse(lsGet("gb_scale_" + S.view, "null") || "null") || { min: 0, max: 255 };
  S.smin = sc.min; S.smax = sc.max; $("#smin").value = S.smin; $("#smax").value = S.smax;
}
function saveScale() { lsSet("gb_scale_" + S.view, JSON.stringify({ min: S.smin, max: S.smax })); }
$("#smin").addEventListener("input", e => { S.smin = Math.min(+e.target.value, S.smax - 1); saveScale(); repaintRing(); });
$("#smax").addEventListener("input", e => { S.smax = Math.max(+e.target.value, S.smin + 1); saveScale(); repaintRing(); });
$("#sreset").addEventListener("click", () => {          // auto-scale to the 2nd..99.8th percentile of what's on screen
  const v = Array.from(VALS.filter((x, i) => i % 7 === 0)).sort((a, b) => a - b);
  if (!v.length) return;
  S.smin = v[Math.floor(v.length * 0.02)]; S.smax = Math.max(S.smin + 1, v[Math.floor(v.length * 0.998)]);
  $("#smin").value = S.smin; $("#smax").value = S.smax; saveScale(); repaintRing();
});
S.cmap = lsGet("gb_cmap", "magma"); $("#cmap").value = S.cmap; LUT = makeLUT(S.cmap);
$("#cmap").addEventListener("change", e => { S.cmap = e.target.value; lsSet("gb_cmap", S.cmap); LUT = makeLUT(S.cmap); repaintRing(); });
$("#caps").checked = lsGet("gb_caps", "1") === "1";
$("#caps").addEventListener("change", e => lsSet("gb_caps", e.target.checked ? "1" : "0"));
let WS = null;
function setView(mode) {
  S.view = mode; lsSet("gb_view", mode); $("#view").value = mode;
  S.T0 = null; S.N = 0; clearRing(ring.width); loadScale(); repaintRing();
  $("#viewNote").textContent = mode === "power" ? "" : "detector map, 0–1 scale · runs ~1.5 s behind live (analysis needs a little future)";
  if (WS && WS.readyState === 1) WS.send(JSON.stringify({ type: "view", mode }));
}
$("#view").addEventListener("change", e => setView(e.target.value));

function setStatus(s) {
  S.status = s;
  syncClock(s.t);
  const st = $("#st");
  const ok = s.state === "running";
  st.className = "pill " + (ok ? "ok" : s.state === "error" || s.state === "no data" ? "bad" : "warn");
  st.textContent = ok ? `● ${s.source} · ${fmtF(s.fc)} · ${(s.fs / 1e6).toFixed(2)} MS/s` : `● ${s.state}${s.error ? ": " + s.error : ""}`;
  const st2 = $("#st2");
  if (ok) {
    st2.className = "pill " + (s.load > 0.9 ? "bad" : s.load > 0.6 ? "warn" : "");
    st2.textContent = `load ${(s.load * 100).toFixed(0)}% · tracks ${s.tracks}${s.dropped ? ` · dropped ${s.dropped}` : ""}${s.error ? " · " + s.error : ""}`;
  } else st2.textContent = "";
}

// ---------------- websocket ----------------
function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  WS = ws;
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello":
        S.cfg = m.config; S.checks = m.checks; S.presets = m.presets; S.hop = m.config.analysis.hop_s; S.views = m.views;
        $("#view").replaceChildren(...m.views.map(v => el("option", { value: v.id }, v.label)));
        resetWaterfall(m.reset); setStatus(m.status || {});
        { const v = lsGet("gb_view", "power"); if (v !== "power" && m.views.some(x => x.id === v)) setView(v); else { S.view = "power"; $("#view").value = "power"; loadScale(); } }
        loadRecs();
        for (const b of m.detections) S.dets.set(b.id, { box: b, playing: [], ended: true });
        buildDetector(); renderDetList(); break;
      case "config": S.cfg = m.config; S.hop = m.config.analysis.hop_s; if (!$("#detector").contains(document.activeElement)) buildDetector(); break;
      case "reset": resetWaterfall(m); break;
      case "status": setStatus(m); break;
      case "wf": onRows(m); break;
      case "det": S.dets.set(m.box.id, { box: m.box, playing: [], ended: false }); renderDetList(); break;
      case "det_update": { const d = S.dets.get(m.box.id); if (d) { d.box = m.box; d.ended = false; renderDetList(); } else S.dets.set(m.box.id, { box: m.box, playing: [], ended: false }); break; }
      case "det_end": { const d = S.dets.get(m.box.id); if (d) { d.box = m.box; d.ended = true; renderDetList(); } break; }
      case "audio": onAudio(m); break;
      case "recording": S.recs.unshift(m.meta); renderRecs(); break;
      case "transcript": onTranscript(m); break;
    }
  };
  ws.onclose = () => { setStatus({ state: "disconnected" }); setTimeout(connect, 1500); };
}
$("#banner").textContent = "waiting for samples…";
connect();
requestAnimationFrame(draw);

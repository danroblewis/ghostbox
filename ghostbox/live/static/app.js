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
};
const RING_H = 3000;
const ring = document.createElement("canvas");
const rctx = ring.getContext("2d");
const cv = $("#wf"), ctx = cv.getContext("2d");

// magma-ish colormap LUT
const LUT = (() => {
  const stops = [[0, 4, 4, 16], [0.25, 50, 18, 92], [0.5, 150, 40, 110], [0.75, 245, 110, 60], [1, 252, 250, 180]];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255; let j = 0; while (j < stops.length - 2 && x > stops[j + 1][0]) j++;
    const [a, b] = [stops[j], stops[j + 1]]; const t = (x - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) lut[i * 4 + c] = a[c + 1] + (b[c + 1] - a[c + 1]) * t;
    lut[i * 4 + 3] = 255;
  }
  return lut;
})();

function resetWaterfall(r) {
  S.fmin = r.fmin; S.fmax = r.fmax; S.fr = r.frame_rate || 25;
  S.T0 = null; S.N = 0; S.offsets = []; S.offset = null; S.dets.clear();
  ring.width = Math.max(1, S.wfW || 1200); ring.height = RING_H;
  rctx.fillStyle = "#000"; rctx.fillRect(0, 0, ring.width, ring.height);
  renderDetList();
}

function onRows(m) {
  const bytes = Uint8Array.from(atob(m.rows), c => c.charCodeAt(0));
  const w = m.w, n = bytes.length / w;
  if (w !== ring.width || m.fmin !== S.fmin || m.fmax !== S.fmax) {
    S.wfW = w; S.fmin = m.fmin; S.fmax = m.fmax; ring.width = w; ring.height = RING_H;
    rctx.fillStyle = "#000"; rctx.fillRect(0, 0, w, RING_H); S.T0 = null; S.N = 0;
  }
  if (S.T0 === null) { S.T0 = m.t0; S.N = 0; }
  const k0 = Math.round((m.t0 - S.T0) / m.dt);
  if (k0 > S.N + 2 || k0 < S.N - 2) { S.N = k0; }
  const img = rctx.createImageData(w, 1);
  for (let r = 0; r < n; r++) {
    for (let x = 0; x < w; x++) { const v = bytes[r * w + x] * 4; img.data.set(LUT.subarray(v, v + 4), x * 4); }
    const k = S.N + r;
    rctx.putImageData(img, 0, RING_H - 1 - (k % RING_H));
  }
  S.N += n;
  const newest = S.T0 + S.N * m.dt;
  S.offsets.push(nowS() - newest); if (S.offsets.length > 20) S.offsets.shift();
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
  const yPlay = (S.delay - S.hop) * pps;
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
    ctx.fillText(`#${b.id} ${fmtF((b.fmin + b.fmax) / 2)} ${(b.conf * 100).toFixed(0)}%`, x1 + 4 * dpr, Math.max(12 * dpr, y0 + 11 * dpr));
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
  // drop boxes that scrolled off
  for (const [id, d] of S.dets) if ((top - d.box.t1) * pps > H * 1.5 && d.ended) S.dets.delete(id);
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
    scoresGrid(b.scores));
  return card;
}
let detTimer = null;
function renderDetList() {
  if (detTimer) return;
  detTimer = setTimeout(() => {
    detTimer = null;
    const list = $("#detList"); list.replaceChildren();
    const ds = [...S.dets.values()].sort((a, b) => b.box.t0 - a.box.t0).slice(0, 60);
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
  const list = $("#recList");
  const recs = await (await fetch("/api/recordings")).json();
  list.replaceChildren(); list.classList.remove("muted");
  if (!recs.length) { list.append(el("div", { class: "muted" }, "No recordings yet.")); return; }
  for (const m of recs) list.append(recCard(m));
}
function recCard(m) {
  const f = m.freq_hz || {};
  return el("div", { class: "card" },
    el("h3", {}, el("span", {}, `#${m.id}${m.segment ? "." + m.segment : ""} · ${fmtF(f.mean || 0)}`, m.demo ? el("span", { class: "stage", title: m.demo, style: "margin-left:6px" }, "demo") : ""), el("span", { class: "mono muted" }, (m.utc_start || "").replace("T", " ").slice(0, 19))),
    el("div", { class: "muted mono" }, `${m.duration_s}s · conf max ${(m.confidence.max * 100).toFixed(0)}% · final ${(m.confidence.final * 100).toFixed(0)}%${f.max - f.min > 500 ? ` · slid ${((f.max - f.min) / 1e3).toFixed(1)} kHz` : ""}${m.chain !== m.id ? ` · hop-chain #${m.chain}` : ""}`),
    el("audio", { controls: "", preload: "none", src: "/rec/" + m.files.wav }),
    el("div", { class: "row" }, el("a", { href: "/rec/" + m.files.wav, download: "" }, "wav"), " ",
      el("a", { href: "/rec/" + m.files.wav.replace(".wav", ".json"), target: "_blank" }, "metadata"), " ",
      m.files.iq ? el("a", { href: "/rec/" + m.files.iq, download: "" }, "IQ (16 kHz complex .npy)") : ""),
    scoresGrid(m.scores));
}
async function loadLib() {
  const list = $("#libList"); const items = await (await fetch("/api/library")).json();
  list.replaceChildren(); list.classList.remove("muted");
  for (const it of items) list.append(el("div", { class: "card" }, el("div", { class: "mono", style: "color:var(--accent)" }, it.name),
    it.note ? el("div", { class: "muted", style: "font-size:12px" }, it.note) : "",
    it.kind === "wav" ? el("audio", { controls: "", preload: "none", src: "/lib/" + it.name }) : el("a", { href: "/lib/" + it.name, target: "_blank" }, el("img", { src: "/lib/" + it.name, loading: "lazy" }))));
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === name));
  if (name === "recs") loadRecs(); if (name === "lib") loadLib(); if (name === "sdr") buildSDR();
}
document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// header controls
S.delay = +lsGet("gb_delay", 4); $("#delay").value = S.delay; $("#delayOut").textContent = S.delay.toFixed(2) + " s";
$("#delay").addEventListener("input", e => { S.delay = +e.target.value; $("#delayOut").textContent = S.delay.toFixed(2) + " s"; lsSet("gb_delay", S.delay); });
S.pps = +lsGet("gb_pps", 25); $("#pps").value = S.pps;
$("#pps").addEventListener("input", e => { S.pps = +e.target.value; lsSet("gb_pps", S.pps); });
$("#dyn").addEventListener("change", e => patch({ display: { dyn_range_db: +e.target.value } }));

function setStatus(s) {
  S.status = s;
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
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello":
        S.cfg = m.config; S.checks = m.checks; S.presets = m.presets; S.hop = m.config.analysis.hop_s;
        $("#dyn").value = m.config.display.dyn_range_db;
        resetWaterfall(m.reset); setStatus(m.status || {});
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
      case "recording": if ($("#recs").classList.contains("active")) $("#recList").prepend(recCard(m.meta)); break;
    }
  };
  ws.onclose = () => { setStatus({ state: "disconnected" }); setTimeout(connect, 1500); };
}
$("#banner").textContent = "waiting for samples…";
connect();
requestAnimationFrame(draw);

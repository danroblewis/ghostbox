"""Live detection engine.

  source -> raw IQ ring -> power spectrogram (frames) -> waterfall rows to clients
                         -> every hop: stage 1 over the last window, on the newest
                            *settled* frames (settle_s behind live, so zero-phase
                            filters and windows have future context)
                         -> novelty / prominence -> per-frame candidates -> tracker
                         -> per track: AM demod of exactly that path from raw IQ
                            -> stage 2 checks -> confidence
                         -> detections: events + audio chunks (stream timestamps)
                            + recordings (WAV + JSON metadata + narrowband IQ)

Times are "stream seconds" (samples / fs since the source started). Clients play
audio at stream time + delay, which keeps every sound locked to its place on the
waterfall.
"""
import base64
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, firwin, resample_poly, sosfilt, sosfilt_zi

from ..maps import power_frames
from . import stage1, stage2
from .config import CHECKS
from .sources import make_source

AUDIO_FS = 16000
_LP = {}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def demod(x, fs, f_offset, bw_hz):
    """Mix x (complex, at fs) down by f_offset (scalar or per-sample Hz), decimate to 16 kHz.
    Returns (narrowband complex IQ at 16 kHz, AM envelope at 16 kHz)."""
    if np.ndim(f_offset) == 0:
        ph = -2 * np.pi * float(f_offset) * np.arange(len(x)) / fs
    else:
        ph = -2 * np.pi * np.cumsum(f_offset, dtype=np.float64) / fs
    y = x * np.exp(1j * ph).astype(np.complex64)
    d1 = max(1, int(fs // 160000))
    n = len(y) // d1 * d1
    y = y[:n].reshape(-1, d1).mean(axis=1)                      # CIC-like first stage
    r1 = fs / d1
    fr = Fraction(AUDIO_FS / r1).limit_denominator(1000)
    y = resample_poly(y, fr.numerator, fr.denominator).astype(np.complex64)
    key = round(bw_hz)
    if key not in _LP:
        _LP[key] = firwin(129, bw_hz, fs=AUDIO_FS).astype(np.float32)
    y = np.convolve(y, _LP[key], mode="same").astype(np.complex64)
    return y, np.abs(y).astype(np.float32)


class Track:
    _next = 1

    def __init__(self, t, f, score, chain):
        self.id = Track._next
        Track._next += 1
        self.chain = chain or self.id
        self.t0 = t
        self.t_last = t
        self.path = [(t, f)]
        self.frames = []                 # (t, f, combined, {check: score})
        self.add(t, f, score, {})
        self.audio_until = None          # stream time up to which audio has been demodulated
        self.audio, self.iq = [], []     # [(t_start, array)]
        self.emitted_until = None
        self.detected = False
        self.conf = 0.0
        self.conf_hist = []
        self.scores, self.info = {}, {}
        self.hp_zi = None
        self.agc = None
        self.seg_start = t               # recording segment start
        self.seg_index = 0

    def add(self, t, f, score, checks):
        self.t_last = t
        if not self.path or abs(self.path[-1][1] - f) > 1 or t - self.path[-1][0] > 0.2:
            self.path.append((t, f))
        self.frames.append((t, f, score, checks))
        if len(self.frames) > 2000:
            self.frames = self.frames[-2000:]

    def freq_at(self, t):
        ts = np.array([p[0] for p in self.frames])
        fs = np.array([p[1] for p in self.frames])
        idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
        return fs[idx]

    def box(self, fc):
        fr = [p[1] for p in self.path]
        return {"id": self.id, "chain": self.chain, "t0": self.t0, "t1": self.t_last,
                "fmin": fc + min(fr), "fmax": fc + max(fr),
                "path": [[round(t, 3), fc + f] for t, f in self.path[-200:]],
                "conf": round(self.conf, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "info": self.info}


class Engine:
    def __init__(self, config, emit, root):
        self.config = config
        self.emit = emit
        self.root = Path(root)
        self._restart = threading.Event()
        self._stop = threading.Event()
        self.status = {"state": "starting"}
        self.tracks, self.ended = [], []
        self.wf_hist, self.recent_boxes = [], []   # replayed to newly connected clients
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------- control ----------------
    def request_restart(self):
        self._restart.set()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        """Detections still on screen: live ones plus those ended in the last ~60 s."""
        with self.lock:
            fc = getattr(self, "fc", 0)
            return self.recent_boxes + [t.box(fc) for t in self.tracks if t.detected]

    # ---------------- main loop ----------------
    def _reset(self, cfg):
        sdr, an = cfg["sdr"], cfg["analysis"]
        self.fs, self.fc = float(sdr["sample_rate"]), float(sdr["center_hz"])
        self.fr = int(an["frame_rate"])
        self.frame_len = int(self.fs // self.fr)
        self.nfft = int(round(self.fs / an["bin_hz"]))
        self.df = self.fs / self.nfft
        f = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1 / self.fs))
        keep = np.ones(self.nfft, bool)
        if sdr.get("fmin_hz") is not None:
            keep &= self.fc + f >= float(sdr["fmin_hz"])
        if sdr.get("fmax_hz") is not None:
            keep &= self.fc + f <= float(sdr["fmax_hz"])
        keep &= np.abs(f) > 2000                       # DC / LO leakage
        self.keep = keep
        self.f = f[keep].astype(np.float32)
        self.f_lo, self.f_hi = float(self.fc + f[keep][0]), float(self.fc + f[keep][-1])
        self.n_samples = 0
        self.raw_keep = int(self.fs * (an["window_s"] + 4))
        self.raw = np.zeros(self.raw_keep, np.complex64)
        self.spec, self.spec_t = [], []
        self.baseline = None
        self.wf_lo = None
        self.analyzed_until = 0.0
        self.utc0 = time.time()
        with self.lock:
            for t in self.tracks:
                if t.detected:
                    self._finish(t, cfg, final=True)
            self.tracks, self.ended = [], []
        self.wf_hist, self.recent_boxes = [], []
        self.emit({"type": "reset", "fc": self.fc, "fs": self.fs, "fmin": self.f_lo, "fmax": self.f_hi,
                   "frame_rate": self.fr})

    def _run(self):
        src = None
        while not self._stop.is_set():
            try:
                cfg = self.config.get()
                if src is None or self._restart.is_set():
                    self._restart.clear()
                    if src:
                        src.stop()
                    self.status = {"state": "starting", "source": cfg["sdr"]["source"]}
                    self._reset(cfg)
                    src = make_source(cfg["sdr"])
                    misses = 0
                n = int(self.fs * cfg["analysis"]["hop_s"]) // self.frame_len * self.frame_len
                x = src.read(n, timeout=3.0)
                if x is None:
                    misses += 1
                    self.status = {"state": "no data", "source": cfg["sdr"]["source"],
                                   "error": getattr(src, "error", "") or "device produced no samples"}
                    self.emit({"type": "status", **self.status})
                    if misses >= 2:
                        src.stop()
                        src = None
                        time.sleep(2)
                    continue
                misses = 0
                t_proc = time.time()
                self._ingest(x, cfg)
                self._analyze(cfg)
                load = (time.time() - t_proc) / (len(x) / self.fs)
                self.status = {"state": "running", "source": cfg["sdr"]["source"], "fs": self.fs,
                               "fc": self.fc, "t": self.n_samples / self.fs, "load": round(load, 2),
                               "dropped": getattr(src, "dropped", 0), "error": getattr(src, "error", ""),
                               "tracks": len(self.tracks), "vad": stage2.vad_available()}
                self.emit({"type": "status", **self.status})
            except Exception as e:
                self.status = {"state": "error", "error": f"{type(e).__name__}: {e}"}
                self.emit({"type": "status", **self.status})
                traceback.print_exc()
                time.sleep(1)
        if src:
            src.stop()

    # ---------------- ingest: raw ring + spectrogram + waterfall ----------------
    def _ingest(self, x, cfg):
        n = len(x)
        i0 = self.n_samples % self.raw_keep
        first = min(n, self.raw_keep - i0)
        self.raw[i0:i0 + first] = x[:first]
        self.raw[: n - first] = x[first:]
        t_start = self.n_samples / self.fs
        self.n_samples += n
        x = x - x.mean()
        _, _, P = power_frames(x, self.fs, self.nfft, self.fr)
        P = P[:, self.keep]
        ts = t_start + (np.arange(len(P)) + 0.5) / self.fr
        self.spec.append(P)
        self.spec_t.append(ts)
        nkeep = int(cfg["analysis"]["window_s"] * self.fr)
        while sum(len(p) for p in self.spec) - len(self.spec[0]) >= nkeep:
            self.spec.pop(0)
            self.spec_t.pop(0)
        self._waterfall(P, ts, cfg)

    def _waterfall(self, P, ts, cfg):
        width = int(cfg["display"]["width"])
        K = P.shape[1]
        pool = max(1, -(-K // width))
        pad = (-K) % pool
        Pp = np.pad(P, ((0, 0), (0, pad)), constant_values=0).reshape(len(P), -1, pool).max(axis=2)
        db = 10 * np.log10(Pp + 1e-12)
        lo = np.percentile(db, 5)
        self.wf_lo = lo if self.wf_lo is None else 0.9 * self.wf_lo + 0.1 * lo
        q = np.clip((db - self.wf_lo) / cfg["display"]["dyn_range_db"] * 255, 0, 255).astype(np.uint8)
        msg = {"type": "wf", "t0": float(ts[0]), "dt": 1.0 / self.fr, "w": q.shape[1],
               "fmin": self.f_lo, "fmax": self.f_hi, "rows": _b64(q)}
        self.wf_hist = (self.wf_hist + [msg])[-int(30 * self.fr / len(q)):]
        self.emit(msg)

    def _raw_span(self, s0, s1):
        """Absolute sample range [s0, s1) from the ring (must still be held)."""
        s0 = max(s0, self.n_samples - self.raw_keep)
        idx = np.arange(s0, s1) % self.raw_keep
        return self.raw[idx], s0

    # ---------------- analysis ----------------
    def _analyze(self, cfg):
        an, det = cfg["analysis"], cfg["detect"]
        P = np.concatenate(self.spec)
        T = np.concatenate(self.spec_t)
        settle = int(an["settle_s"] * self.fr)
        need = int(an["window_s"] * self.fr)
        if len(P) < max(need // 2, settle + 10):
            return
        new = T > self.analyzed_until
        idx = np.flatnonzero(new[: len(T) - settle])
        if len(idx) == 0:
            return
        sl = slice(idx[0], idx[-1] + 1)
        self.analyzed_until = float(T[idx[-1]])
        scores, comb = stage1.run(P, self.df, self.fr, cfg, sl)
        ts = T[sl]

        # novelty: subtract a slow per-bin baseline (always-on emitters cancel)
        alpha = float(det["novelty"])
        if self.baseline is None or self.baseline.shape != comb.shape[1:]:
            self.baseline = comb[0].copy()
        k = 1.0 / max(1.0, det["novelty_tau_s"] * self.fr)
        ev = np.empty_like(comb)
        for i in range(len(comb)):
            ev[i] = comb[i] - alpha * self.baseline
            self.baseline += k * (comb[i] - self.baseline)
        beta = float(det["prominence"])
        if beta > 0:
            w = int(2 * det["prominence_hz"] / self.df) | 1
            ev -= beta * uniform_filter1d(ev, size=w, axis=1, mode="nearest")
        ev = np.clip(ev, 0, None)

        with self.lock:
            for i, t in enumerate(ts):
                cands = self._candidates(ev[i], det)
                self._track_step(t, cands, ev[i], comb[i], {c: s[i] for c, s in scores.items()}, det)
            self._tracks_audio_and_stage2(cfg, float(ts[0]) - 0.5 / self.fr, float(ts[-1]) + 0.5 / self.fr)

    def _candidates(self, row, det):
        thr = det["candidate_threshold"]
        top = np.argpartition(row, -64)[-64:] if len(row) > 64 else np.arange(len(row))
        top = top[np.argsort(row[top])[::-1]]
        picks = []
        sep = det["min_sep_hz"]
        for k in top:
            if row[k] < thr or len(picks) >= det["max_candidates"]:
                break
            if all(abs(self.f[k] - self.f[j]) >= sep for j in picks):
                picks.append(int(k))
        return picks

    def _track_step(self, t, cands, ev_row, comb_row, check_rows, det):
        tol = det["max_slide_hz_s"] / self.fr + 4 * self.df
        used = set()
        for tr in sorted(self.tracks, key=lambda x: -x.conf):
            if t - tr.t_last > det["track_gap_s"]:
                continue
            last_f = tr.frames[-1][1]
            best, bd = None, None
            for k in cands:
                if k in used:
                    continue
                d = abs(self.f[k] - last_f)
                if d <= tol and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                used.add(best)
                tr.add(t, float(self.f[best]), float(ev_row[best]),
                       {c: float(r[best]) for c, r in check_rows.items()})
        for k in cands:
            if k in used or len(self.tracks) >= det["max_tracks"]:
                continue
            if any(abs(self.f[k] - tr.frames[-1][1]) < det["min_sep_hz"] for tr in self.tracks):
                continue
            chain = None
            for (t_end, ch) in self.ended:
                if 0 <= t - t_end <= det["link_hops_s"]:
                    chain = ch
            tr = Track(t, float(self.f[k]), float(ev_row[k]), chain)
            tr.frames[-1] = (t, float(self.f[k]), float(ev_row[k]),
                             {c: float(r[k]) for c, r in check_rows.items()})
            self.tracks.append(tr)

    # ---------------- per track: audio, stage 2, confidence, events ----------------
    def _tracks_audio_and_stage2(self, cfg, ta, tb):
        det, checks = cfg["detect"], cfg["checks"]
        enabled = {c["id"]: checks[c["id"]]["enabled"] and checks[c["id"]]["weight"] > 0
                   for c in CHECKS if c["stage"] == 2}
        weights = {k: v["weight"] for k, v in checks.items() if v["enabled"]}
        alive = []
        for tr in self.tracks:
            ended = tb - tr.t_last > det["track_gap_s"]
            t_from = tr.audio_until if tr.audio_until is not None else tr.t0 - 0.5 / self.fr
            t_to = min(tb, tr.t_last + 0.5 / self.fr)
            if t_to - t_from > 0.02:
                self._demod_track(tr, t_from, t_to, cfg)
            # stage 2 on the last 3 s, stage 1 averaged over the last 1 s of the track
            recent = [fr for fr in tr.frames if fr[0] >= tr.t_last - 1.0]
            s1 = {}
            for fr in recent:
                for c, v in fr[3].items():
                    s1.setdefault(c, []).append(v)
            s1 = {c: float(np.mean(v)) for c, v in s1.items()}
            audio = self._recent(tr.audio, 3.0)
            iq = self._recent(tr.iq, 3.0)
            s2, info = stage2.run(audio, iq, checks, enabled) if len(audio) else ({}, {})
            tr.scores = {**s1, **s2}
            tr.info = {**info, "evidence": round(float(np.mean([fr[2] for fr in recent])) if recent else 0, 3)}
            g = stage1.geo_mean({k: np.array(v) for k, v in tr.scores.items()}, weights)
            tr.conf = float(g) if g is not None else 0.0
            tr.conf_hist.append((round(tb, 2), round(tr.conf, 3)))
            dur = tr.t_last - tr.t0
            if not tr.detected and dur >= det["min_duration_s"] and tr.conf >= det["threshold"]:
                tr.detected = True
                self.emit({"type": "det", "box": tr.box(self.fc)})
            if tr.detected:
                self._emit_audio(tr)
                self.emit({"type": "det_update", "box": tr.box(self.fc)})
                if tr.t_last - tr.seg_start >= cfg["record"]["max_s"]:
                    self._finish(tr, cfg, final=False)
            if ended:
                self.ended.append((tr.t_last, tr.chain))
                if tr.detected:
                    self._finish(tr, cfg, final=True)
                    box = tr.box(self.fc)
                    self.recent_boxes = [b for b in self.recent_boxes if tb - b["t1"] < 60] + [box]
                    self.emit({"type": "det_end", "box": box})
            else:
                alive.append(tr)
        self.tracks = alive
        self.ended = [e for e in self.ended if tb - e[0] < 5]

    @staticmethod
    def _recent(chunks, secs):
        if not chunks:
            return np.zeros(0, np.float32)
        a = np.concatenate([c for _, c in chunks])
        return a[-int(secs * AUDIO_FS):]

    def _demod_track(self, tr, t_from, t_to, cfg):
        pad = int(0.02 * self.fs)
        s0, s1 = int(t_from * self.fs), int(t_to * self.fs)
        x, s0r = self._raw_span(s0 - pad, s1 + pad)
        if len(x) < 2 * pad + 100:
            return
        t_samp = (s0r + np.arange(len(x))) / self.fs
        f = tr.freq_at(t_samp)
        iq, env = demod(x, self.fs, f, cfg["audio"]["bandwidth_hz"])
        cut = int(round((s0 - s0r) / self.fs * AUDIO_FS))
        n = int(round((s1 - s0) / self.fs * AUDIO_FS))
        iq, env = iq[cut:cut + n], env[cut:cut + n]
        if tr.hp_zi is None:
            self._hp = butter(2, 150, btype="highpass", fs=AUDIO_FS, output="sos")
            tr.hp_zi = sosfilt_zi(self._hp) * float(env[0] if len(env) else 0)
        audio, tr.hp_zi = sosfilt(self._hp, env, zi=tr.hp_zi)
        audio = audio.astype(np.float32)
        tr.audio.append((t_from, audio))
        tr.iq.append((t_from, iq.astype(np.complex64)))
        tr.audio_until = t_to

    def _emit_audio(self, tr):
        for t, a in tr.audio:
            if tr.emitted_until is not None and t < tr.emitted_until - 1e-6:
                continue
            rms = float(np.sqrt(np.mean(a ** 2)) + 1e-9)
            tr.agc = rms if tr.agc is None else 0.8 * tr.agc + 0.2 * rms
            pcm = np.clip(a / (4 * tr.agc), -1, 1)
            self.emit({"type": "audio", "id": tr.id, "t": round(t, 4), "rate": AUDIO_FS,
                       "pcm": _b64((pcm * 32767).astype(np.int16))})
            tr.emitted_until = t + len(a) / AUDIO_FS

    # ---------------- recordings ----------------
    def _finish(self, tr, cfg, final):
        rec = cfg["record"]
        seg_t0 = tr.seg_start
        audio = [(t, a) for t, a in tr.audio if t >= seg_t0 - 1e-6]
        iq = [(t, a) for t, a in tr.iq if t >= seg_t0 - 1e-6]
        if not final:
            tr.seg_start = tr.audio_until or tr.t_last
            tr.seg_index += 1
            keep_from = tr.seg_start - 3.0                  # stage 2 still needs recent audio
            tr.audio = [(t, a) for t, a in tr.audio if t >= keep_from]
            tr.iq = [(t, a) for t, a in tr.iq if t >= keep_from]
        if not rec["enabled"] or not audio:
            return
        a = np.concatenate([x for _, x in audio])
        if len(a) < AUDIO_FS // 4:
            return
        out = self.root / rec["dir"]
        out.mkdir(parents=True, exist_ok=True)
        utc = datetime.fromtimestamp(self.utc0 + seg_t0, timezone.utc)
        fr = [p[1] for p in tr.path]
        f_mean = self.fc + float(np.mean([fr_[1] for fr_ in tr.frames if fr_[0] >= seg_t0] or fr))
        stem = f"{utc:%Y%m%d-%H%M%S}_det{tr.id}{'_' + str(tr.seg_index) if tr.seg_index or not final else ''}_{f_mean / 1e3:.1f}kHz"
        pcm = np.clip(a / (4 * (np.sqrt(np.mean(a ** 2)) + 1e-9)), -1, 1)
        wavfile.write(out / f"{stem}.wav", AUDIO_FS, (pcm * 32767).astype(np.int16))
        files = {"wav": f"{stem}.wav"}
        if rec.get("save_iq") and iq:
            np.save(out / f"{stem}_iq16k.npy", np.concatenate([x for _, x in iq]))
            files["iq"] = f"{stem}_iq16k.npy"
        meta = {
            "id": tr.id, "chain": tr.chain, "segment": tr.seg_index,
            "utc_start": utc.isoformat(), "stream_t0": seg_t0, "stream_t1": tr.t_last,
            "duration_s": round(len(a) / AUDIO_FS, 2),
            "freq_hz": {"mean": f_mean, "min": self.fc + min(fr), "max": self.fc + max(fr)},
            "path": [[round(t - seg_t0, 3), self.fc + f] for t, f in tr.path if t >= seg_t0 - 0.5],
            "confidence": {"final": tr.conf, "max": max([c for _, c in tr.conf_hist] or [0]),
                           "history": tr.conf_hist[-600:]},
            "scores": tr.scores, "info": tr.info,
            "sdr": cfg["sdr"], "checks": cfg["checks"], "detect": cfg["detect"],
            "files": files,
        }
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1, default=float))
        self.emit({"type": "recording", "meta": meta})

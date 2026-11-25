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
import collections
import io
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
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
from .sources import _to_complex, make_source
from .audio import FS as _AFS, SmoothGain, StreamingDenoiser, fade, render_clip
from .profile import BandProfile
from .vad_batch import vad_probs_batch
from .transcribe import Transcriber

AUDIO_FS = 16000
_LP = {}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _oscillator(f_offset, n, fs, L=4096):
    """exp(-j 2 pi integral f dt) for a piecewise-constant frequency, without n trig calls:
    each constant run is built as (block steps) x (one block), i.e. ~n/L + L exps per run."""
    f = np.full(n, float(f_offset)) if np.ndim(f_offset) == 0 else np.asarray(f_offset, np.float64)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(f)) + 1, [n]])
    out = np.empty(n, np.complex64)
    phase = 0.0
    for a, b in zip(starts[:-1], starts[1:]):
        m = b - a
        w = -2 * np.pi * f[a] / fs
        base = np.exp(1j * w * np.arange(L))
        steps = np.exp(1j * (phase + w * L * np.arange(-(-m // L))))
        out[a:b] = (steps[:, None] * base[None, :]).ravel()[:m]
        phase += w * m
    return out


_SSB = {}


def _ssb_taps(bw_hz, side):
    """Complex band-pass keeping one side of the carrier: 150 Hz .. bw on the upper (+1) or lower (-1) side."""
    key = (round(bw_hz), side)
    if key not in _SSB:
        half = (bw_hz - 150) / 2
        n = np.arange(129) - 64
        h = firwin(129, half, fs=AUDIO_FS) * np.exp(side * 2j * np.pi * (150 + half) * n / AUDIO_FS)
        _SSB[key] = h.astype(np.complex64)
    return _SSB[key]


def demod(x, fs, f_offset, bw_hz, mode=0):
    """Mix x (complex, at fs) down by f_offset (scalar or per-sample Hz), decimate to 16 kHz.
    mode 0: AM (envelope). +1/-1: upper/lower single sideband (product detector: keep one side,
    take the real part). Returns (narrowband complex IQ at 16 kHz, audio at 16 kHz)."""
    y = x * _oscillator(f_offset, len(x), fs)
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
    if mode:
        s = np.convolve(y, _ssb_taps(bw_hz, int(np.sign(mode))), mode="same")
        return y, (2 * s.real).astype(np.float32)
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
        self.out = []                    # processed playback audio [(t_start, float32)]: denoise + gain + fades
        self.den, self.gain = None, SmoothGain()
        self.audio_t0 = None             # stream time where this track's audio (incl. pre-roll) starts
        self.vad_p, self.vad_t = np.zeros(0, np.float32), -1e18   # incremental ML voice scores
        self.agc = None
        self.seg_start = t               # recording segment start
        self.seg_index = 0
        self.wavs = []                   # recording files for click-to-play
        self.clips = []                  # playable URLs (saved WAV or in-memory clip)
        self.modes = []                  # per frame: 0 AM, +1 USB, -1 LSB (from stage 1)
        self.pitch = []                  # per frame: (t, f0_hz, comb strength) at this track's frequency
        self.hop_ref = None              # pitch where the previous hop of this chain ended

    def mode(self):
        recent = self.modes[-25:]
        if not recent:
            return 0
        s = sum(recent)
        return int(np.sign(s)) if abs(s) > len(recent) / 2 else 0

    def f0_over(self, t_from, t_to, min_c=0.15):
        v = [f for t, f, c in self.pitch if t_from <= t <= t_to and c >= min_c and f > 0]
        return float(np.median(v)) if len(v) >= 3 else None

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
                "conf_max": round(max([c for _, c in self.conf_hist] or [self.conf]), 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "info": self.info, "wavs": list(self.wavs), "clips": list(self.clips), "detected": self.detected,
                "mode": {0: "am", 1: "usb", -1: "lsb"}[self.mode()],
                "audio_t0": self.audio_t0 if self.audio_t0 is not None else self.t0}


class Engine:
    def __init__(self, config, emit, root, source=None, oneshot=False, transcribe=True, score_all=()):
        """source/oneshot: run once over a given source (auto-tune) without housekeeping or disk writes.
        score_all: check ids to compute for every track even when disabled (auto-tune needs every
        column); they don't affect which candidates are found or their confidence."""
        self.score_all = set(score_all)
        self.on_hop = None                       # (engine, cfg, tb) after each hop's tracking (band survey)
        self.on_maps = None                      # (engine, ts, {view: [frames, bins] in 0..1}) each hop
        self.show_conf = 0.0                     # what counts as "caught something" for auto-recalibration
        self.last_hit = time.time()
        self.config = config
        self.emit = emit
        self.root = Path(root)
        self.source_override, self.oneshot = source, oneshot
        self.done = threading.Event()
        self.mem = collections.OrderedDict()        # key -> {"meta", "pcm"}: detections kept in memory, not on disk
        self.mem_bytes = 0
        self._restart = threading.Event()
        self._stop = threading.Event()
        self.status = {"state": "starting"}
        self.tracks, self.ended = [], []
        self.t_origin = time.time()                 # timeline zero: all times are seconds since this
        self.hist = collections.defaultdict(collections.deque)   # view -> waterfall chunks (history)
        self.box_hist = []                          # finished detections still inside the history window
        self.iq_file, self.iq_seg_t0 = None, None
        self._last_prune = 0.0
        self.storage = {"paused": False, "free_gb": None}   # disk guard state
        self.chains = {}                            # hop chain id -> finished member clips
        self.comod = None                           # latest co-modulation candidates / pairs
        self.transcriber = Transcriber(config, emit, on_result=self._on_transcript) if transcribe else None
        self.pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
        self.lock = threading.RLock()                # re-entrant: _finish runs inside locked analysis
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------- control ----------------
    def request_restart(self):
        self._restart.set()

    def stop(self):
        self._stop.set()

    def session(self):
        """(epoch, stream time) of the last receiver (re)start: IQ before it was recorded differently."""
        return (self.t_origin, getattr(self, "session_t0", 0.0))

    def snapshot(self):
        """Live detections (history comes from history())."""
        with self.lock:
            fc = getattr(self, "fc", 0)
            return [t.box(fc) for t in self.tracks if t.detected]

    def history(self, view, t0, t1, max_chunks=1500):
        """Waterfall chunks of one view and detection boxes overlapping [t0, t1]."""
        with self.lock:
            chunks = [c for c in list(self.hist.get(view, ())) if c["t0"] + c["n"] * c["dt"] >= t0 and c["t0"] <= t1]
            fc = getattr(self, "fc", 0)
            boxes = [b for b in self.box_hist if b["t1"] >= t0 and b["t0"] <= t1]
            boxes += [t.box(fc) for t in self.tracks if t.detected]
        return chunks[-max_chunks:], boxes

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
        # continuous timeline across receiver restarts: start this session at "now"
        self.n_samples = int(round((time.time() - self.t_origin) * self.fs))
        self.session_t0 = self.n_samples / self.fs              # when the current receiver settings began
        self.last_hit = time.time()                              # last detection at/above show_conf (wall clock)
        self.raw_keep = int(self.fs * (an["window_s"] + 4))
        self.raw = np.zeros(self.raw_keep, np.complex64)
        self.spec, self.spec_t = [], []
        self.pitch_spec = []                        # per hop: (C, F0) [frames, kept bins] or None
        self._fine_setup()
        self.comod_view = None
        if getattr(self, "profile", None):
            self.profile.maybe_save(force=True)
        pkey = BandProfile.key(self.fc, self.fs, an["bin_hz"], self.f_lo, self.f_hi)
        self.profile = BandProfile(self.root / "live_data" / "profiles" / pkey, len(self.f))
        self.baseline = None
        self.wf_lo = None
        self.analyzed_until = self.n_samples / self.fs
        self.utc0 = self.t_origin
        self._close_iq()
        with self.lock:
            for t in self.tracks:
                if t.detected:
                    self._finish(t, cfg, final=True)
            self.tracks, self.ended = [], []
        self.chains = {}
        self.emit({"type": "reset", "fc": self.fc, "fs": self.fs, "fmin": self.f_lo, "fmax": self.f_hi,
                   "frame_rate": self.fr, "t": self.n_samples / self.fs})

    def _fine_setup(self):
        """Fine (25 Hz) spectrum layout for the pitch comb: which fine bins feed each kept coarse bin."""
        self.fine_n = self.frame_len                         # one 40 ms segment per 25 fps frame
        self.fine_df = self.fs / self.fine_n
        ff = np.fft.fftshift(np.fft.fftfreq(self.fine_n, 1 / self.fs))
        margin = 3400 + 100
        lo, hi = float(self.f[0]) - self.df / 2, float(self.f[-1]) + self.df / 2
        self.fine_sel = np.flatnonzero((ff >= lo - margin) & (ff <= hi + margin))
        fsel = ff[self.fine_sel]
        # for each kept coarse bin, the fine bins within it (contiguous): reduceat start indices
        starts = np.searchsorted(fsel, self.f - self.df / 2)
        self.fine_starts = np.clip(starts, 0, len(fsel) - 1)

    def _pitch_maps(self, x, cfg):
        """Comb strength and pitch per kept coarse bin, for each 40 ms frame of x."""
        ch = cfg["checks"]
        n = len(x) // self.fine_n
        frames = x[: n * self.fine_n].reshape(n, self.fine_n)
        win = np.hanning(self.fine_n).astype(np.float32)

        def job(rows):
            X = np.fft.fftshift(np.fft.fft(frames[rows] * win, axis=1), axes=1)[:, self.fine_sel]
            C, F0 = stage1.pitch_frames((X.real ** 2 + X.imag ** 2).astype(np.float32),
                                        ch["comb"]["f0_min"], ch["comb"]["f0_max"], df=self.fine_df)
            Cc = np.maximum.reduceat(C, self.fine_starts, axis=1)
            w = np.maximum(C, 1e-6)                                # pitch: comb-weighted within each coarse bin
            Fc = np.add.reduceat(w * F0, self.fine_starts, axis=1) / np.add.reduceat(w, self.fine_starts, axis=1)
            return Cc.astype(np.float32), Fc.astype(np.float32)

        parts = list(self.pool.map(job, [slice(i, min(n, i + 5)) for i in range(0, n, 5)]))
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

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
                    if cfg["sdr"]["source"] == "file":     # a capture's name says how it was recorded
                        from ..iq import parse_name
                        fc_f, fs_f = parse_name(cfg["sdr"]["file"]["path"])
                        if fc_f:
                            cfg["sdr"]["center_hz"], cfg["sdr"]["sample_rate"] = fc_f, fs_f
                    if self.source_override:
                        cfg["sdr"]["center_hz"], cfg["sdr"]["sample_rate"] = self.source_override.fc, self.source_override.fs
                    self._reset(cfg)
                    src = self.source_override or make_source(cfg["sdr"])
                    misses = 0
                cfg["sdr"]["center_hz"], cfg["sdr"]["sample_rate"] = self.fc, self.fs
                n = int(self.fs * cfg["analysis"]["hop_s"]) // self.frame_len * self.frame_len
                t_wait = time.time()
                x = src.read(n, timeout=3.0)
                t_got = time.time()
                if x is None and self.oneshot:                 # end of the recording: finish every track
                    with self.lock:
                        self._tracks_audio_and_stage2(cfg, self.analyzed_until, self.analyzed_until + 1e6)
                    break
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
                self._src = src
                self.timing = {}
                t_a = time.time()
                self._ingest(x, cfg)
                self.timing["ingest"] = time.time() - t_a
                self._analyze(cfg)
                # load = all work for this second of signal (conversion + processing) / its duration.
                # Time spent blocked waiting for samples isn't work; conversion happens inside read().
                busy = time.time() - t_got + getattr(src, "convert_s", 0.0)
                load = busy / (len(x) / self.fs)
                self.status = {"state": "running", "source": cfg["sdr"]["source"], "fs": self.fs,
                               "epoch": self.t_origin, "fmin": self.f_lo, "fmax": self.f_hi,
                               "fc": self.fc, "t": self.n_samples / self.fs, "load": round(load, 2),
                               "dropped": getattr(src, "dropped", 0), "error": getattr(src, "error", ""),
                               "tracks": len(self.tracks), "vad": stage2.vad_available(),
                               "storage": self.storage,
                               "profile": {"learned_min": round(self.profile.learned_s / 60, 1)} if getattr(self, "profile", None) else None,
                               "timing": {k: round(v, 3) for k, v in getattr(self, "timing", {}).items()},
                               "progress": (src.sent / src.total) if getattr(src, "total", None) else None,
                               "whisper": {"model": self.transcriber.model_name, "error": self.transcriber.error,
                                           "queue": self.transcriber.qsize(), "busy": self.transcriber.busy,
                                           **self.transcriber.stats} if self.transcriber else {}}
                self.emit({"type": "status", **self.status})
            except Exception as e:
                self.status = {"state": "error", "error": f"{type(e).__name__}: {e}"}
                self.emit({"type": "status", **self.status})
                traceback.print_exc()
                time.sleep(1)
        if src:
            src.stop()
        if getattr(self, "profile", None) and not self.oneshot:
            self.profile.maybe_save(force=True)
        self.done.set()

    # ---------------- ingest: raw ring + spectrogram + waterfall ----------------
    def _ingest(self, x, cfg):
        n = len(x)
        i0 = self.n_samples % self.raw_keep
        first = min(n, self.raw_keep - i0)
        self.raw[i0:i0 + first] = x[:first]
        self.raw[: n - first] = x[first:]
        t_start = self.n_samples / self.fs
        self.n_samples += n
        self._write_iq(x, t_start, cfg)
        if not self.oneshot and time.time() - self._last_prune > 5:
            self._last_prune = time.time()
            self._prune(cfg)
        x = x - x.mean()
        _, _, P = power_frames(x, self.fs, self.nfft, self.fr)
        P = P[:, self.keep]
        ts = t_start + (np.arange(len(P)) + 0.5) / self.fr
        self.spec.append(P)
        self.spec_t.append(ts)
        ch = cfg["checks"]
        if any(ch[k]["enabled"] or k in self.score_all for k in stage1.PITCH) and self.fine_n * self.fr == int(self.fs):
            t_p = time.time()
            self.pitch_spec.append(self._pitch_maps(x, cfg))
            self.timing["pitch"] = time.time() - t_p
        else:
            self.pitch_spec.append(None)
        nkeep = int(cfg["analysis"]["window_s"] * self.fr)
        while sum(len(p) for p in self.spec) - len(self.spec[0]) >= nkeep:
            self.spec.pop(0)
            self.spec_t.pop(0)
            self.pitch_spec.pop(0)
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
        self._chunk("power", ts, q, cfg)

    def _chunk(self, mode, ts, q, cfg):
        """One waterfall chunk (n rows): sent to clients watching `mode` and kept as history."""
        msg = {"type": "wf", "mode": mode, "t0": float(ts[0] - 0.5 / self.fr), "dt": 1.0 / self.fr,
               "n": int(q.shape[0]), "w": int(q.shape[1]), "fmin": self.f_lo, "fmax": self.f_hi, "rows": _b64(q)}
        with self.lock:
            h = self.hist[mode]
            h.append(msg)
            keep_from = msg["t0"] - 60 * cfg["display"]["history_min"]
            while h and h[0]["t0"] < keep_from:
                h.popleft()
        self.emit(msg)

    def _view_rows(self, scores, comb, ev, ts, cfg, extra_views=()):
        """Waterfall rows for the detector's maps (values 0..1 -> 0..255): the combined score, novelty,
        and every check that was computed (only enabled checks are), so each view has history."""
        if not len(ts):
            return
        width = int(cfg["display"]["width"])
        if self.on_maps is not None:
            self.on_maps(self, ts, dict([("score", comb), ("novelty", ev)] + list(scores.items()) + list(extra_views)))
        for mode, a in [("score", comb), ("novelty", ev)] + list(scores.items()) + list(extra_views):
            K = a.shape[1]
            pool = max(1, -(-K // width))
            pad = (-K) % pool
            ap = np.pad(a, ((0, 0), (0, pad))).reshape(len(a), -1, pool).max(axis=2)
            q = (np.clip(ap, 0, 1) * 255).astype(np.uint8)
            self._chunk(mode, ts, q, cfg)

    def _raw_span(self, s0, s1):
        """Absolute sample range [s0, s1) from the ring (must still be held)."""
        s0 = max(s0, self.n_samples - self.raw_keep)
        s1 = min(s1, self.n_samples)
        if s1 <= s0:
            return self.raw[:0], s0
        a, b = s0 % self.raw_keep, s1 % self.raw_keep           # a slice, or two when it wraps
        if a < b:
            return self.raw[a:b], s0
        return np.concatenate([self.raw[a:], self.raw[:b]]), s0

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
        t_s1 = time.time()
        ch = cfg["checks"]
        pitch = None
        if all(p is not None for p in self.pitch_spec) and self.pitch_spec:
            pitch = (np.concatenate([p[0] for p in self.pitch_spec]), np.concatenate([p[1] for p in self.pitch_spec]))
        scores, comb, extra = stage1.run_blocks(P, self.df, self.fr, cfg, sl, force=tuple(self.score_all & set(stage1.STAGE1)),
                                                pool=self.pool, pitch=pitch)
        self.timing["stage1"] = time.time() - t_s1
        ts = T[sl]

        # learned band profile: how unusual is this evidence for these frequencies?
        prof = ch["profile"]
        prof_map = self.profile.score(comb, prof["full_z"]) if prof["enabled"] else None
        if prof["enabled"]:
            if not self.oneshot:
                self.profile.update(comb, self.fr, prof["tau_h"])
                self.profile.maybe_save()
            if self.profile.learned_s >= prof["warmup_min"] * 60:
                scores["profile"] = prof_map
                comb = stage1.combine(scores, ch).astype(np.float32)

        # cross-channel co-modulation among the strongest candidates
        comod_map = None
        self.comod = None
        if ch["comod"]["enabled"] or "comod" in self.score_all:
            t_c = time.time()
            comod_map = self._comodulation(extra["env"], comb, sl, cfg)
            self.timing["comod"] = time.time() - t_c

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
        t_v = time.time()
        views = []
        if comod_map is not None:
            views.append(("comod", comod_map))
        if prof_map is not None and "profile" not in scores:
            views.append(("profile", prof_map))
        self._view_rows(scores, comb, ev, ts, cfg, views)
        self.timing["views"] = time.time() - t_v

        mode, f0r, cr = extra["mode"], extra.get("f0"), extra.get("c")
        with self.lock:
            for i, t in enumerate(ts):
                cands = self._candidates(ev[i], det)
                self._track_step(t, cands, ev[i], comb[i], {c: s[i] for c, s in scores.items()}, det,
                                 mode[i], None if f0r is None else f0r[i], None if cr is None else cr[i])
            t_t = time.time()
            self._tracks_audio_and_stage2(cfg, float(ts[0]) - 0.5 / self.fr, float(ts[-1]) + 0.5 / self.fr)
            self.timing["tracks"] = time.time() - t_t

    def _comodulation(self, env, comb, sl, cfg):
        """Pairs of candidate frequencies whose syllable-band loudness moves together (at once or
        up to max_lag later). The band-wide common movement is removed first, so broadband
        pulsing doesn't count. Returns a sparse [frames, bins] map; keeps pairs for chain merging."""
        prm = cfg["checks"]["comod"]
        T, K = env.shape
        win = min(T, int(2.5 * self.fr))
        e = env[sl.stop - win: sl.stop] if sl.stop >= win else env[:win]
        e = e - np.median(e, axis=1, keepdims=True)              # remove the common (broadband) movement
        strength = comb.max(axis=0)
        order = np.argsort(strength)[::-1]
        sep = max(1, int(cfg["detect"]["min_sep_hz"] / self.df))
        cands = []
        for k in order[:400]:
            if strength[k] < cfg["detect"]["candidate_threshold"] or len(cands) >= 40:
                break
            if all(abs(int(k) - j) >= sep for j in cands):
                cands.append(int(k))
        out = np.zeros_like(comb)
        if len(cands) < 2:
            return out
        X = e[:, cands]
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        nfft = 1 << int(np.ceil(np.log2(2 * win)))
        F = np.fft.rfft(X, nfft, axis=0)
        L = int(prm["max_lag_s"] * self.fr)
        best = np.zeros((len(cands), len(cands)), np.float32)
        lag_at = np.zeros_like(best)
        for i in range(len(cands)):
            xc = np.fft.irfft(F[:, i:i + 1] * np.conj(F), nfft, axis=0) / win   # corr of i with every j
            lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
            seg = xc[lags]                                         # [lags, cands]
            j = np.argmax(seg, axis=0)
            best[i] = seg[j, np.arange(len(cands))]
            lag_at[i] = lags[j] / self.fr
        far = np.abs(self.f[cands][:, None] - self.f[cands][None, :]) >= prm["min_khz"] * 1e3
        best = np.where(far, best, 0)
        score = np.clip((best.max(axis=1) - 0.3) / 0.5, 0, 1)
        for k, s in zip(cands, score):
            out[:, max(0, k - 1): k + 2] = s
        pairs = [(self.f[cands[i]], self.f[cands[j]], float(best[i, j]), float(lag_at[i, j]))
                 for i in range(len(cands)) for j in range(i + 1, len(cands)) if best[i, j] >= 0.6]
        self.comod = {"f": self.f[cands].copy(), "score": score, "pairs": pairs}
        return out

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

    def _track_step(self, t, cands, ev_row, comb_row, check_rows, det, mode_row=None, f0_row=None, c_row=None):
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
                self._track_extras(tr, t, best, mode_row, f0_row, c_row)
        for k in cands:
            if k in used or len(self.tracks) >= det["max_tracks"]:
                continue
            if any(abs(self.f[k] - tr.frames[-1][1]) < det["min_sep_hz"] for tr in self.tracks):
                continue
            chain, ref = None, None
            for (t_end, ch, f0_end) in self.ended:
                if 0 <= t - t_end <= det["link_hops_s"]:
                    chain, ref = ch, f0_end
            tr = Track(t, float(self.f[k]), float(ev_row[k]), chain)
            tr.hop_ref = ref
            tr.frames[-1] = (t, float(self.f[k]), float(ev_row[k]),
                             {c: float(r[k]) for c, r in check_rows.items()})
            self._track_extras(tr, t, k, mode_row, f0_row, c_row)
            self.tracks.append(tr)

    @staticmethod
    def _track_extras(tr, t, k, mode_row, f0_row, c_row):
        if mode_row is not None:
            tr.modes.append(int(mode_row[k]))
            tr.modes = tr.modes[-200:]
        if f0_row is not None:
            tr.pitch.append((t, float(f0_row[k]), float(c_row[k])))
            tr.pitch = tr.pitch[-500:]

    # ---------------- per track: audio, stage 2, confidence, events ----------------
    def _tracks_audio_and_stage2(self, cfg, ta, tb):
        det, checks = cfg["detect"], cfg["checks"]
        enabled = {c["id"]: (checks[c["id"]]["enabled"] and checks[c["id"]]["weight"] > 0) or c["id"] in self.score_all
                   for c in CHECKS if c["stage"] == 2}
        alive = []
        # demodulate every track's new audio in parallel (the heavy, full-rate part)
        jobs = []
        au = cfg["audio"]
        post = min(au.get("postroll_s", 0.5), max(0.0, cfg["analysis"]["settle_s"] - 0.3))
        for tr in self.tracks:
            ending = tb - tr.t_last > det["track_gap_s"]
            t_from = tr.audio_until if tr.audio_until is not None else tr.t0 - au.get("preroll_s", 0.5)
            t_to = tr.t_last + post if ending else min(tb, tr.t_last + 0.5 / self.fr)
            if t_to - t_from > 0.02:
                jobs.append((tr, t_from, t_to, ending))
        self.timing["tracks_n"] = len(self.tracks)
        list(self.pool.map(lambda j: self._demod_track(j[0], j[1], j[2], cfg, j[3]), jobs))
        # ML voice scores: only audio not seen before, all tracks in one batched call
        if enabled.get("vad") and stage2.vad_available():
            new = [(tr, t, a) for tr in self.tracks for t, a in tr.audio if t > tr.vad_t + 1e-6]
            if new:
                for (tr, t, _), p in zip(new, vad_probs_batch([a for _, _, a in new])):
                    tr.vad_p = np.concatenate([tr.vad_p, p])[-int(3.0 * AUDIO_FS / 512):]
                    tr.vad_t = max(tr.vad_t, t)

        def s2job(tr):
            audio, iq = self._recent(tr.audio, 3.0), self._recent(tr.iq, 3.0)
            if not len(audio):
                return {}, {}
            return stage2.run(audio, iq, checks, {**enabled, "am_phase": enabled.get("am_phase") and tr.mode() == 0},
                              vad=tr.vad_p if enabled.get("vad") else None)
        s2_results = dict(zip([id(t) for t in self.tracks], self.pool.map(s2job, self.tracks)))
        for tr in self.tracks:
            ended = tb - tr.t_last > det["track_gap_s"]
            # stage 2 on the last 3 s, stage 1 averaged over the last 1 s of the track
            recent = [fr for fr in tr.frames if fr[0] >= tr.t_last - 1.0]
            s1 = {}
            for fr in recent:
                for c, v in fr[3].items():
                    s1.setdefault(c, []).append(v)
            s1 = {c: float(np.mean(v)) for c, v in s1.items()}
            s2, info = s2_results[id(tr)]
            s2 = dict(s2)
            if enabled.get("comod") and self.comod is not None:  # nearest co-modulation candidate
                d = np.abs(self.comod["f"] - tr.frames[-1][1])
                if len(d) and d.min() <= 2 * self.df:
                    s2["comod"] = float(self.comod["score"][int(d.argmin())])
            if enabled.get("hop") and tr.hop_ref:                  # pitch continues across the hop
                f0_new = tr.f0_over(tr.t0, tr.t0 + 0.6)
                if f0_new:
                    dst = abs(12 * np.log2(f0_new / tr.hop_ref))
                    dst = min(dst, abs(dst - 12))                  # octave errors happen; don't count them
                    s2["hop"] = float(np.exp(-(dst / 3.0) ** 2))
            tr.scores = {**s1, **s2}
            f0_now = tr.f0_over(tr.t_last - 1.0, tr.t_last)
            tr.info = {**info, "evidence": round(float(np.mean([fr[2] for fr in recent])) if recent else 0, 3),
                       **({"f0_median_hz": f0_now} if f0_now else {})}
            g = stage1.combine({k: np.array(v) for k, v in tr.scores.items()}, checks)
            tr.conf = float(g) if g is not None else 0.0
            tr.conf_hist.append((round(tb, 2), round(tr.conf, 3)))
            dur = tr.t_last - tr.t0
            if not tr.detected and dur >= det["min_duration_s"] and tr.conf >= det["threshold"]:
                tr.detected = True
                self.emit({"type": "det", "box": tr.box(self.fc)})
            if tr.detected:
                if tr.conf >= self.show_conf:
                    self.last_hit = time.time()
                self._emit_audio(tr)
                self.emit({"type": "det_update", "box": tr.box(self.fc)})
                if tr.t_last - tr.seg_start >= cfg["record"]["max_s"]:
                    self._finish(tr, cfg, final=False)
            if ended:
                self.ended.append((tr.t_last, tr.chain, tr.f0_over(tr.t_last - 0.6, tr.t_last)))
                if tr.detected:
                    self._finish(tr, cfg, final=True)
                    box = tr.box(self.fc)
                    hist_s = 60 * cfg["display"]["history_min"]
                    self.box_hist = [b for b in self.box_hist if tb - b["t1"] < hist_s] + [box]
                    self.emit({"type": "det_end", "box": box})
            else:
                alive.append(tr)
        if self.on_hop is not None:              # before ended tracks are dropped
            self.on_hop(self, cfg, tb)
        self.tracks = alive
        self.ended = [e for e in self.ended if tb - e[0] < 5]
        self._merge_comod_chains()
        self._flush_chains(cfg, tb)

    def _merge_comod_chains(self):
        """Live tracks on strongly co-modulated frequencies are one voice: join their chains."""
        if not self.comod or not self.comod["pairs"]:
            return
        for fa, fb, c, lag in self.comod["pairs"]:
            ta = [t for t in self.tracks if abs(t.frames[-1][1] - fa) <= 2 * self.df]
            tb_ = [t for t in self.tracks if abs(t.frames[-1][1] - fb) <= 2 * self.df]
            for a in ta:
                for b in tb_:
                    c0 = min(a.chain, b.chain)
                    a.chain = b.chain = c0

    def _flush_chains(self, cfg, tb):
        """A hop chain is finished when none of its tracks is alive and no new hop can link to it.
        Then the members' audio is stitched in time order and transcribed as one utterance."""
        w = cfg["whisper"]
        live = {t.chain for t in self.tracks}
        for cid in list(self.chains):
            ch = self.chains[cid]
            if cid in live or tb - ch["last_end"] < cfg["detect"]["link_hops_s"] + 0.5:
                continue
            del self.chains[cid]
            if len(ch["members"]) < 2 or not (w["enabled"] and w["chains"]) or not self.transcriber:
                continue
            parts = []
            for m in sorted(ch["members"], key=lambda m: m[0]):    # (t0, audio, track id, meta path, key)
                parts += [m[1], np.zeros(int(0.12 * AUDIO_FS), np.float32)]
            self.transcriber.submit({"kind": "chain", "id": cid, "chain": cid, "audio": np.concatenate(parts),
                                     "members": [m[2] for m in ch["members"]],
                                     "keys": [m[4] for m in ch["members"]],
                                     "meta_paths": [m[3] for m in ch["members"] if m[3]]},
                                    priority=ch.get("conf", 0.5))

    @staticmethod
    def _recent(chunks, secs):
        if not chunks:
            return np.zeros(0, np.float32)
        a = np.concatenate([c for _, c in chunks])
        return a[-int(secs * AUDIO_FS):]

    def _demod_track(self, tr, t_from, t_to, cfg, ending=False):
        pad = int(0.02 * self.fs)
        s0, s1 = int(t_from * self.fs), int(t_to * self.fs)
        x, s0r = self._raw_span(s0 - pad, s1 + pad)
        if len(x) < 2 * pad + 100:
            return
        t_samp = (s0r + np.arange(len(x))) / self.fs
        f = tr.freq_at(t_samp)
        iq, env = demod(x, self.fs, f, cfg["audio"]["bandwidth_hz"], tr.mode())
        cut = int(round((s0 - s0r) / self.fs * AUDIO_FS))
        n = int(round((s1 - s0) / self.fs * AUDIO_FS))
        iq, env = iq[cut:cut + n], env[cut:cut + n]
        if tr.hp_zi is None:
            self._hp = butter(2, 150, btype="highpass", fs=AUDIO_FS, output="sos")
            tr.hp_zi = sosfilt_zi(self._hp) * float(env[0] if len(env) else 0)
        audio, tr.hp_zi = sosfilt(self._hp, env, zi=tr.hp_zi)
        audio = audio.astype(np.float32)
        tr.audio.append((t_from, audio))                     # raw: what the checks judge
        tr.iq.append((t_from, iq.astype(np.complex64)))
        tr.audio_until = t_to
        # playback chain: noise reduction -> gliding level -> fades at the very start / end
        au = cfg["audio"]
        first = tr.audio_t0 is None
        if first:
            tr.audio_t0 = t_from
            tr.seg_start = t_from
            if au.get("denoise", True):
                tr.den = StreamingDenoiser(au.get("denoise_strength", 1.0), au.get("denoise_floor_db", -18.0))
        y = tr.den.process(audio) if tr.den else audio
        y = tr.gain.process(y)
        y = fade(y, AUDIO_FS, au.get("fade_s", 0.15) if first else 0.0, au.get("fade_s", 0.15) if ending else 0.0)
        lag = 512 / _AFS if tr.den else 0.0                 # the denoiser delays its output by 32 ms
        tr.out.append((t_from - lag, y))

    def _emit_audio(self, tr):
        for t, a in tr.out:
            if tr.emitted_until is not None and t < tr.emitted_until - 1e-6:
                continue
            pcm = np.clip(a, -1, 1)
            self.emit({"type": "audio", "id": tr.id, "t": round(t, 4), "rate": AUDIO_FS,
                       "pcm": _b64((pcm * 32767).astype(np.int16))})
            tr.emitted_until = t + len(a) / AUDIO_FS

    # ---------------- recordings ----------------
    def _finish(self, tr, cfg, final):
        rec = cfg["record"]
        if self.storage["paused"]:
            rec = dict(rec, enabled=False)           # disk guard: keep detecting, stop writing
        seg_t0 = tr.seg_start
        audio = [(t, a) for t, a in tr.out if t >= seg_t0 - 0.05]
        iq = [(t, a) for t, a in tr.iq if t >= seg_t0 - 1e-6]
        if not final:
            tr.seg_start = tr.audio_until or tr.t_last
            tr.seg_index += 1
            keep_from = tr.seg_start - 3.0                  # stage 2 still needs recent audio
            tr.audio = [(t, a) for t, a in tr.audio if t >= keep_from]
            tr.out = [(t, a) for t, a in tr.out if t >= keep_from]
            tr.iq = [(t, a) for t, a in tr.iq if t >= keep_from]
        if not audio:
            return
        a = np.concatenate([x for _, x in audio])
        if len(a) < AUDIO_FS // 4:
            return
        utc = datetime.fromtimestamp(self.utc0 + seg_t0, timezone.utc)
        fr = [p[1] for p in tr.path]
        f_mean = self.fc + float(np.mean([fr_[1] for fr_ in tr.frames if fr_[0] >= seg_t0] or fr))
        pcm = np.clip(a, -1, 1)                               # already level-controlled and faded
        conf_max = max([c for _, c in tr.conf_hist] or [0])
        meta = {
            "id": tr.id, "chain": tr.chain, "segment": tr.seg_index, "epoch": self.t_origin,
            "utc_start": utc.isoformat(), "stream_t0": seg_t0, "stream_t1": tr.t_last,
            "duration_s": round(len(a) / AUDIO_FS, 2),
            "freq_hz": {"mean": f_mean, "min": self.fc + min(fr), "max": self.fc + max(fr)},
            "path": [[round(t - seg_t0, 3), self.fc + f] for t, f in tr.path if t >= seg_t0 - 0.5],
            "confidence": {"final": tr.conf, "max": conf_max, "history": tr.conf_hist[-600:]},
            "scores": tr.scores, "info": tr.info, "fc": self.fc, "fs": self.fs,
            "sdr": cfg["sdr"], "checks": cfg["checks"], "detect": cfg["detect"],
        }
        if cfg["sdr"]["source"] == "file":
            meta["demo"] = f"Replay of {Path(cfg['sdr']['file']['path']).name} (not a live detection)"
        # Everything detected is kept in memory; only detections above the save threshold go to disk.
        save = (rec["enabled"] and not self.oneshot and conf_max >= rec.get("min_save_conf", 0.7))
        pcm16 = (pcm * 32767).astype(np.int16)
        meta_path = None
        if save:
            meta_path = self._write_recording(meta, pcm16, np.concatenate([x for _, x in iq]) if iq and rec.get("save_iq") else None,
                                              f"{utc:%Y%m%d-%H%M%S}_det{tr.id}{'_' + str(tr.seg_index) if tr.seg_index or not final else ''}_{f_mean / 1e3:.1f}kHz", cfg)
            tr.wavs.append(meta["files"]["wav"])
        else:
            meta.update(saved=False, key=f"mem:{int(self.t_origin)}:{tr.id}:{tr.seg_index}", files={})
            meta["clip"] = "/api/clip?key=" + meta["key"]
            if not self.oneshot:
                self._mem_put(meta, pcm16, cfg)
        tr.clips.append(meta["clip"])
        self.emit({"type": "recording", "meta": {k: v for k, v in meta.items() if k not in ("checks", "detect", "sdr")}})
        w = cfg["whisper"]
        if self.transcriber and w["enabled"] and conf_max >= w["min_confidence"]:
            self.transcriber.submit({"kind": "track", "id": tr.id, "chain": tr.chain, "audio": pcm.astype(np.float32),
                                     "wav": meta["files"].get("wav"), "keys": [meta["key"]],
                                     "meta_paths": [meta_path] if meta_path else []}, priority=conf_max)
        if final:
            ch = self.chains.setdefault(tr.chain, {"members": [], "last_end": tr.t_last})
            ch["members"].append((seg_t0, pcm.astype(np.float32), tr.id, meta_path, meta["key"]))
            ch["last_end"] = max(ch["last_end"], tr.t_last)
            ch["conf"] = max(ch.get("conf", 0.0), conf_max)

    def _write_recording(self, meta, pcm16, iq, stem, cfg):
        out = self.root / cfg["record"]["dir"]
        out.mkdir(parents=True, exist_ok=True)
        self._no_index(out)
        wavfile.write(out / f"{stem}.wav", AUDIO_FS, pcm16)
        files = {"wav": f"{stem}.wav"}
        if iq is not None:
            np.save(out / f"{stem}_iq16k.npy", iq)
            files["iq"] = f"{stem}_iq16k.npy"
        meta.update(files=files, saved=True, key="wav:" + files["wav"], clip="/rec/" + files["wav"])
        path = out / f"{stem}.json"
        path.write_text(json.dumps(meta, indent=1, default=float))
        return path

    # ---------------- in-memory detections ----------------
    def _mem_put(self, meta, pcm16, cfg):
        with self.lock:
            self.mem[meta["key"]] = {"meta": meta, "pcm": pcm16}
            self.mem_bytes += pcm16.nbytes
            cap = cfg["record"].get("memory_mb", 300) * 1e6
            for v in self.mem.values():                 # over budget: drop the oldest audio, keep the listing
                if self.mem_bytes <= cap:
                    break
                if v["pcm"] is not None:
                    self.mem_bytes -= v["pcm"].nbytes
                    v["pcm"] = None
            while len(self.mem) > 50000:
                _, v = self.mem.popitem(last=False)
                if v["pcm"] is not None:
                    self.mem_bytes -= v["pcm"].nbytes

    def mem_list(self):
        with self.lock:
            return [dict(v["meta"], in_memory=v["pcm"] is not None) for v in self.mem.values()]

    def _on_transcript(self, keys, field, res):
        with self.lock:
            for k in keys:
                if k in self.mem:
                    self.mem[k]["meta"][field] = res

    def clip(self, key):
        """WAV bytes for a detection: from memory, from disk, or rebuilt from the IQ ring."""
        cfg = self.config.get()
        if key.startswith("wav:"):
            p = self.root / cfg["record"]["dir"] / key[4:]
            return p.read_bytes() if p.is_file() else None
        with self.lock:
            v = self.mem.get(key)
            pcm, meta = (v["pcm"], v["meta"]) if v else (None, None)
        if meta is None:
            return None
        if pcm is None:
            pcm = self._clip_from_iq(meta, cfg)
        if pcm is None:
            return None
        buf = io.BytesIO()
        wavfile.write(buf, AUDIO_FS, pcm)
        return buf.getvalue()

    def save(self, key):
        """Write an in-memory detection to disk (audio from memory or the IQ ring)."""
        cfg = self.config.get()
        with self.lock:
            v = self.mem.get(key)
        if not v:
            return None
        pcm = v["pcm"] if v["pcm"] is not None else self._clip_from_iq(v["meta"], cfg)
        if pcm is None:
            return None
        meta = dict(v["meta"])
        utc = datetime.fromisoformat(meta["utc_start"])
        self._write_recording(meta, pcm, None, f"{utc:%Y%m%d-%H%M%S}_det{meta['id']}_{meta['freq_hz']['mean'] / 1e3:.1f}kHz", cfg)
        with self.lock:
            old = self.mem.pop(key, None)
            if old and old["pcm"] is not None:
                self.mem_bytes -= old["pcm"].nbytes
        self.emit({"type": "saved", "old_key": key, "meta": {k: x for k, x in meta.items() if k not in ("checks", "detect", "sdr")}})
        return meta

    def _clip_from_iq(self, meta, cfg):
        """Re-demodulate a detection from the continuous IQ ring (if those segments still exist)."""
        d = self.root / cfg["iq_ring"]["dir"]
        if not d.exists():
            return None
        t0, t1 = meta["stream_t0"] - 0.05, meta["stream_t1"] + cfg["audio"].get("postroll_s", 0.5)
        segs = []
        for j in sorted(d.glob("*.json")):
            try:
                sj = json.loads(j.read_text())
            except Exception:
                continue
            p = j.with_suffix(".cs8")
            if abs(sj.get("epoch", 0) - meta["epoch"]) > 1e-3 or not p.exists():
                continue
            dur = p.stat().st_size / 2 / sj["fs"]
            if sj["t0"] < t1 and sj["t0"] + dur > t0:
                segs.append((sj, p))
        if not segs:
            return None
        fs, fc = segs[0][0]["fs"], segs[0][0]["fc"]
        parts, start = [], None
        for sj, p in segs:
            raw = np.memmap(p, dtype=np.int8, mode="r")
            a = max(0, int((t0 - sj["t0"]) * fs))
            b = min(len(raw) // 2, int((t1 - sj["t0"]) * fs))
            if b > a:
                start = sj["t0"] + a / fs if start is None else start
                parts.append(np.asarray(raw[2 * a:2 * b]))
        if not parts:
            return None
        x = _to_complex(np.concatenate(parts), True)
        pts = np.array([[meta["stream_t0"] + r, f - fc] for r, f in meta["path"]] or [[t0, meta["freq_hz"]["mean"] - fc]])
        ts = start + np.arange(len(x)) / fs
        f_off = pts[np.clip(np.searchsorted(pts[:, 0], ts, side="right") - 1, 0, len(pts) - 1), 1]
        _, env = demod(x, fs, f_off, cfg["audio"]["bandwidth_hz"])
        hp = butter(2, 150, btype="highpass", fs=AUDIO_FS, output="sos")
        audio = sosfilt(hp, env - env[:200].mean())
        audio = np.clip(render_clip(audio, cfg["audio"]), -1, 1)
        return (audio * 32767).astype(np.int16)

    # ---------------- continuous IQ ring + size limits ----------------
    def _close_iq(self):
        if self.iq_file:
            self.iq_file.close()
        self.iq_file = None

    @staticmethod
    def _no_index(d):
        """macOS: a .metadata_never_index file stops Spotlight indexing (and burning CPU on) this folder."""
        try:
            (d / ".metadata_never_index").touch(exist_ok=True)
        except OSError:
            pass

    def _write_iq(self, x, t_start, cfg):
        ring = cfg["iq_ring"]
        if (not ring["enabled"] or self.storage["paused"]
                or (cfg["sdr"]["source"] == "file" and not ring["record_replays"])):
            self._close_iq()
            return
        if self.iq_file is None or t_start - self.iq_seg_t0 >= ring["segment_s"]:
            self._close_iq()
            self._prune(cfg)                       # trim on every new segment, so the cap can't overshoot
            d = self.root / ring["dir"]
            d.mkdir(parents=True, exist_ok=True)
            self._no_index(d)
            utc = datetime.fromtimestamp(self.t_origin + t_start, timezone.utc)
            name = f"{utc:%Y%m%d-%H%M%S}_{self.fc / 1e6:g}MHz_{self.fs / 1e6:g}Msps.cs8"
            self.iq_file = open(d / name, "wb")
            self.iq_seg_t0 = t_start
            (d / (name[:-4] + ".json")).write_text(json.dumps(
                {"t0": t_start, "epoch": self.t_origin, "utc": utc.isoformat(), "fc": self.fc, "fs": self.fs,
                 "source": cfg["sdr"]["source"], "format": "cs8"}))
        src = getattr(self, "_src", None)
        raw = getattr(src, "last_raw", None)
        if raw is not None and len(raw) == 2 * len(x):
            from .sources import _to_cs8
            q = _to_cs8(raw, getattr(src, "signed", True))
        else:
            q = np.empty(2 * len(x), np.int8)
            q[0::2] = np.clip(np.round(x.real * 128), -127, 127)
            q[1::2] = np.clip(np.round(x.imag * 128), -127, 127)
        self.iq_file.write(q.tobytes())

    def _prune(self, cfg):
        """Ring-buffer limits and disk guard.

        1. Delete the oldest IQ segments / recordings beyond their size caps. The IQ cap reserves room
           for the segment being written, so ring + current segment never exceeds max_gb.
        2. If free disk space is below storage.min_free_gb, delete more of the IQ ring (it is a cache);
           if that isn't enough, pause all writing (IQ ring and recordings) until space comes back."""
        def enforce(d, max_bytes, group):
            if not d.exists():
                return
            files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.name)
            total = sum(p.stat().st_size for p in files)
            groups = collections.OrderedDict()
            for p in files:
                groups.setdefault(group(p), []).append(p)
            current = self.iq_file.name if self.iq_file else None
            for _, ps in groups.items():
                if total <= max_bytes:
                    break
                if current and any(str(p) == current for p in ps):
                    continue
                for p in ps:
                    total -= p.stat().st_size
                    p.unlink(missing_ok=True)
        seg_bytes = 2 * self.fs * cfg["iq_ring"]["segment_s"] if hasattr(self, "fs") else 0
        iq_dir = self.root / cfg["iq_ring"]["dir"]
        enforce(iq_dir, max(0.0, cfg["iq_ring"]["max_gb"] * 1e9 - seg_bytes), lambda p: p.stem)
        enforce(self.root / cfg["record"]["dir"], cfg["record"]["max_mb"] * 1e6,
                lambda p: p.stem.replace("_iq16k", ""))
        min_free = cfg.get("storage", {}).get("min_free_gb", 10.0) * 1e9
        free = shutil.disk_usage(self.root).free
        if free < min_free and iq_dir.exists():
            for p in sorted((p for p in iq_dir.iterdir() if p.suffix == ".cs8"), key=lambda p: p.name):
                if free >= min_free + 1e9:
                    break
                if self.iq_file and str(p) == self.iq_file.name:
                    continue
                size = p.stat().st_size
                p.unlink(missing_ok=True)
                p.with_suffix(".json").unlink(missing_ok=True)
                free += size
        free = shutil.disk_usage(self.root).free
        paused = free < min_free
        if paused and not self.storage["paused"]:
            self._close_iq()
        self.storage = {"paused": paused, "free_gb": round(free / 1e9, 1)}

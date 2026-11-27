"""Live engine, stripped to the base the filters will be rebuilt on.

  source -> raw IQ (kept in memory for a few seconds, and optionally the IQ ring on disk)
         -> power spectrogram (frames) -> the power waterfall, rows to clients
         -> the pipeline (pipeline.py): IQ steps (none yet), then the sum of everything, as audio

There are no filters and no detections: nothing is chosen automatically. Saved recordings from
before can still be browsed (detail, clip).

Times are "stream seconds" (samples / fs since the engine started). Clients play audio at stream
time + delay, which keeps every sound locked to its place on the waterfall.
"""
import base64
import collections
import json
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ..maps import power_frames
from .audio import fade
from .pipeline import AUDIO_FS, Pipeline
from .sources import _to_complex, make_source
from .transcribe import Transcriber



def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def iq_cut_wav(segs, t0, t1, f_center, rate=48000):
    """Narrowband IQ around f_center from IQ-ring segments [(segment json, path)] for times [t0, t1]
    (the segments' own timeline), shifted to 0 Hz and resampled to `rate`: a stereo (I, Q) 16-bit WAV
    that SDR programs open as IQ. Returns (bytes, actual start time) or (None, None)."""
    import io
    from math import gcd
    from scipy.signal import resample_poly
    parts, start, fs, fc = [], None, None, None
    for sj, p in segs:
        fs, fc = sj["fs"], sj["fc"]
        raw = np.memmap(p, dtype=np.int8, mode="r")
        a = max(0, int((t0 - sj["t0"]) * fs))
        b = min(len(raw) // 2, int((t1 - sj["t0"]) * fs))
        if b > a:
            start = sj["t0"] + a / fs if start is None else start
            parts.append(np.asarray(raw[2 * a:2 * b]))
    if not parts:
        return None, None
    x = _to_complex(np.concatenate(parts), True)
    x = x * np.exp(-2j * np.pi * (f_center - fc) / fs * np.arange(len(x)))
    g = gcd(int(rate), int(fs))
    y = resample_poly(x, int(rate) // g, int(fs) // g)
    y = y / (np.abs(y).max() + 1e-12) * 0.9
    st = (np.stack([y.real, y.imag], axis=1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, int(rate), st)
    return buf.getvalue(), start


def load_iq(segs, t0, t1):
    """Raw IQ of [t0, t1] from IQ-ring segments: (x complex64, start time, fs, fc) or None."""
    if not segs:
        return None
    parts, start, fs, fc = [], None, segs[0][0]["fs"], segs[0][0]["fc"]
    for sj, p in segs:
        raw = np.memmap(p, dtype=np.int8, mode="r")
        a = max(0, int((t0 - sj["t0"]) * fs))
        b = min(len(raw) // 2, int((t1 - sj["t0"]) * fs))
        if b > a:
            start = sj["t0"] + a / fs if start is None else start
            parts.append(np.asarray(raw[2 * a:2 * b]))
    if not parts:
        return None
    return _to_complex(np.concatenate(parts), True), start, fs, fc


def render_segments(segs, t0, t1, au, steps=()):
    """Audio (int16, 16 kHz) of [t0, t1] from IQ-ring segments: that IQ through the pipeline (its
    steps, then the sum of everything), faded in and out."""
    loaded = load_iq(segs, t0, t1)
    if loaded is None:
        return None
    x, start, fs, fc = loaded
    _, y = Pipeline(fs, int(round(start * fs)), steps).process(x)
    y = fade(y, AUDIO_FS, au.get("fade_s", 0.15), au.get("fade_s", 0.15))
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)


class Engine:
    def __init__(self, config, emit, root, source=None, oneshot=False, transcribe=True):
        """source/oneshot: run once over a given source (a replay) without housekeeping or disk writes."""
        self.on_hop = None                       # (engine, cfg, t) after each hop (band survey)
        self.config = config
        self.emit = emit
        self.root = Path(root)
        self.source_override, self.oneshot = source, oneshot
        self.done = threading.Event()
        self._restart = threading.Event()
        self._stop = threading.Event()
        self.status = {"state": "starting"}
        self.t_origin = time.time()                 # timeline zero: all times are seconds since this
        self.hist = collections.defaultdict(collections.deque)   # view -> waterfall chunks (history)
        self.iq_file, self.iq_seg_t0 = None, None
        self._last_prune = 0.0
        self.storage = {"paused": False, "free_gb": None}   # disk guard state
        self.pipe, self.pipe_key = None, None       # the pipeline (pipeline.py) and the steps it was built from
        self.transcriber = Transcriber(config, emit) if transcribe else None
        self.pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
        self.lock = threading.RLock()
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------- control ----------------
    def request_restart(self):
        self._restart.set()

    def stop(self):
        self._stop.set()

    def session(self):
        """(epoch, stream time) of the last receiver (re)start: IQ before it was recorded differently."""
        return (self.t_origin, getattr(self, "session_t0", 0.0))

    def history(self, view, t0, t1, max_chunks=1500):
        """Waterfall chunks of one view overlapping [t0, t1]."""
        with self.lock:
            chunks = [c for c in list(self.hist.get(view, ())) if c["t0"] + c["n"] * c["dt"] >= t0 and c["t0"] <= t1]
        return chunks[-max_chunks:]

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
        self.raw_keep = int(self.fs * 8)                         # a few seconds of raw IQ in memory
        self.raw = np.zeros(self.raw_keep, np.complex64)
        self.spec, self.spec_t = [], []                          # the last few hops' power frames (survey)
        self.wf_lo = {}                                          # per view: the waterfall's floor (dB), tracked
        self._close_iq()
        self.pipe = None                                         # rebuilt for the new rate at the next hop
        self.emit({"type": "reset", "fc": self.fc, "fs": self.fs, "fmin": self.f_lo, "fmax": self.f_hi,
                   "frame_rate": self.fr, "t": self.n_samples / self.fs})

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
                x = src.read(n, timeout=3.0)
                t_got = time.time()
                if x is None and self.oneshot:                 # end of the recording
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
                t_p = time.time()
                self._audio(x, cfg)
                self.timing["pipeline"] = time.time() - t_p
                if self.on_hop is not None:
                    self.on_hop(self, cfg, self.n_samples / self.fs)
                # load = all work for this second of signal (conversion + processing) / its duration.
                # Time spent blocked waiting for samples isn't work; conversion happens inside read().
                busy = time.time() - t_got + getattr(src, "convert_s", 0.0)
                load = busy / (len(x) / self.fs)
                self.status = {"state": "running", "source": cfg["sdr"]["source"], "fs": self.fs,
                               "epoch": self.t_origin, "fmin": self.f_lo, "fmax": self.f_hi,
                               "fc": self.fc, "t": self.n_samples / self.fs, "load": round(load, 2),
                               "dropped": getattr(src, "dropped", 0), "error": getattr(src, "error", ""),
                               "storage": self.storage,
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
        self.done.set()

    # ---------------- ingest: raw IQ + spectrogram + waterfall ----------------
    def _ingest(self, x, cfg):
        n = len(x)
        self._x0 = self.n_samples                            # where this block starts (for the pipeline)
        i0 = self.n_samples % self.raw_keep
        first = min(n, self.raw_keep - i0)
        self.raw[i0:i0 + first] = x[:first]
        self.raw[: n - first] = x[first:]
        t_start = self.n_samples / self.fs
        self.n_samples += n
        self.in_seen = time.time()                            # for the device test: samples are arriving
        self.in_last = x[-min(n, 65536):]
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
        while len(self.spec) > 4:
            self.spec.pop(0)
            self.spec_t.pop(0)
        self._ts = ts
        self._waterfall(P, ts, cfg)

    def _waterfall(self, P, ts, cfg, view="power"):
        width = int(cfg["display"]["width"])
        K = P.shape[1]
        pool = max(1, -(-K // width))
        pad = (-K) % pool
        Pp = np.pad(P, ((0, 0), (0, pad)), constant_values=0).reshape(len(P), -1, pool).max(axis=2)
        db = 10 * np.log10(Pp + 1e-12)
        lo = np.percentile(db, 5)
        self.wf_lo[view] = lo if view not in self.wf_lo else 0.9 * self.wf_lo[view] + 0.1 * lo
        q = np.clip((db - self.wf_lo[view]) / cfg["display"]["dyn_range_db"] * 255, 0, 255).astype(np.uint8)
        self._chunk(view, ts, q, cfg)
        if view == "power":
            self._last_q = q

    def _chunk(self, mode, ts, q, cfg, keep=True):
        """One waterfall chunk (n rows): sent to clients watching `mode` and (keep) kept as history."""
        msg = {"type": "wf", "mode": mode, "t0": float(ts[0] - 0.5 / self.fr), "dt": 1.0 / self.fr,
               "n": int(q.shape[0]), "w": int(q.shape[1]), "fmin": self.f_lo, "fmax": self.f_hi, "rows": _b64(q)}
        if not keep:
            self.emit(msg)
            return
        with self.lock:
            h = self.hist[mode]
            h.append(msg)
            keep_from = msg["t0"] - 60 * cfg["display"]["history_min"]
            while h and h[0]["t0"] < keep_from:
                h.popleft()
        self.emit(msg)

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

    # ---------------- the pipeline: IQ steps, then the sum of everything as audio ----------------
    def _audio(self, x, cfg):
        """This hop's IQ through the pipeline; the audio goes to clients with its stream time, and the
        IQ after the radio steps becomes the "pipe" waterfall (the same as "power" with no radio step on)."""
        steps = cfg.get("pipeline") or []
        key = json.dumps(steps, sort_keys=True)
        if self.pipe is None:
            self.pipe = Pipeline(self.fs, self._x0, steps)
        elif key != self.pipe_key:                               # steps changed: same pipeline, new settings
            self.pipe.configure(steps)
        self.pipe_key = key
        k0, y = self.pipe.process(x)
        if self.pipe.radio_on:
            iq = self.pipe.iq
            _, _, P = power_frames(iq - iq.mean(), self.fs, self.nfft, self.fr)
            self._waterfall(P[:, self.keep], self._ts, cfg, view="pipe")
        else:
            self._chunk("pipe", self._ts, self._last_q, cfg)
        if self.oneshot or not len(y):
            return
        self.emit({"type": "audio", "id": "out", "t": round(k0 / AUDIO_FS, 5), "rate": AUDIO_FS,
                   "pcm": _b64((np.clip(y, -1, 1) * 32767).astype(np.int16))})

    # ---------------- saved recordings (from before the filters were removed) ----------------
    def detail(self, key):
        """Full metadata of one saved recording, for the detail dialog."""
        if not key.startswith("wav:"):
            return None
        p = (self.root / self.config.get()["record"]["dir"] / key[4:]).with_suffix(".json")
        return json.loads(p.read_text()) if p.is_file() else None

    def clip(self, key):
        """WAV bytes of a saved recording."""
        if not key.startswith("wav:"):
            return None
        p = self.root / self.config.get()["record"]["dir"] / key[4:]
        return p.read_bytes() if p.is_file() else None

    def ring_segments(self, epoch, t0, t1):
        """IQ-ring segments [(segment json, path)] of one session overlapping stream times [t0, t1]."""
        d = self.root / self.config.get()["iq_ring"]["dir"]
        out = []
        for j in sorted(d.glob("*.json")) if d.exists() else []:
            try:
                sj = json.loads(j.read_text())
            except Exception:
                continue
            p = j.with_suffix(".cs8")
            if abs(sj.get("epoch", 0) - epoch) > 1e-3 or not p.exists():
                continue
            if sj["t0"] < t1 and sj["t0"] + p.stat().st_size / 2 / sj["fs"] > t0:
                out.append((sj, p))
        return out

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

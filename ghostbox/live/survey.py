"""Band survey: step the receiver across a wide range and map what the detector picks.

At each step the live detector runs exactly as currently tuned for `dwell_s` seconds. The survey
records, on a common time x frequency grid:

- power (mean dB), the background,
- the candidates the detector actually tracks, painted with their confidence: this is the map of
  what the tuned detector chooses, not of an intermediate evidence layer,
- per candidate: frequency, confidence, duration and the ML voice score (computed for every
  candidate during the survey even if the voice check is off; it doesn't change confidence).

From that it ranks the range in cells: where the most voice-like activity is, and where it is
quietest. The receiver goes back to where it was at the end. Surveys are saved to
live_data/surveys/ and can be reopened.
"""
import base64
import copy
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROWS = 60                      # time rows per step (the map is wide and short)
MAX_COLS = 4000                # columns across the whole range
USABLE = 0.8                   # fraction of the sample rate used per step (edges roll off)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def nice_cell(span_hz, target=60):
    raw = span_hz / target
    for c in (5e3, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6):
        if c >= raw:
            return c
    return 10e6


class Survey:
    def __init__(self, root, config, apply, engine, emit, busy=lambda: False):
        self.root, self.config, self.apply, self.engine, self.emit = Path(root), config, apply, engine, emit
        self.busy = busy                       # () -> True if something else is using the receiver
        self.state = {"state": "idle"}
        self.result = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._step = None
        self.ended_at = 0.0

    # ---------------- control ----------------
    def start(self, lo_hz, hi_hz, dwell_s):
        with self._lock:
            if self.state.get("state") == "running":
                return False
            self._stop.clear()
            self.state = {"state": "running", "progress": 0.0, "message": "Starting…"}
        threading.Thread(target=self._run, args=(float(lo_hz), float(hi_hz), float(dwell_s)), daemon=True).start()
        return True

    def stop(self):
        self._stop.set()

    @property
    def running(self):
        return self.state.get("state") == "running"

    def _set(self, **kw):
        self.state = {**self.state, **kw}
        self.emit({"type": "survey", **self.state})

    # ---------------- collection (called on the engine thread every hop) ----------------
    def _on_hop(self, eng, cfg, tb):
        st = self._step
        if st is None:
            return
        try:
            with self._lock:
                self._collect(eng, cfg, st)
        except Exception:
            traceback.print_exc()

    def _collect(self, eng, cfg, st):
        t_a, t_b = st["t_a"], st["t_b"]
        # power frames not seen yet
        for P, ts in zip(eng.spec, eng.spec_t):
            new = (ts > st["last_t"]) & (ts >= t_a) & (ts < t_b)
            if not new.any():
                continue
            if st["bins"] is None or st["bins"][0] != P.shape[1]:
                f_abs = eng.fc + eng.f
                col = np.floor((f_abs - st["lo"]) / st["col_hz"]).astype(int)
                ok = (col >= 0) & (col < st["ncols"])
                st["bins"] = (P.shape[1], col, ok)
            _, col, ok = st["bins"]
            db = 10 * np.log10(P[new][:, ok] + 1e-12)
            rows = np.clip(((ts[new] - t_a) / (t_b - t_a) * ROWS).astype(int), 0, ROWS - 1)
            c = col[ok]
            for r, row in zip(rows, db):
                np.add.at(st["pow"][r], c, row)
                np.add.at(st["pow_n"][r], c, 1)
            st["last_t"] = max(st["last_t"], float(ts[new].max()))
        # candidate tracks, painted with their current confidence
        half = max(1, int(round(1500 / st["col_hz"])))
        for tr in eng.tracks:
            seen = st["seen"].get(tr.id, -1e18)
            fr = [x for x in tr.frames if x[0] > seen and t_a <= x[0] < t_b]
            if fr:
                st["seen"][tr.id] = fr[-1][0]
            for t, f, _, _ in fr:
                r = min(ROWS - 1, int((t - t_a) / (t_b - t_a) * ROWS))
                c0 = int((eng.fc + f - st["lo"]) / st["col_hz"])
                a, b = max(0, c0 - half), min(st["ncols"], c0 + half + 1)
                if a < b:
                    st["det"][r, a:b] = np.maximum(st["det"][r, a:b], tr.conf)
            if tr.t0 >= t_a - 1 and tr.t0 < t_b:
                fs_ = [p[1] for p in tr.path]
                st["tracks"][tr.id] = {"f": float(eng.fc + np.mean(fs_)), "t0": tr.t0 - t_a,
                                       "dur": float(tr.t_last - tr.t0), "conf": float(max(tr.conf, st["tracks"].get(tr.id, {}).get("conf", 0))),
                                       "vad": tr.scores.get("vad"), "detected": bool(tr.detected)}

    # ---------------- the run ----------------
    def _run(self, lo, hi, dwell):
        cfg0 = self.config.get()
        orig_sdr = copy.deepcopy(cfg0["sdr"])
        eng = self.engine()
        fs = float(cfg0["sdr"]["sample_rate"])
        width = fs * USABLE
        n = max(1, int(np.ceil((hi - lo) / width)))
        hi = lo + n * width
        ncols_total = min(MAX_COLS, int((hi - lo) / eng.df))
        col_hz = (hi - lo) / ncols_total
        per = int(round(width / col_hz))
        ncols_total = per * n
        col_hz = (hi - lo) / ncols_total
        thr = float(cfg0["detect"]["threshold"])
        res = {"lo": lo, "hi": hi, "dwell_s": dwell, "col_hz": col_hz, "rows": ROWS, "step_hz": width,
               "source": cfg0["sdr"]["source"], "sample_rate": fs, "threshold": thr,
               "floor": float(cfg0["autotune"].get("floor", 0.3)),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
        self.result = res
        prev_force = set(eng.score_all)
        eng.score_all = prev_force | {"vad"}         # voice score for every candidate (confidence unchanged)
        eng.on_hop = self._on_hop
        try:
            for i in range(n):
                if self._stop.is_set():
                    break
                s_lo = lo + i * width
                fc = s_lo + width / 2
                self._set(progress=i / n, step=i, steps=n,
                          message=f"Step {i + 1} of {n}: {s_lo / 1e6:.3f}–{(s_lo + width) / 1e6:.3f} MHz")
                before = eng.session()
                self.apply({"sdr": {"center_hz": fc, "fmin_hz": s_lo, "fmax_hz": s_lo + width}})
                t_w = time.time()
                while eng.session() == before and time.time() - t_w < 15 and not self._stop.is_set():
                    time.sleep(0.2)
                if eng.session() == before:
                    raise RuntimeError("The receiver did not retune.")
                t_a = eng.session()[1] + 1.5                 # skip the retune transient
                st = {"lo": s_lo, "col_hz": col_hz, "ncols": per, "t_a": t_a, "t_b": t_a + dwell,
                      "pow": np.zeros((ROWS, per)), "pow_n": np.zeros((ROWS, per)),
                      "det": np.zeros((ROWS, per), np.float32), "seen": {}, "tracks": {},
                      "last_t": -1e18, "bins": None}
                self._step = st
                # wait until the detector has processed the whole dwell (it runs a few seconds behind)
                t_w = time.time()
                while not self._stop.is_set():
                    done = (st["last_t"] - t_a) / dwell
                    self._set(progress=(i + min(1, max(0, done))) / n)
                    if st["last_t"] >= st["t_b"] - 2.0 / eng.fr or time.time() - t_w > dwell + 30:
                        break
                    time.sleep(0.5)
                time.sleep(0.5)                            # let the last hop's tracks finish scoring
                with self._lock:
                    self._step = None
                res["steps"].append(self._summarise_step(st, fc, s_lo, width, dwell, thr))
                self._set(steps_done=len(res["steps"]))
            self._finish(res)
            stopped = self._stop.is_set()
            self._set(state="done", progress=1.0, file=res.get("file"),
                      message=("Stopped" if stopped else "Done") + f": {len(res['steps'])} of {n} steps.")
        except Exception as e:
            traceback.print_exc()
            self._set(state="error", message=str(e))
        finally:
            self._step = None
            eng.on_hop = None
            eng.score_all = prev_force
            self.apply({"sdr": orig_sdr})                  # back to where you were
            self.ended_at = time.time()

    def _summarise_step(self, st, fc, s_lo, width, dwell, thr):
        with np.errstate(invalid="ignore", divide="ignore"):
            pw = st["pow"] / st["pow_n"]
        filled = st["pow_n"] > 0
        floor = float(np.nanmedian(pw[filled])) if filled.any() else -100.0
        if filled.any():                                    # fill empty rows from neighbours
            colmean = np.nanmean(np.where(filled, pw, np.nan), axis=0)
            pw = np.where(filled, pw, np.nan_to_num(colmean, nan=floor))
        else:
            pw = np.full_like(pw, floor)
        tracks = [t for t in st["tracks"].values() if t["dur"] >= 0.3]
        return {"fc": fc, "lo": s_lo, "hi": s_lo + width, "floor_db": floor,
                "pow": pw.astype(np.float32), "det": st["det"], "tracks": tracks,
                "candidates": len(tracks), "detections": sum(t["conf"] >= thr for t in tracks)}

    # ---------------- results ----------------
    def _finish(self, res):
        steps = res["steps"]
        if not steps:
            return
        allpow = np.concatenate([s["pow"] for s in steps], axis=1)
        db_lo, db_hi = np.percentile(allpow, 2), np.percentile(allpow, 99.7)
        res["db_range"] = [float(db_lo), float(db_hi)]
        res["cells"] = self._cells(res, allpow)
        # recommendations
        cells = res["cells"]
        busy = sorted([c for c in cells if c["candidates_per_min"] > 0],
                      key=lambda c: (-c["detections_per_min"], -c["candidates_per_min"]))[:6]
        voice = sorted([c for c in cells if c["voice_score"] > 0], key=lambda c: -c["voice_score"])[:6]
        quiet = sorted(cells, key=lambda c: (c["candidates_per_min"], c["occupancy"], c["floor_db"]))[:6]
        res["suggest"] = {"activity": [c["i"] for c in busy], "voice": [c["i"] for c in voice],
                          "quiet": [c["i"] for c in quiet]}
        self._save(res)

    def _cells(self, res, allpow):
        lo, hi, col_hz = res["lo"], res["hi"], res["col_hz"]
        cell = nice_cell(hi - lo)
        res["cell_hz"] = cell
        ncell = int(np.ceil((hi - lo) / cell))
        floor_all = np.median(allpow)
        active = (allpow > np.median(allpow, axis=1, keepdims=True) + 10)   # >10 dB over the row median
        mins = res["dwell_s"] / 60
        tracks = [t for s in res["steps"] for t in s["tracks"]]
        cells = []
        for i in range(ncell):
            a, b = lo + i * cell, min(hi, lo + (i + 1) * cell)
            ca, cb = int((a - lo) / col_hz), max(int((a - lo) / col_hz) + 1, int((b - lo) / col_hz))
            ts = [t for t in tracks if a <= t["f"] < b]
            vads = [t["vad"] for t in ts if t["vad"] is not None]
            # cells span several steps' columns but each column was watched for one dwell
            cells.append({
                "i": i, "lo": a, "hi": b,
                "occupancy": round(float(active[:, ca:cb].mean()), 4),
                "floor_db": round(float(np.median(allpow[:, ca:cb]) - floor_all), 2),
                "candidates_per_min": round(len(ts) / mins, 2),
                "detections_per_min": round(sum(t["conf"] >= res["threshold"] for t in ts) / mins, 2),
                "voice_max": round(max(vads), 3) if vads else None,
                # only candidates the voice detector rates above the calibration floor count as voice-like
                "voice_like_per_min": round(sum((t["vad"] or 0) >= res["floor"] for t in ts) / mins, 2),
                "voice_score": round(sum(t["vad"] for t in ts if (t["vad"] or 0) >= res["floor"]) / mins, 3),
            })
        return cells

    def public(self, res=None):
        """The result as sent to the page: maps as 8-bit images, numbers as plain JSON."""
        res = res or self.result
        if not res:
            return None
        steps = res["steps"]
        if not steps:
            return {k: v for k, v in res.items() if k != "steps"} | {"steps": []}
        if "db_range" in res:
            db_lo, db_hi = res["db_range"]
        else:                                                # still running: scale from what we have
            allpow = np.concatenate([s["pow"] for s in steps], axis=1)
            db_lo, db_hi = float(np.percentile(allpow, 2)), float(np.percentile(allpow, 99.7))
        out = {k: v for k, v in res.items() if k != "steps"}
        out["steps"] = [{
            **{k: v for k, v in s.items() if k not in ("pow", "det")},
            "cols": int(s["pow"].shape[1]),
            "pow": _b64(np.clip((s["pow"] - db_lo) / max(1e-6, db_hi - db_lo) * 255, 0, 255)),
            "det": _b64(np.clip(s["det"] * 255, 0, 255)),
        } for s in steps]
        return out

    def _save(self, res):
        d = self.root / "live_data" / "surveys"
        d.mkdir(parents=True, exist_ok=True)
        name = (f"{time.strftime('%Y-%m-%d_%H%M')}_{res['source']}_"
                f"{res['lo'] / 1e6:.3f}-{res['hi'] / 1e6:.3f}MHz.json")
        res["file"] = name
        (d / name).write_text(json.dumps(self.public(res), default=float))

    def load(self, name):
        p = self.root / "live_data" / "surveys" / name
        return json.loads(p.read_text()) if p.is_file() else None

    def saved(self):
        d = self.root / "live_data" / "surveys"
        return sorted((p.name for p in d.glob("*.json")), reverse=True) if d.exists() else []

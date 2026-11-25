"""Band survey: step the receiver across a wide range and map what the detector sees and picks.

At each step the live detector runs exactly as currently tuned for `dwell_s` seconds. The survey
records, per step, on a time x frequency grid at the analysis resolution (about 1 kHz columns):

- power (mean dB),
- every detector map the waterfall can show (combined score, novelty, each enabled check), max-pooled,
- the candidates the detector tracks, as boxes with their confidence, ML voice score (computed for
  every candidate during the survey; it doesn't change confidence) and playable clip.

The receiver goes back to where it was at the end. Surveys are saved to live_data/surveys/ as a
JSON description plus a compressed .npz of the map images, and can be reopened.
"""
import copy
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROWS = 60                      # time rows per step (the map is wide and short)
MAX_COLS = 4000                # columns per step
USABLE = 0.8                   # fraction of the sample rate used per step (edges roll off)


class Survey:
    def __init__(self, root, config, apply, engine, emit):
        self.root, self.config, self.apply, self.engine, self.emit = Path(root), config, apply, engine, emit
        self.dir = self.root / "live_data" / "surveys"
        self.state = {"state": "idle"}
        self.result = None
        self.ended_at = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._step = None

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

    # ---------------- collection (engine thread) ----------------
    def _layout(self, eng, st, K):
        """Map the engine's K analysis bins onto this step's columns (contiguous runs for reduceat)."""
        if st["bins"] is not None and st["bins"][0] == K:
            return st["bins"]
        col = np.floor((eng.fc + eng.f - st["lo"]) / st["col_hz"]).astype(int)
        ok = np.flatnonzero((col >= 0) & (col < st["ncols"]))
        c = col[ok]
        starts = np.flatnonzero(np.r_[True, np.diff(c) != 0])
        st["bins"] = (K, ok, starts, c[starts])
        return st["bins"]

    def _rows(self, st, ts):
        return np.clip(((ts - st["t_a"]) / (st["t_b"] - st["t_a"]) * ROWS).astype(int), 0, ROWS - 1)

    def _on_maps(self, eng, ts, maps):
        st = self._step
        if st is None:
            return
        try:
            with self._lock:
                sel = (ts >= st["t_a"]) & (ts < st["t_b"])
                if not sel.any():
                    return
                rows = self._rows(st, ts[sel])
                for name, a in maps.items():
                    _, ok, starts, ucol = self._layout(eng, st, a.shape[1])
                    q = (np.clip(a[sel][:, ok], 0, 1) * 255).astype(np.uint8)
                    red = np.maximum.reduceat(q, starts, axis=1)
                    img = st["maps"].setdefault(name, np.zeros((ROWS, st["ncols"]), np.uint8))
                    for r in np.unique(rows):
                        img[r, ucol] = np.maximum(img[r, ucol], red[rows == r].max(axis=0))
        except Exception:
            traceback.print_exc()

    def _on_hop(self, eng, cfg, tb):
        st = self._step
        if st is None:
            return
        try:
            with self._lock:
                self._collect(eng, st)
        except Exception:
            traceback.print_exc()

    def _collect(self, eng, st):
        t_a, t_b = st["t_a"], st["t_b"]
        for P, ts in zip(eng.spec, eng.spec_t):            # power frames not seen yet
            new = (ts > st["last_t"]) & (ts >= t_a) & (ts < t_b)
            if not new.any():
                continue
            _, ok, starts, ucol = self._layout(eng, st, P.shape[1])
            db = 10 * np.log10(P[new][:, ok] + 1e-12)
            s = np.add.reduceat(db, starts, axis=1)
            n = np.diff(np.r_[starts, len(ok)])
            rows = self._rows(st, ts[new])
            for r in np.unique(rows):
                st["pow"][r, ucol] += s[rows == r].sum(axis=0)
                st["pow_n"][r, ucol] += n * int((rows == r).sum())
            st["last_t"] = max(st["last_t"], float(ts[new].max()))
        for tr in eng.tracks:                               # candidate tracks as boxes
            if tr.t_last < t_a or tr.t0 >= t_b:
                continue
            fr = [p[1] for p in tr.path]
            prev = st["tracks"].get(tr.id, {})
            st["tracks"][tr.id] = {
                "id": tr.id, "f": float(eng.fc + np.mean(fr)),
                "fmin": float(eng.fc + min(fr)), "fmax": float(eng.fc + max(fr)),
                "t0": round(max(0.0, tr.t0 - t_a), 2), "t1": round(min(t_b, tr.t_last) - t_a, 2),
                "conf": round(float(max(tr.conf, prev.get("conf", 0))), 3),
                "vad": None if tr.scores.get("vad") is None else round(float(tr.scores["vad"]), 3),
                "clips": list(tr.clips), "detected": bool(tr.detected)}

    # ---------------- the run ----------------
    def _run(self, lo, hi, dwell):
        cfg0 = self.config.get()
        orig_sdr = copy.deepcopy(cfg0["sdr"])
        eng = self.engine()
        fs = float(cfg0["sdr"]["sample_rate"])
        width = fs * USABLE
        n = max(1, int(np.ceil((hi - lo) / width)))
        hi = lo + n * width
        per = int(min(MAX_COLS, width / eng.df))
        res = {"lo": lo, "hi": hi, "dwell_s": dwell, "rows": ROWS, "step_hz": width, "cols_per_step": per,
               "source": cfg0["sdr"]["source"], "sample_rate": fs,
               "threshold": float(cfg0["detect"]["threshold"]), "floor": float(cfg0["autotune"].get("floor", 0.3)),
               "checks": {k: v["weight"] for k, v in cfg0["checks"].items() if v["enabled"]},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
        self.result = res
        prev_force = set(eng.score_all)
        eng.score_all = prev_force | {"vad"}              # voice score for every candidate (confidence unchanged)
        eng.on_hop, eng.on_maps = self._on_hop, self._on_maps
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
                st = {"lo": s_lo, "col_hz": width / per, "ncols": per, "t_a": t_a, "t_b": t_a + dwell,
                      "pow": np.zeros((ROWS, per)), "pow_n": np.zeros((ROWS, per)), "maps": {},
                      "tracks": {}, "last_t": -1e18, "bins": None}
                self._step = st
                t_w = time.time()                          # the detector runs a few seconds behind
                while not self._stop.is_set():
                    self._set(progress=(i + min(1, max(0, (st["last_t"] - t_a) / dwell))) / n)
                    if st["last_t"] >= st["t_b"] - 2.0 / eng.fr or time.time() - t_w > dwell + 30:
                        break
                    time.sleep(0.5)
                time.sleep(1.0)                            # last hop's tracks finish scoring
                with self._lock:
                    self._step = None
                res["steps"].append(self._summarise(st, s_lo, width))
                self._set(steps_done=len(res["steps"]))
            stopped = self._stop.is_set()
            if res["steps"]:
                self._save(res)
            self._set(state="done", progress=1.0, file=res.get("file"),
                      message=("Stopped" if stopped else "Done") + f": {len(res['steps'])} of {n} steps.")
        except Exception as e:
            traceback.print_exc()
            self._set(state="error", message=str(e))
        finally:
            self._step = None
            eng.on_hop = eng.on_maps = None
            eng.score_all = prev_force
            self.apply({"sdr": orig_sdr})                  # back to where you were
            self.ended_at = time.time()

    def _summarise(self, st, s_lo, width):
        with np.errstate(invalid="ignore", divide="ignore"):
            pw = st["pow"] / st["pow_n"]
        filled = st["pow_n"] > 0
        floor = float(np.median(pw[filled])) if filled.any() else -100.0
        if filled.any():                                    # empty cells: column mean, else the floor
            colmean = np.nanmean(np.where(filled, pw, np.nan), axis=0)
            pw = np.where(filled, pw, np.nan_to_num(colmean, nan=floor))
        else:
            pw = np.full_like(pw, floor)
        tracks = sorted((t for t in st["tracks"].values() if t["t1"] - t["t0"] >= 0.3), key=lambda t: t["t0"])
        return {"lo": s_lo, "hi": s_lo + width, "cols": st["ncols"], "floor_db": floor,
                "pow": pw.astype(np.float32), "maps": st["maps"], "tracks": tracks}

    # ---------------- results ----------------
    def meta(self, res=None):
        """The survey without its images (those come per view from image())."""
        res = res if res is not None else self.result
        if not res:
            return None
        out = {k: v for k, v in res.items() if k != "steps"}
        out["steps"] = [{k: v for k, v in s.items() if k not in ("pow", "maps")} for s in res["steps"]]
        views = ["power"]
        for s in res["steps"]:
            views += [v for v in s.get("maps", {}) if v not in views]
        out["views"] = out.get("views") or views
        return out

    def image(self, view, file=None):
        """uint8 image [rows, all steps' columns side by side] for one view, or None."""
        if file:
            p = self.dir / (file[:-5] + ".npz")
            if not p.is_file():
                return None
            with np.load(p) as z:
                return z[view] if view in z.files else None
        res = self.result
        if not res or not res["steps"]:
            return None
        if view == "power":
            allpow = np.concatenate([s["pow"] for s in res["steps"]], axis=1)
            lo_db, hi_db = np.percentile(allpow, 2), np.percentile(allpow, 99.7)
            return np.clip((allpow - lo_db) / max(1e-6, hi_db - lo_db) * 255, 0, 255).astype(np.uint8)
        return np.concatenate([s["maps"].get(view, np.zeros((ROWS, s["cols"]), np.uint8)) for s in res["steps"]], axis=1)

    def _save(self, res):
        self.dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y-%m-%d_%H%M')}_{res['source']}_{res['lo'] / 1e6:.3f}-{res['hi'] / 1e6:.3f}MHz"
        res["file"] = name + ".json"
        m = self.meta(res)
        np.savez_compressed(self.dir / (name + ".npz"), **{v: self.image(v) for v in m["views"]})
        (self.dir / (name + ".json")).write_text(json.dumps(m, default=float))

    def load(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.is_file() else None

    def saved(self):
        return sorted((p.name for p in self.dir.glob("*.json")), reverse=True) if self.dir.exists() else []

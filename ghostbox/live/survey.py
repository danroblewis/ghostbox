"""Band survey: step the receiver across a wide range and map its power.

Recording and processing overlap. The receiver records `dwell_s` of raw IQ at each step (the IQ
ring) and moves on; a worker replays each finished step once and adds its power to the map. (The
detector and the per-step calibration and transcription were removed with the filters; older
surveys that have them can still be opened.)

The receiver goes back to where it was. Surveys are saved to live_data/surveys/<name>.json (+ .npz
maps) after every step, and can be reopened.
"""
import copy
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .autotune import pick_segments, replay

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
        self.previews = {}                   # step index -> power preview while recorded / queued
        self._preview_live = None

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

    # ---------------- collection during a step's replay ----------------
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

    def _window(self, eng, st):
        if st["t_a"] is None:                           # the replay's own timeline starts at its session
            st["t_a"] = eng.session()[1] + 1.0         # (skip the retune transient)
            st["t_b"] = st["t_a"] + st["dur"] - 1.0 - st["lag"]   # the last settle+hop is never analysed
        return st["t_a"], st["t_b"]

    def _rows(self, st, ts):
        return np.clip(((ts - st["t_a"]) / (st["t_b"] - st["t_a"]) * ROWS).astype(int), 0, ROWS - 1)

    def _on_live_hop(self, eng, cfg, tb):
        """Live receiver hook while recording: paint the step being recorded (power preview)."""
        st = self._preview_live
        if st is not None:
            with self._lock:
                self._power(eng, st)

    def _on_hop(self, eng, cfg, tb):
        st = self._step
        if st is None:
            return
        self._power(eng, st)

    def _power(self, eng, st):
        try:
            t_a, t_b = self._window(eng, st)
            for P, ts in zip(eng.spec, eng.spec_t):     # power frames not seen yet
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
        except Exception:
            traceback.print_exc()

    # ---------------- the run: record each step, process it in the background ----------------
    def _run(self, lo, hi, dwell):
        cfg0 = self.config.get()
        orig_sdr = copy.deepcopy(cfg0["sdr"])
        ring_was = cfg0["iq_ring"]["enabled"]
        eng = self.engine()
        fs = float(cfg0["sdr"]["sample_rate"])
        width = fs * USABLE
        n = max(1, int(np.ceil((hi - lo) / width)))
        hi = lo + n * width
        per = int(min(MAX_COLS, width / eng.df))
        name = f"{time.strftime('%Y-%m-%d_%H%M')}_{cfg0['sdr']['source']}_{lo / 1e6:.3f}-{hi / 1e6:.3f}MHz"
        res = {"lo": lo, "hi": hi, "dwell_s": dwell, "rows": ROWS, "step_hz": width, "cols_per_step": per,
               "source": cfg0["sdr"]["source"], "sample_rate": fs, "n_steps": n, "file": name + ".json",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
        self.result = res
        jobs, prog = queue.Queue(), {"rec": 0, "done": 0}
        worker = threading.Thread(target=self._worker, args=(jobs, res, cfg0, name, prog), daemon=True)
        worker.start()

        def status(msg):
            self._set(progress=(prog["rec"] + prog["done"]) / (2 * n), steps=n, message=msg)

        pending = None
        eng.on_hop = self._on_live_hop
        try:
            if not ring_was:                               # each step's IQ is recorded in the IQ ring
                self.apply({"iq_ring": {"enabled": True}})
            for i in range(n):
                if self._stop.is_set():
                    break
                s_lo = lo + i * width
                fc = s_lo + width / 2
                before = eng.session()
                target = {"center_hz": fc, "fmin_hz": s_lo, "fmax_hz": s_lo + width}
                if all(self.config.get()["sdr"].get(k) == v for k, v in target.items()):
                    eng.request_restart()                  # already there: restart anyway (a fresh session)
                else:
                    self.apply({"sdr": target})
                t_w = time.time()
                while eng.session() == before and time.time() - t_w < 15 and not self._stop.is_set():
                    time.sleep(0.2)
                if eng.session() == before:
                    raise RuntimeError("The receiver did not retune.")
                if pending:                                # the previous step's files are closed now
                    jobs.put(pending)
                    pending = None
                session, cfg_i = eng.session(), self.config.get()
                pv = {"i": i, "lo": s_lo, "hi": s_lo + width, "col_hz": width / per, "ncols": per, "dur": dwell + 1.0,
                      "t_a": session[1] + 1.5, "t_b": session[1] + 1.5 + dwell, "lag": 0.0,
                      "pow": np.zeros((ROWS, per)), "pow_n": np.zeros((ROWS, per)), "last_t": -1e18, "bins": None}
                with self._lock:
                    self.previews[i] = pv
                    self._preview_live = pv
                t_w = time.time()
                while not self._stop.is_set():             # record this step
                    paths, have = pick_segments(self.root, cfg_i, (dwell + 2) / 60, session)
                    status(f"Recording step {i + 1} of {n} ({s_lo / 1e6:.3f}–{(s_lo + width) / 1e6:.3f} MHz): "
                           f"{min(have, dwell):.0f} of {dwell:.0f} s" + self._worker_note(prog, n))
                    if have >= dwell + 1.0 or time.time() - t_w > dwell + 40:
                        break
                    time.sleep(0.5)
                prog["rec"] = i + 1
                self._preview_live = None
                if have >= 10:
                    pending = {"i": i, "lo": s_lo, "hi": s_lo + width, "fc": fc, "fs": fs, "paths": paths,
                               "dur": min(have, dwell + 1.0), "per": per}
        except Exception as e:
            traceback.print_exc()
            self._set(message=str(e))
        finally:
            eng.on_hop = None
            self._preview_live = None
            before = eng.session()
            self.apply({"sdr": orig_sdr})                  # back to where you were
            t_w = time.time()                              # the retune closes the last step's segment
            while eng.session() == before and time.time() - t_w < 15:
                time.sleep(0.2)
            if not ring_was:
                self.apply({"iq_ring": {"enabled": False}})
        if pending:
            jobs.put(pending)
        jobs.put(None)
        while worker.is_alive():
            status("Processing" + self._worker_note(prog, n))
            worker.join(1.0)
        stopped = self._stop.is_set()
        self.previews = {}
        self._set(state="done", progress=1.0, file=res["file"],
                  message=("Stopped" if stopped else "Done") + f": {len(res['steps'])} of {n} steps.")
        self.ended_at = time.time()

    def _worker_note(self, prog, n):
        w = self.state.get("worker")
        return f" · {w}" if w else ""

    def _worker(self, jobs, res, cfg0, name, prog):
        while True:
            job = jobs.get()
            if job is None:
                break
            try:
                res["steps"].append(self._process(job, res, cfg0, name))
                with self._lock:
                    self.previews.pop(job["i"], None)
                res["steps"].sort(key=lambda s: s["lo"])
                self._save(res, name)
                self._set(steps_done=len(res["steps"]))
            except Exception:
                traceback.print_exc()
            prog["done"] += 1
        self.state.pop("worker", None)

    # ---------------- one step: replay it, map its power ----------------
    def _process(self, job, res, cfg0, name):
        i, n = job["i"], res["n_steps"]
        tcfg = copy.deepcopy(cfg0)
        tcfg["sdr"].update(center_hz=job["fc"], sample_rate=job["fs"], fmin_hz=job["lo"], fmax_hz=job["hi"])  # this step's band
        st = {"lo": job["lo"], "col_hz": (job["hi"] - job["lo"]) / job["per"], "ncols": job["per"],
              "dur": job["dur"], "t_a": None, "t_b": None, "lag": 0.0, "pow": np.zeros((ROWS, job["per"])),
              "pow_n": np.zeros((ROWS, job["per"])), "last_t": -1e18, "bins": None}
        self._step = st
        self.state["worker"] = f"processing step {i + 1} of {n}"
        try:
            replay(self.root, tcfg, job["paths"], job["fs"], job["fc"], job["dur"], hooks={"on_hop": self._on_hop})
        finally:
            self._step = None
        with np.errstate(invalid="ignore", divide="ignore"):
            pw = st["pow"] / st["pow_n"]
        filled = st["pow_n"] > 0
        floor = float(np.median(pw[filled])) if filled.any() else -100.0
        if filled.any():
            colmean = np.nanmean(np.where(filled, pw, np.nan), axis=0)
            pw = np.where(filled, pw, np.nan_to_num(colmean, nan=floor))
        else:
            pw = np.full_like(pw, floor)
        return {"lo": job["lo"], "hi": job["hi"], "cols": job["per"], "fc": job["fc"], "fs": job["fs"],
                "iq_files": [Path(p).name for p in job["paths"]], "floor_db": floor, "pow": pw.astype(np.float32),
                "maps": {}, "tracks": [], "calibration": None,
                "dwell_s": round((st["t_b"] or 0) - (st["t_a"] or 0), 2)}

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
        out["views"] = views
        if res is self.result:
            with self._lock:
                out["previews"] = [{"i": p["i"], "lo": p["lo"], "hi": p["hi"], "cols": p["ncols"]} for p in self.previews.values()]
        return out

    def preview_image(self, i):
        """Power preview of a step still being recorded or processed: uint8 [rows, cols], or None."""
        with self._lock:
            p = self.previews.get(i)
            if p is None:
                return None
            n = p["pow_n"].copy()
            pw = p["pow"].copy()
        filled = n > 0
        if not filled.any():
            return np.zeros_like(pw, np.uint8)
        db = np.where(filled, pw / np.maximum(n, 1), np.nan)
        lo, hi = np.nanpercentile(db, 2), np.nanpercentile(db, 99.7)
        return np.where(filled, np.clip((db - lo) / max(1e-6, hi - lo) * 255, 1, 255), 0).astype(np.uint8)

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

    def _save(self, res, name):
        self.dir.mkdir(parents=True, exist_ok=True)
        m = self.meta(res)
        np.savez_compressed(self.dir / (name + ".npz"), **{v: self.image(v) for v in m["views"]})
        (self.dir / (name + ".json")).write_text(json.dumps(m, default=float))

    def track(self, file, track_id):
        """(step, track) of a saved or the current survey."""
        m = self.load(file) if file else self.meta()
        for st in (m or {}).get("steps", []):
            for t in st["tracks"]:
                if t["id"] == track_id:
                    return st, t
        return None, None

    def clip_path(self, file, clip):
        p = self.dir / file[:-5] / clip
        return p if p.is_file() else None

    def load(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.is_file() else None

    def saved(self):
        return sorted((p.name for p in self.dir.glob("*.json")), reverse=True) if self.dir.exists() else []

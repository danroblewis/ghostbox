"""Band survey: step the receiver across a wide range, calibrate on each step, map what it finds.

Recording and processing overlap. The receiver records `dwell_s` of raw IQ at each step (the IQ
ring) and moves on; a worker processes each finished step once:

1. replay the step's IQ through the detector with permissive candidate finding, keeping every
   candidate with all of its check scores and audio, and the maps (power, novelty, every check);
2. calibrate on those candidates (the same weight search as Calibration, judged by the ML voice
   detector) and re-evaluate them with the calibrated settings: confidence is a function of the
   check scores, so this is a recomputation, not a second replay. The combined-score map is
   rebuilt from the check maps with the calibrated weights;
3. keep the clips of what the calibrated detector shows and transcribe them with Whisper tiny.

The receiver goes back to where it was. Surveys are saved to live_data/surveys/<name>.json (+ .npz
maps, + <name>/ clips) after every step, and can be reopened.
"""
import copy
import json
import queue
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import stage1
from .autotune import (ENOUGH, SKIP, TUNE_DETECT, analyse_iq, analysis_config, calibrate_boxes,
                       pick_segments, rescore)
from .config import CHECKS

ROWS = 60                      # time rows per step (the map is wide and short)
MAX_COLS = 4000                # columns per step
USABLE = 0.8                   # fraction of the sample rate used per step (edges roll off)
MAX_CLIPS = 25                 # clips kept (and transcribed) per step
STAGE1_IDS = {c["id"] for c in CHECKS if c["stage"] == 1}


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
        self._asr = None

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

    def _on_maps(self, eng, ts, maps):
        st = self._step
        if st is None:
            return
        try:
            t_a, t_b = self._window(eng, st)
            sel = (ts >= t_a) & (ts < t_b)
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

    def _on_clip(self, tr, meta, pcm16):
        st = self._step
        if st is not None:
            st["clips"].setdefault(tr.id, []).append(pcm16)

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
               "source": cfg0["sdr"]["source"], "sample_rate": fs, "n_steps": n, "calibrated": True,
               "floor": float(cfg0["autotune"].get("floor", 0.3)), "file": name + ".json",
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}
        self.result = res
        jobs, prog = queue.Queue(), {"rec": 0, "done": 0}
        worker = threading.Thread(target=self._worker, args=(jobs, res, cfg0, name, prog), daemon=True)
        worker.start()

        def status(msg):
            self._set(progress=(prog["rec"] + prog["done"]) / (2 * n), steps=n, message=msg)

        pending = None
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
                t_w = time.time()
                while not self._stop.is_set():             # record this step
                    paths, have = pick_segments(self.root, cfg_i, (dwell + 2) / 60, session)
                    status(f"Recording step {i + 1} of {n} ({s_lo / 1e6:.3f}–{(s_lo + width) / 1e6:.3f} MHz): "
                           f"{min(have, dwell):.0f} of {dwell:.0f} s" + self._worker_note(prog, n))
                    if have >= dwell + 1.0 or time.time() - t_w > dwell + 40:
                        break
                    time.sleep(0.5)
                prog["rec"] = i + 1
                if have >= 10:
                    pending = {"i": i, "lo": s_lo, "hi": s_lo + width, "fc": fc, "fs": fs, "paths": paths,
                               "dur": min(have, dwell + 1.0), "per": per}
        except Exception as e:
            traceback.print_exc()
            self._set(message=str(e))
        finally:
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
                res["steps"].sort(key=lambda s: s["lo"])
                self._save(res, name)
                self._set(steps_done=len(res["steps"]))
            except Exception:
                traceback.print_exc()
            prog["done"] += 1
        self.state.pop("worker", None)

    # ---------------- one step: replay once, calibrate, re-evaluate, transcribe ----------------
    def _process(self, job, res, cfg0, name):
        i, n = job["i"], res["n_steps"]
        tcfg, score_all = analysis_config(cfg0)
        tcfg["sdr"].update(center_hz=job["fc"], sample_rate=job["fs"], fmin_hz=job["lo"], fmax_hz=job["hi"])  # this step's band
        score_all |= STAGE1_IDS - SKIP - ({"slide"} if not cfg0["checks"]["slide"]["enabled"] else set())
        boxes = []
        for ladder in (0, 2):                              # loosen once if the step is very quiet
            st = {"lo": job["lo"], "col_hz": (job["hi"] - job["lo"]) / job["per"], "ncols": job["per"],
                  "dur": job["dur"], "t_a": None, "t_b": None,
                  "lag": float(cfg0["analysis"]["settle_s"] + cfg0["analysis"]["hop_s"]), "pow": np.zeros((ROWS, job["per"])),
                  "pow_n": np.zeros((ROWS, job["per"])), "maps": {}, "clips": {}, "last_t": -1e18, "bins": None}
            self._step = st
            self.state["worker"] = f"processing step {i + 1} of {n}"
            try:
                boxes = analyse_iq(self.root, tcfg, score_all, job["paths"], job["fs"], job["fc"], job["dur"], ladder,
                                   hooks={"on_hop": self._on_hop, "on_maps": self._on_maps, "on_clip": self._on_clip})
            finally:
                self._step = None
            if len(boxes) >= ENOUGH:
                break
        # calibrate on this step's candidates, then re-evaluate them with the calibrated settings
        cal = None
        checks = copy.deepcopy(cfg0["checks"])
        show = float(cfg0["detect"]["threshold"])
        if len(boxes) >= 5:
            self.state["worker"] = f"calibrating step {i + 1} of {n}"
            cal = calibrate_boxes(boxes, cfg0, job["dur"] / 60, cfg0["autotune"]["per_min"], score_all - SKIP)
            for k, w in cal["weights"].items():
                checks[k] = {**checks[k], "enabled": w > 0, "weight": w}
            show = cal["show_threshold"]
        t_a = st["t_a"] or 0.0
        sess0 = t_a - 1.0                                  # the replay's timeline starts with the first file
        tracks = []
        for b in boxes:
            conf = rescore(b, checks) if cal else float(b.get("conf_max", b.get("conf", 0)))
            if b["t1"] - b["t0"] < 0.3:
                continue
            tracks.append({"id": b["id"], "f": float(np.mean([p[1] for p in b["path"]])) if b.get("path") else (b["fmin"] + b["fmax"]) / 2,
                           "fmin": b["fmin"], "fmax": b["fmax"],
                           "t0": round(max(0.0, b["t0"] - t_a), 2), "t1": round(max(0.0, b["t1"] - t_a), 2),
                           "conf": round(conf, 3), "vad": b["scores"].get("vad"), "clips": [], "text": None,
                           "iq_t": [round(b["t0"] - sess0, 3), round(b["t1"] - sess0, 3)],
                           "scores": {k: round(float(v), 3) for k, v in b["scores"].items()},
                           "info": b.get("info", {}), "dur": round(b["t1"] - b["t0"], 2)})
        # keep (and transcribe) the clips of what the calibrated detector shows
        shown = sorted([t for t in tracks if t["conf"] >= show and st["clips"].get(t["id"])], key=lambda t: -t["conf"])[:MAX_CLIPS]
        cdir = self.dir / name
        cdir.mkdir(parents=True, exist_ok=True)
        audio = []
        for t in shown:
            pcm = np.concatenate(st["clips"][t["id"]])
            fn = f"s{i + 1}_{t['id']}.wav"
            wavfile.write(cdir / fn, 16000, pcm)
            t["clips"] = [f"/api/survey/clip?file={name}.json&clip={fn}"]
            audio.append(pcm.astype(np.float32) / 32768)
        if audio:
            self.state["worker"] = f"transcribing step {i + 1} of {n}"
            texts = self._transcribe(audio)
            for t, tx in zip(shown, texts):
                t["text"] = tx
        # maps: power, and the combined score rebuilt with the calibrated weights
        with np.errstate(invalid="ignore", divide="ignore"):
            pw = st["pow"] / st["pow_n"]
        filled = st["pow_n"] > 0
        floor = float(np.median(pw[filled])) if filled.any() else -100.0
        if filled.any():
            colmean = np.nanmean(np.where(filled, pw, np.nan), axis=0)
            pw = np.where(filled, pw, np.nan_to_num(colmean, nan=floor))
        else:
            pw = np.full_like(pw, floor)
        maps = st["maps"]
        if cal:
            parts = {k: maps[k].astype(np.float32) / 255 for k in maps if k in STAGE1_IDS}
            g = stage1.combine(parts, checks)
            if g is not None:
                maps["score"] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
        summary = None
        if cal:
            summary = {"candidates": len(boxes), "voice_like": cal["voice_like"], "kept_checks": bool(cal.get("kept_checks")),
                       "show_threshold": show, "weights": {k: w for k, w in cal["weights"].items() if w > 0}}
        return {"lo": job["lo"], "hi": job["hi"], "cols": job["per"], "fc": job["fc"], "fs": job["fs"],
                "iq_files": [Path(p).name for p in job["paths"]], "floor_db": floor, "pow": pw.astype(np.float32),
                "maps": maps, "tracks": sorted(tracks, key=lambda t: t["t0"]), "calibration": summary,
                "dwell_s": round((st["t_b"] or 0) - (st["t_a"] or 0), 2)}

    def _transcribe(self, audio):
        try:
            from .asr import load
            from .transcribe import strip_stock
            if self._asr is None:
                self._asr = load("tiny", cpu_threads=4, workers=1)
            phrases = self.config.get()["whisper"].get("filter_phrases", [])
            out = []
            for k in range(0, len(audio), 8):
                out += [strip_stock(tx, phrases) or None for tx, _ in self._asr.transcribe(audio[k:k + 8], "en")]
            return out
        except Exception:
            traceback.print_exc()
            return [None] * len(audio)

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

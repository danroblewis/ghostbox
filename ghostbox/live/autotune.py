"""Auto-tune: calibrate the detector to the current band from recorded IQ.

Ghost signals can't be labelled, but the band's background can be measured. Auto-tune:

1. takes the newest few minutes of continuous raw IQ for the current band (the IQ ring);
2. runs the detector over it once, unthrottled, with permissive thresholds so that every
   candidate is kept along with all of its check scores;
3. searches check-weight combinations offline (confidence is just a weighted geometric mean of
   those scores, so trying hundreds costs milliseconds), scoring each by how speech-like its
   top detections are at the requested rate, judged by the ML voice detector (whose own weight
   is held fixed, so it can't vote for itself);
4. sets the check weights, a capture floor that keeps ~3x the target rate in memory, and
   reports the display threshold that shows the target rate.

The result puts the detector at the edge of what this band's background allows: on a quiet
band that means much more sensitive settings than the defaults.
"""
import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

from . import stage1
from .config import CHECKS
from .engine import Engine
from .sources import FileSource

TUNE_DETECT = {"candidate_threshold": 0.15, "threshold": 0.0, "min_duration_s": 0.3,
               "max_candidates": 12, "max_tracks": 30, "novelty": 0.5, "prominence": 0.3}


class StaticConfig:
    def __init__(self, cfg):
        self.cfg = cfg

    def get(self):
        return copy.deepcopy(self.cfg)


def pick_segments(root, cfg, minutes):
    """Newest continuous run of IQ-ring segments recorded at the current frequency and rate."""
    d = Path(root) / cfg["iq_ring"]["dir"]
    fc, fs = float(cfg["sdr"]["center_hz"]), float(cfg["sdr"]["sample_rate"])
    segs = []
    for j in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            sj = json.loads(j.read_text())
        except Exception:
            continue
        p = j.with_suffix(".cs8")
        if p.exists() and abs(sj["fc"] - fc) < 1 and abs(sj["fs"] - fs) < 1:
            segs.append((sj, p, p.stat().st_size / 2 / sj["fs"]))
    chosen, total = [], 0.0
    for sj, p, dur in reversed(segs):                   # newest first, stop at a gap or epoch change
        if chosen:
            nxt = chosen[0][0]
            if abs(sj["epoch"] - nxt["epoch"]) > 1e-3 or nxt["t0"] - (sj["t0"] + dur) > 2.0:
                break
        chosen.insert(0, (sj, p, dur))
        total += dur
        if total >= minutes * 60:
            break
    if chosen and chosen[-1][2] < 1.0:                  # the segment being written may be tiny
        total -= chosen[-1][2]
        chosen = chosen[:-1]
    return [p for _, p, _ in chosen], total


def optimise(boxes, minutes, per_min, live_checks, n_random=600, seed=0, available=None):
    ids = [c["id"] for c in CHECKS if c["id"] != "vad" and (available is None or c["id"] in available)]
    X = np.array([[b["scores"].get(i, np.nan) for i in ids] for b in boxes], dtype=float)
    X = np.where(np.isnan(X), 0.5, X)                   # a check that couldn't run is neutral
    cols = {i: X[:, j] for j, i in enumerate(ids)}

    def conf_of(w, extra=None, extra_w=0.0):
        """Confidence for weight vector w (AM/SSB families as alternatives, as live)."""
        chk = {i: {"enabled": wi > 0, "weight": float(wi)} for i, wi in zip(ids, w)}
        sc = dict(cols)
        if extra is not None and extra_w > 0:
            chk["vad"], sc["vad"] = {"enabled": True, "weight": extra_w}, extra
        g = stage1.combine(sc, chk)
        return np.zeros(len(boxes)) if g is None else g
    judge = np.array([b["scores"].get("vad", np.nan) for b in boxes], dtype=float)
    if np.all(np.isnan(judge)):                         # no ML judge available: fall back to the pitch comb
        judge = X[:, ids.index("comb")]
    judge = np.nan_to_num(judge)
    n = len(boxes)
    K = int(max(3, min(n, round(per_min * minutes))))

    def evaluate(w):
        conf = conf_of(w)
        top = np.argsort(-conf)[:K]
        return float(judge[top].mean())

    current = np.array([live_checks[i]["weight"] if live_checks[i]["enabled"] else 0.0 for i in ids])
    rng = np.random.default_rng(seed)
    cands = [current, np.ones(len(ids))]
    cands += [rng.choice([0.0, 0.5, 1.0, 2.0], size=len(ids)) for _ in range(n_random)]
    cands = [w for w in cands if w.sum() > 0]
    scores = [evaluate(w) for w in cands]
    best = int(np.argmax(scores))
    w_best = cands[best]

    # final confidence as the live detector will compute it (including the ML check at its own weight)
    vad = live_checks["vad"]
    w_vad = vad["weight"] if vad["enabled"] else 0.0
    vad_s = np.nan_to_num(np.array([b["scores"].get("vad", 0.5) for b in boxes], float), nan=0.5)
    conf_full = conf_of(w_best, vad_s, w_vad)
    order = np.sort(conf_full)[::-1]
    show = float(order[K - 1])
    capture = float(order[min(n, 3 * K) - 1])
    return {
        "weights": {i: float(w) for i, w in zip(ids, w_best)},
        "target_count": K,
        "score_before": evaluate(current) if current.sum() > 0 else 0.0,
        "score_after": float(scores[best]),
        "show_threshold": round(show, 3),
        "capture_threshold": round(max(0.0, capture - 0.01), 3),
        "conf_median": float(np.median(conf_full)),
    }


class AutoTune:
    def __init__(self, root, config, emit, apply):
        self.root, self.config, self.emit, self.apply = Path(root), config, emit, apply
        self.state = {"state": "idle"}
        self.prev = None
        self._lock = threading.Lock()

    def _set(self, **kw):
        self.state = {**self.state, **kw}
        self.emit({"type": "autotune", **self.state})

    def start(self, minutes, per_min):
        with self._lock:
            if self.state.get("state") == "running":
                return False
            self.state = {"state": "running", "progress": 0.0, "message": "Finding recorded IQ…"}
        threading.Thread(target=self._run, args=(float(minutes), float(per_min)), daemon=True).start()
        return True

    def revert(self):
        if not self.prev:
            return False
        self.apply(self.prev)
        self.prev = None
        self._set(state="reverted", message="Previous detector settings restored.")
        return True

    def _run(self, minutes, per_min):
        try:
            cfg = self.config.get()
            if cfg["sdr"]["source"] == "file":
                raise RuntimeError("Auto-tune works on live receivers (it needs IQ recorded at the current frequency).")
            paths, dur = pick_segments(self.root, cfg, minutes)
            if dur < 30:
                raise RuntimeError("Not enough raw IQ recorded at this frequency yet. Turn on Settings > Storage > "
                                   f"Record raw IQ, stay on this band for {max(1, int(minutes))} min, then try again.")
            fc, fs = float(cfg["sdr"]["center_hz"]), float(cfg["sdr"]["sample_rate"])
            tcfg = copy.deepcopy(cfg)
            tcfg["detect"].update(TUNE_DETECT)
            for c in CHECKS:                            # compute every check; candidate search uses stage 1
                cc = tcfg["checks"][c["id"]]
                if c["id"] == "slide" and not cc["enabled"]:
                    continue                            # several times the CPU: only if you use it
                if c["id"] == "profile":
                    continue                            # learned live, not from a few minutes of replay
                cc["enabled"], cc["weight"] = True, max(cc["weight"], 0.5)
            tcfg["whisper"]["enabled"] = False
            tcfg["record"]["enabled"] = False
            tcfg["iq_ring"]["enabled"] = False
            boxes = []

            def collect(m):
                if m.get("type") == "det_end":
                    boxes.append(m["box"])
                elif m.get("type") == "status" and m.get("progress") is not None:
                    self._set(progress=round(0.9 * m["progress"], 3),
                              message=f"Analysing {dur / 60:.1f} min of recorded IQ… {len(boxes)} candidates")

            self._set(message=f"Analysing {dur / 60:.1f} min of recorded IQ…")
            src = FileSource(paths, fs, fc, paced=False, loop=False)
            eng = Engine(StaticConfig(tcfg), collect, self.root, source=src, oneshot=True, transcribe=False)
            t_start = time.time()
            while not eng.done.wait(1.0):
                if time.time() - t_start > dur * 4 + 120:
                    eng.stop()
                    raise RuntimeError("Analysis took too long; try fewer minutes.")
            if len(boxes) < 5:
                raise RuntimeError(f"Only {len(boxes)} candidate signals in {dur / 60:.1f} min: this band is nearly silent "
                                   "even at the most permissive settings.")
            self._set(progress=0.95, message=f"Searching settings over {len(boxes)} candidates…")
            avail = {c["id"] for c in CHECKS if tcfg["checks"][c["id"]]["enabled"]}
            res = optimise(boxes, dur / 60, per_min, cfg["checks"], available=avail)
            self.prev = {"checks": copy.deepcopy(cfg["checks"]), "detect": copy.deepcopy(cfg["detect"])}
            live_detect = {**TUNE_DETECT, "max_tracks": 16, "max_candidates": 8,   # the analysis pass is wide open;
                           "threshold": res["capture_threshold"]}                      # live keeps CPU in check
            patch = {"checks": {i: {"weight": w, "enabled": w > 0} for i, w in res["weights"].items()},
                     "detect": live_detect}
            self.apply(patch)
            band = [fc - fs / 2, fc + fs / 2]
            if cfg["sdr"].get("fmin_hz") is not None:
                band = [cfg["sdr"]["fmin_hz"], cfg["sdr"]["fmax_hz"]]
            self._set(state="done", progress=1.0, result={
                **res, "minutes": round(dur / 60, 2), "candidates": len(boxes),
                "candidates_per_min": round(len(boxes) / (dur / 60), 1), "per_min": per_min, "band": band},
                message="Applied.")
        except Exception as e:
            self._set(state="error", message=str(e), progress=None)

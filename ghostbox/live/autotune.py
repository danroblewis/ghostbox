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

# the analysis pass is permissive (keep every candidate) but tracks normally whatever is set live
TUNE_DETECT = {"candidate_threshold": 0.15, "threshold": 0.0, "min_duration_s": 0.3,
               "max_candidates": 12, "max_tracks": 30, "novelty": 0.5, "prominence": 0.3,
               "track_gap_s": 0.3, "max_slide_hz_s": 100e3, "min_sep_hz": 6e3, "link_hops_s": 0.4}
LEVELS = (0.0, 0.5, 1.0, 2.0)
# If a band gives too few candidates to learn from, loosen candidate finding step by step.
LADDER = (
    {},
    {"candidate_threshold": 0.10, "prominence": 0.15, "novelty": 0.3},
    {"candidate_threshold": 0.06, "prominence": 0.0, "novelty": 0.15},
    {"candidate_threshold": 0.03, "prominence": 0.0, "novelty": 0.0, "min_duration_s": 0.2},
)
ENOUGH = 10                          # candidates needed to tune on
# Switched off only if that strictly improves things (a tie keeps them): the modulation checks decide
# which kinds of signal can be found at all, and co-modulation / hop continuity only matter in the
# moments they apply, which a few minutes of recording may not contain.
PROTECT = {c["id"] for c in CHECKS if c.get("family")} | {"comod", "hop"}
SKIP = {"vad", "profile"}            # the judge itself; the profile is learned live, not from a replay


class StaticConfig:
    def __init__(self, cfg):
        self.cfg = cfg

    def get(self):
        return copy.deepcopy(self.cfg)


def pick_segments(root, cfg, minutes, since=None):
    """Newest continuous run of IQ-ring segments recorded at the current frequency and rate.
    since=(epoch, t0): only segments recorded after that moment (e.g. the last receiver change)."""
    d = Path(root) / cfg["iq_ring"]["dir"]
    fc, fs = float(cfg["sdr"]["center_hz"]), float(cfg["sdr"]["sample_rate"])
    segs = []
    for j in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            sj = json.loads(j.read_text())
        except Exception:
            continue
        p = j.with_suffix(".cs8")
        if since is not None and (abs(sj.get("epoch", 0) - since[0]) > 1e-3 or sj["t0"] < since[1] - 0.5):
            continue
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


def optimise(boxes, minutes, per_min, live_checks, floor=0.3, available=None, keep=False):
    """Choose check weights by coordinate descent, judged by the ML voice detector.

    K = how many detections to aim for: per_min x minutes, but never more than the candidates the
    judge finds at least `floor` speech-like (so on a dead band it aims for nothing, not junk).
    Objective: mean judge score of the K most confident candidates. Starting with every check
    on, each check in turn is tried at weights 0 / 0.5 / 1 / 2 with the others fixed; a check
    stays off (weight 0) when that's at least as good, otherwise it keeps its best weight.
    A few passes, from the current weights and from all-on; the better result wins."""
    ids = [c["id"] for c in CHECKS if c["id"] not in SKIP and (available is None or c["id"] in available)]
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
        judge = X[:, ids.index("comb")] if "comb" in ids else np.zeros(len(boxes))
    judge = np.nan_to_num(judge)
    n = len(boxes)
    voicey = int((judge >= floor).sum())
    K = int(min(n, max(1, round(per_min * minutes)), max(1, voicey)))

    def evaluate(w):
        if w.sum() <= 0:
            return -1.0
        top = np.argsort(-conf_of(w))[:K]
        return float(judge[top].mean())

    def descend(w):
        w = w.copy()
        best = evaluate(w)
        for _ in range(3):
            changed = False
            for j in range(len(ids)):
                trial = {}
                for lv in LEVELS:
                    v = w.copy()
                    v[j] = lv
                    trial[lv] = evaluate(v)
                top = max(trial.values())
                keep = trial.get(w[j], -2) >= top - 0.005
                if ids[j] in PROTECT:
                    # off only if strictly better; otherwise keep the current weight on a tie
                    pick = w[j] if keep else max(trial, key=trial.get)
                else:
                    # prefer off (saves CPU), then the current weight, when they're as good as the best
                    pick = 0.0 if trial[0.0] >= top - 0.005 else (w[j] if keep else max(trial, key=trial.get))
                if pick != w[j]:
                    w[j], changed = pick, True
                best = evaluate(w)
            if not changed:
                break
        return w, best

    current = np.array([live_checks[i]["weight"] if live_checks[i]["enabled"] else 0.0 for i in ids])
    if keep and current.sum() > 0:                      # rank with the current checks, no search
        w_best, best_score = current, evaluate(current)
    else:
        runs = [descend(np.ones(len(ids)))]
        if current.sum() > 0:
            runs.append(descend(np.maximum(current, 0)))
        w_best, best_score = max(runs, key=lambda r: r[1])

    # what each check contributed: the objective with it switched off vs as chosen
    decisions = []
    for j, i in enumerate(ids):
        off = w_best.copy()
        off[j] = 0.0
        decisions.append({"id": i, "weight": float(w_best[j]), "on": bool(w_best[j] > 0),
                          "effect": round(best_score - evaluate(off), 3)})
    decisions.sort(key=lambda d: (-d["on"], -d["effect"]))

    # final confidence as the live detector will compute it (including the ML check at its own weight)
    vad = live_checks["vad"]
    w_vad = vad["weight"] if vad["enabled"] else 0.0
    vad_s = np.nan_to_num(np.array([b["scores"].get("vad", 0.5) for b in boxes], float), nan=0.5)
    conf_full = conf_of(w_best, vad_s, w_vad)
    order = np.sort(conf_full)[::-1]
    # nothing (much) speech-like: still show the best per_min of what there is, rather than nothing
    K_show = K if voicey >= 3 else int(min(n, max(1, round(per_min * minutes))))
    show = float(order[K_show - 1])
    capture = float(order[min(n, 3 * K) - 1])
    return {
        "weights": {i: float(w) for i, w in zip(ids, w_best)},
        "decisions": decisions,
        "target_count": K, "voice_like": voicey,
        "score_before": evaluate(current) if current.sum() > 0 else 0.0,
        "score_after": float(best_score),
        "show_threshold": round(min(1.0, show), 3),
        "capture_threshold": round(max(0.0, min(show, capture) - 0.01), 3),
        "conf_median": float(np.median(conf_full)),
    }


class AutoTune:
    def __init__(self, root, config, emit, apply, session=lambda: None):
        self.root, self.config, self.emit, self.apply = Path(root), config, emit, apply
        self.session = session                          # () -> (epoch, t0) of the last receiver change
        self.state = {"state": "idle"}
        self.prev = None
        self.auto_band = None                           # receiver settings of the last tune (any outcome)
        self.last_run = 0.0
        self.last_end = 0.0                             # when the last run finished (any outcome)
        self.show_conf = None                           # display threshold the last successful run chose
        self._lock = threading.Lock()

    def _set(self, **kw):
        self.state = {**self.state, **kw}
        self.emit({"type": "autotune", **self.state})

    def start(self, minutes, per_min, auto=False):
        with self._lock:
            if self.state.get("state") == "running":
                return False
            self.state = {"state": "running", "progress": 0.0, "auto": auto,
                          "message": ("Automatic: " if auto else "") + "Finding recorded IQ…"}
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
            s_ = cfg["sdr"]
            self.auto_band = json.dumps({k: v for k, v in s_.items() if k != "file"}, sort_keys=True)
            self.last_run = time.time()
            paths, dur = pick_segments(self.root, cfg, minutes, self.session())
            if dur < 30:
                raise RuntimeError("Not enough raw IQ recorded at this frequency yet. Turn on Settings > Storage > "
                                   f"Record raw IQ, stay on this band for {max(1, int(minutes))} min, then try again.")
            fc, fs = float(cfg["sdr"]["center_hz"]), float(cfg["sdr"]["sample_rate"])
            tcfg = copy.deepcopy(cfg)
            tcfg["detect"].update(TUNE_DETECT)
            # Candidates are found with the checks as they are now (switching everything on flattens
            # the combined score so that nothing stands out); every check is still computed for each
            # candidate so the search below can try any of them.
            score_all = {c["id"] for c in CHECKS if c["id"] not in SKIP
                         and (c["id"] != "slide" or tcfg["checks"]["slide"]["enabled"])} | {"vad"}
            if not any(tcfg["checks"][c["id"]]["enabled"] for c in CHECKS if c["stage"] == 1 and c["id"] != "profile"):
                for k in ("twinning", "syllabic", "comb"):       # nothing to find candidates with: a basic set
                    tcfg["checks"][k]["enabled"] = True
            tcfg["checks"]["profile"]["enabled"] = False    # learned live; a replay mustn't feed it
            tcfg["whisper"]["enabled"] = False
            tcfg["record"]["enabled"] = False
            tcfg["iq_ring"]["enabled"] = False
            def analyse(step):
                boxes = []
                t = copy.deepcopy(tcfg)
                t["detect"].update(LADDER[step])
                pre = f"Analysing {dur / 60:.1f} min of recorded IQ" + (f" (looser, step {step + 1}/{len(LADDER)})" if step else "")

                def collect(m):
                    if m.get("type") == "det_end":
                        boxes.append(m["box"])
                    elif m.get("type") == "status" and m.get("progress") is not None:
                        self._set(progress=round(0.9 * (step + m["progress"]) / len(LADDER), 3),
                                  message=f"{pre}… {len(boxes)} candidates")

                self._set(message=pre + "…")
                src = FileSource(paths, fs, fc, paced=False, loop=False)
                eng = Engine(StaticConfig(t), collect, self.root, source=src, oneshot=True, transcribe=False,
                             score_all=score_all)
                t_start = time.time()
                while not eng.done.wait(1.0):
                    if time.time() - t_start > dur * 4 + 120:
                        eng.stop()
                        raise RuntimeError("Analysis took too long; try fewer minutes.")
                return boxes

            for step in range(len(LADDER)):
                boxes = analyse(step)
                if len(boxes) >= ENOUGH:
                    break
            found = {**TUNE_DETECT, **LADDER[step]}
            if len(boxes) < 5:
                # still too little to learn weights from: keep the checks, but apply working tracking
                # and the loosest candidate finding so the live detector has its best chance
                self.prev = {"checks": copy.deepcopy(cfg["checks"]), "detect": copy.deepcopy(cfg["detect"])}
                self.apply({"detect": {**found, "max_tracks": 16, "max_candidates": 8, "threshold": 0.0}})
                self._set(state="quiet", progress=1.0,
                          message=f"Only {len(boxes)} candidate signal{'s' if len(boxes) != 1 else ''} in {dur / 60:.1f} min "
                                  "even at the loosest settings. Check weights unchanged; tracking and candidate "
                                  "finding set as loose as they go.")
                return
            self._set(progress=0.95, message=f"Searching settings over {len(boxes)} candidates…")
            avail = score_all - SKIP
            res = optimise(boxes, dur / 60, per_min, cfg["checks"], floor=cfg["autotune"].get("floor", 0.3), available=avail)
            self.prev = {"checks": copy.deepcopy(cfg["checks"]), "detect": copy.deepcopy(cfg["detect"])}
            if res["voice_like"] < 3:
                # nothing (or almost nothing) speech-like to learn from: weights chosen against noise would
                # be arbitrary, so keep the current checks and only fix tracking and candidate finding;
                # the thresholds then come from the current checks' ranking
                res = optimise(boxes, dur / 60, per_min, {**cfg["checks"]}, floor=cfg["autotune"].get("floor", 0.3),
                               available=avail, keep=True)
                res["decisions"], res["kept_checks"] = [], True
            live_detect = {**found, "max_tracks": 16, "max_candidates": 8,   # the analysis pass is wide open;
                           "threshold": res["capture_threshold"]}                      # live keeps CPU in check
            labels = {c["id"]: c["label"] for c in CHECKS}
            for d in res["decisions"]:
                d["label"] = labels.get(d["id"], d["id"])
            patch = {"checks": {i: {"weight": w, "enabled": w > 0} for i, w in res["weights"].items()},
                     "detect": live_detect}
            self.apply(patch)
            band = [fc - fs / 2, fc + fs / 2]
            if cfg["sdr"].get("fmin_hz") is not None:
                band = [cfg["sdr"]["fmin_hz"], cfg["sdr"]["fmax_hz"]]
            self.show_conf = res["show_threshold"]
            self._set(state="done", progress=1.0, result={
                **res, "minutes": round(dur / 60, 2), "candidates": len(boxes),
                "candidates_per_min": round(len(boxes) / (dur / 60), 1), "per_min": per_min, "band": band},
                message="Applied.")
        except Exception as e:
            self._set(state="error", message=str(e), progress=None)
        finally:
            self.last_end = time.time()

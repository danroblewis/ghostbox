"""Calibration, kept as a shell while the filters are rebuilt.

What stays is the plumbing a calibration will need: finding the newest continuous run of recorded
IQ for the current band (the IQ ring), replaying recorded IQ through the engine (also used by band
surveys), and a run with progress and a result for the page. There is nothing to calibrate yet, so
a run only checks that enough IQ is recorded and says so.
"""
import copy
import json
import threading
import time
from pathlib import Path

from .engine import Engine
from .sources import FileSource


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


def replay(root, cfg, paths, fs, fc, dur, hooks=None):
    """Run the engine once over recorded IQ (unthrottled, nothing written). hooks: engine attributes
    to set first (e.g. on_hop)."""
    t = copy.deepcopy(cfg)
    t["iq_ring"]["enabled"] = False
    t["whisper"]["enabled"] = False
    src = FileSource(paths, fs, fc, paced=False, loop=False)
    eng = Engine(StaticConfig(t), lambda m: None, root, source=src, oneshot=True, transcribe=False)
    for k, v in (hooks or {}).items():
        setattr(eng, k, v)
    t_start = time.time()
    while not eng.done.wait(1.0):
        if time.time() - t_start > dur * 4 + 120:
            eng.stop()
            raise RuntimeError("Replay took too long.")
    return eng


class AutoTune:
    def __init__(self, root, config, emit, session=lambda: None):
        self.root, self.config, self.emit = Path(root), config, emit
        self.session = session                          # () -> (epoch, t0) of the last receiver change
        self.state = {"state": "idle"}
        self._lock = threading.Lock()

    def _set(self, **kw):
        self.state = {**self.state, **kw}
        self.emit({"type": "autotune", **self.state})

    def start(self, minutes):
        with self._lock:
            if self.state.get("state") == "running":
                return False
            self.state = {"state": "running", "progress": 0.0, "message": "Finding recorded IQ…"}
        threading.Thread(target=self._run, args=(float(minutes),), daemon=True).start()
        return True

    def _run(self, minutes):
        try:
            cfg = self.config.get()
            if cfg["sdr"]["source"] == "file":
                raise RuntimeError("Calibration works on live receivers (it needs IQ recorded at the current frequency).")
            paths, dur = pick_segments(self.root, cfg, minutes, self.session())
            if dur < 30:
                raise RuntimeError("Not enough raw IQ recorded at this frequency yet. Turn on ⚙ › Receiver & storage > "
                                   f"Record raw IQ, stay on this band for {max(1, int(minutes))} min, then try again.")
            self._set(state="done", progress=1.0, result={"minutes": round(dur / 60, 2), "files": len(paths)},
                      message=f"{dur / 60:.1f} min of recorded IQ found. There are no filters yet, so nothing to calibrate.")
        except Exception as e:
            self._set(state="error", message=str(e), progress=None)

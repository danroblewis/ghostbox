"""Live-engine configuration: receiver, waterfall, the pipeline, storage, transcription.

There are no filters or detector settings: the filters are being rebuilt one at a time. Keys a
saved config still has from before (checks, filters, detector, composed voice, ...) are dropped
when it is loaded.
"""
import copy
import json
import threading
from pathlib import Path

DEFAULT = {
    "sdr": {
        "source": "hackrf",                 # hackrf | rtlsdr | file
        "device": "",                       # which one if several are connected: HackRF serial / RTL-SDR index or serial ("" = first)
        "center_hz": 2.2e6,
        "sample_rate": 4e6,
        "fmin_hz": 520e3,                   # analysis/display crop (absolute Hz); null = whole capture
        "fmax_hz": 1720e3,
        "gain_db": 36,                      # one gain for both radios (HackRF: split into LNA + VGA)
        "amp": True,                        # HackRF RF amplifier (+14 dB); ignored by RTL-SDR
        "ppm": 0,                           # RTL-SDR crystal correction
        "file": {"path": "captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8"},
    },
    "analysis": {"bin_hz": 125, "frame_rate": 25, "hop_s": 1.0,
                 "settle_s": 1.5},              # how far behind live the waterfall is shown (s)
    # Frequency ranges marked on the waterfall (known stations, steady lines): [{"lo": Hz, "hi": Hz, "label": ""}]
    "blocked": [],
    # The pipeline (pipeline.py): IQ steps in order, [{"type", "on", "params"}], then the sum of everything
    # as audio. No step types exist yet.
    "pipeline": [],
    # What you hear: the pipeline's audio, played delay_s behind live
    "audio": {"delay_s": 4.0, "fade_s": 0.15,
              "preroll_s": 0.5, "postroll_s": 0.5}, # a saved recording taken again from the IQ ring: margins
    "record": {"dir": "recordings", "max_mb": 2000},   # saved recordings (from before); oldest deleted beyond max_mb
    "iq_ring": {"enabled": False, "dir": "iq_ring", "max_gb": 4.0, "segment_s": 10,
                "record_replays": False},       # continuous raw IQ, oldest segments deleted beyond max_gb
    "storage": {"min_free_gb": 10.0},           # disk guard: below this, trim the IQ ring, then pause writing
    "autotune": {"minutes": 2},                 # calibration: how much recorded IQ it would use
    "display": {"width": 1200, "dyn_range_db": 60.0, "history_min": 20},
    "whisper": {
        "enabled": True,
        "model": "base",                    # an id from asr.MODELS (Whisper, Moonshine, wav2vec2, Vosk)
        "language": "auto",                 # auto or an ISO code (en, es, ...)
        "workers": 2,                       # worker threads sharing one resident model
        "batch_size": 16,                   # clips per batch (each in its own 30 s window)
        "cpu_threads": 8,
        "max_queue": 64,                    # beyond this the least confident clip is dropped
        "max_age_s": 120,                   # clips waiting longer than this are skipped
        "filter_phrases": [                 # removed wherever they appear (on top of the built-in list)
            "Thank you very much for your continued support.", "dragonball", "Piaw", "Click on the top",
            "discontinue eligibility", "I'm gonna be the one who's gonna be the one"],
    },
}

VIEWS = [{"id": "power", "label": "Power (dB)"}]

PRESETS = {
    "am_hackrf": {"label": "Medium wave 520-1720 kHz (HackRF)",
                  "sdr": {"source": "hackrf", "center_hz": 2.2e6, "sample_rate": 4e6,
                          "fmin_hz": 520e3, "fmax_hz": 1720e3, "gain_db": 36, "amp": True}},
    "air_hackrf": {"label": "Airband 118-122 MHz (HackRF)",
                   "sdr": {"source": "hackrf", "center_hz": 120.0e6, "sample_rate": 4e6,
                           "fmin_hz": 118.0e6, "fmax_hz": 121.9e6, "gain_db": 48, "amp": True}},
    "fm_hackrf": {"label": "FM band 96-104 MHz (HackRF, ghosts only)",
                  "sdr": {"source": "hackrf", "center_hz": 100.0e6, "sample_rate": 8e6,
                          "fmin_hz": 96.2e6, "fmax_hz": 103.8e6, "gain_db": 24, "amp": False}},
    "air_rtl": {"label": "Airband 119.5 MHz (RTL-SDR)",
                "sdr": {"source": "rtlsdr", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                        "fmin_hz": None, "fmax_hz": None, "gain_db": 29.7}},
    "file_inj": {"label": "Replay: FM capture (test file)",
                 "sdr": {"source": "file", "center_hz": 96.6e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8"}}},
    "file_air": {"label": "Replay: airband capture",
                 "sdr": {"source": "file", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/air_119.5MHz_2.4Msps_g29.7.cu8"}}},
    "file_am": {"label": "Replay: medium-wave capture (HackRF)",
                "sdr": {"source": "file", "center_hz": 2.2e6, "sample_rate": 4e6,
                        "fmin_hz": 520e3, "fmax_hz": 1720e3,
                        "file": {"path": "captures/am_2.2MHz_4Msps_hackrf_a1l16g20.cs8"}}},
}


def hackrf_gains(gain_db):
    """Split one gain figure into HackRF LNA (0-40, step 8) + VGA (0-62, step 2)."""
    g = max(0.0, float(gain_db))
    lna = int(min(40, 8 * int(g / 2 / 8)))
    vga = int(min(62, max(0, 2 * round((g - lna) / 2))))
    return lna, vga


def prune_to(defaults, cfg):
    """Drop keys cfg has that defaults doesn't (recursively, where the default is a non-empty dict)."""
    for k in [k for k in cfg if k not in defaults]:
        del cfg[k]
    for k, d in defaults.items():
        if isinstance(d, dict) and d and isinstance(cfg.get(k), dict):
            prune_to(d, cfg[k])
    return cfg


def deep_merge(base, patch):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class Config:
    """Thread-safe config holder persisted to JSON."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cfg = copy.deepcopy(DEFAULT)
        if self.path.exists():
            try:
                deep_merge(self._cfg, json.loads(self.path.read_text()))
            except Exception:
                pass
        prune_to(DEFAULT, self._cfg)
        self.version = 0

    def get(self):
        with self._lock:
            return copy.deepcopy(self._cfg)

    def update(self, patch):
        with self._lock:
            before = copy.deepcopy(self._cfg["sdr"])
            deep_merge(self._cfg, patch)
            prune_to(DEFAULT, self._cfg)
            self.version += 1
            sdr_changed = before != self._cfg["sdr"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._cfg, indent=1))
            return copy.deepcopy(self._cfg), sdr_changed

    def reset(self):
        return self.update(copy.deepcopy(DEFAULT))

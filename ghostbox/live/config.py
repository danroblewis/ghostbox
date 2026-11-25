"""Live-engine configuration and the registry of detection checks.

Every check has an on/off switch, a weight and its own parameters. The UI is
generated from CHECKS, so adding a check here (plus its implementation in
stage1.py / stage2.py) is all it takes to expose it.

Confidence is a weighted geometric mean of the enabled checks' scores (each in
[0, 1]):  exp( sum w_i log max(s_i, eps) / sum w_i ). Weight 0 = no influence;
a heavily weighted check acts as a veto when it scores low.
"""
import copy
import json
import threading
from pathlib import Path

CHECKS = [
    # ---------- stage 1: whole band, every frequency bin, straight from IQ ----------
    {"id": "twinning", "stage": 1, "label": "Sideband twinning",
     "desc": "Upper and lower sidebands rise and fall together (AM's mirror-image sidebands).",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
    {"id": "syllabic", "stage": 1, "label": "Syllable rhythm",
     "desc": "Sideband loudness pulses in the syllable band (2-10 Hz) above the noise level.",
     "default": {"enabled": True, "weight": 1.0, "band_lo": 2.0, "band_hi": 10.0, "full_scale": 8.0},
     "params": [{"key": "band_lo", "label": "band low (Hz)", "min": 0.5, "max": 6, "step": 0.5},
                {"key": "band_hi", "label": "band high (Hz)", "min": 6, "max": 12, "step": 0.5},
                {"key": "full_scale", "label": "x noise for full score", "min": 2, "max": 30, "step": 1}]},
    {"id": "mirror", "stage": 1, "label": "Fine mirror",
     "desc": "Bin-by-bin mirror image of the sideband spectrum around the carrier. Only matches at the exact centre.",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
    {"id": "carrier", "stage": 1, "label": "Carrier present",
     "desc": "A carrier line stands above its own sideband level. Turn off to allow carrierless (DSB-SC) voices.",
     "default": {"enabled": True, "weight": 0.7, "full_db": 15.0},
     "params": [{"key": "full_db", "label": "dB above sidebands for full score", "min": 3, "max": 30, "step": 1}]},
    {"id": "ring", "stage": 1, "label": "FM rejection (ring)",
     "desc": "Rejects signals whose loudness also pulses 4.5-8 kHz out (wideband FM). Off in the AM broadcast band.",
     "default": {"enabled": False, "weight": 1.0, "lo_hz": 4500, "hi_hz": 8000},
     "params": [{"key": "lo_hz", "label": "ring inner (Hz)", "min": 3500, "max": 20000, "step": 500},
                {"key": "hi_hz", "label": "ring outer (Hz)", "min": 5000, "max": 40000, "step": 500}]},
    {"id": "broadband", "stage": 1, "label": "Broadband-pulse rejection",
     "desc": "Rejects candidates whose loudness pulses in step with the wide surrounding band (local electrical noise).",
     "default": {"enabled": True, "weight": 1.0, "lo_hz": 20000, "hi_hz": 60000},
     "params": [{"key": "lo_hz", "label": "inner (Hz)", "min": 8000, "max": 50000, "step": 1000},
                {"key": "hi_hz", "label": "outer (Hz)", "min": 20000, "max": 200000, "step": 5000}]},
    # ---------- stage 2: per track, on the demodulated voice candidate ----------
    {"id": "comb", "stage": 2, "label": "Pitch comb",
     "desc": "Harmonic comb from vocal-fold pitch (autocorrelation peak in the human pitch range).",
     "default": {"enabled": True, "weight": 1.0, "f0_min": 75, "f0_max": 350},
     "params": [{"key": "f0_min", "label": "pitch min (Hz)", "min": 50, "max": 150, "step": 5},
                {"key": "f0_max", "label": "pitch max (Hz)", "min": 200, "max": 500, "step": 10}]},
    {"id": "pitch_motion", "stage": 2, "label": "Pitch movement",
     "desc": "Pitch glides smoothly over a few semitones (speech), not fixed (tones) or stepped (music).",
     "default": {"enabled": True, "weight": 0.7, "range_st": 4.0},
     "params": [{"key": "range_st", "label": "semitone range for full score", "min": 1, "max": 12, "step": 0.5}]},
    {"id": "shape", "stage": 2, "label": "Speech spectral shape",
     "desc": "Energy concentrated below 1 kHz, falling toward 3.4 kHz (long-term speech spectrum).",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "voicing", "stage": 2, "label": "Voiced/unvoiced alternation",
     "desc": "Switches between harmonic (vowels) and non-harmonic (consonants, gaps) several times a second.",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "rhythm", "stage": 2, "label": "Pause rhythm",
     "desc": "Loudness rhythm peaking near 4 Hz with short pauses between words.",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "am_phase", "stage": 2, "label": "AM phase coherence",
     "desc": "Cross-term test: carrier/sideband pairs in phase (AM) rather than anti-phase (FM) or random (noise).",
     "default": {"enabled": True, "weight": 0.7, "full_z": 8.0},
     "params": [{"key": "full_z", "label": "noise sigmas for full score", "min": 2, "max": 30, "step": 1}]},
    {"id": "vad", "stage": 2, "label": "ML voice detector (Silero)",
     "desc": "Neural voice-activity detector run only on the demodulated candidates.",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
]

DEFAULT = {
    "sdr": {
        "source": "hackrf",                 # hackrf | rtlsdr | file
        "center_hz": 2.2e6,
        "sample_rate": 4e6,
        "fmin_hz": 520e3,                   # analysis/display crop (absolute Hz); null = whole capture
        "fmax_hz": 1720e3,
        "hackrf": {"amp": True, "lna": 16, "vga": 20},
        "rtlsdr": {"gain": 8.7, "ppm": 0},
        "file": {"path": "captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8"},
    },
    "analysis": {"bin_hz": 125, "frame_rate": 25, "hop_s": 1.0, "window_s": 5.0,
                 "sideband_lo_hz": 250, "sideband_hi_hz": 3400, "stat_win_s": 0.5, "settle_s": 1.5},
    "detect": {
        "candidate_threshold": 0.30,        # stage-1 score a bin needs to become a candidate
        "threshold": 0.55,                  # final confidence to count as a detection
        "novelty": 0.8,                     # subtract this x each bin's running baseline (0 = off)
        "novelty_tau_s": 15.0,
        "prominence": 0.5,                  # subtract this x the local frequency average (0 = off)
        "prominence_hz": 30e3,
        "max_candidates": 6,                # per frame
        "min_sep_hz": 6e3,
        "max_tracks": 12,
        "max_slide_hz_s": 100e3,
        "track_gap_s": 0.3,
        "link_hops_s": 0.4,                 # a new track starting this soon after one ends = same voice (hop)
        "min_duration_s": 0.4,
    },
    "audio": {"delay_s": 4.0, "bandwidth_hz": 3400},
    "record": {"enabled": True, "dir": "recordings", "save_iq": True, "max_s": 60.0},
    "display": {"width": 1200, "dyn_range_db": 50.0},
    "checks": {c["id"]: dict(c["default"]) for c in CHECKS},
}

PRESETS = {
    "am_hackrf": {"label": "AM broadcast (HackRF)",
                  "sdr": {"source": "hackrf", "center_hz": 2.2e6, "sample_rate": 4e6,
                          "fmin_hz": 520e3, "fmax_hz": 1720e3},
                  "checks": {"ring": {"enabled": False}}},
    "air_hackrf": {"label": "Airband 118-122 MHz (HackRF)",
                   "sdr": {"source": "hackrf", "center_hz": 120.0e6, "sample_rate": 4e6,
                           "fmin_hz": 118.0e6, "fmax_hz": 121.9e6},
                   "checks": {"ring": {"enabled": False}}},
    "fm_hackrf": {"label": "FM band 96-104 MHz (HackRF, ghosts only)",
                  "sdr": {"source": "hackrf", "center_hz": 100.0e6, "sample_rate": 8e6,
                          "fmin_hz": 96.2e6, "fmax_hz": 103.8e6},
                  "checks": {"ring": {"enabled": True}}},
    "air_rtl": {"label": "Airband 119.5 MHz (RTL-SDR)",
                "sdr": {"source": "rtlsdr", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                        "fmin_hz": None, "fmax_hz": None},
                "checks": {"ring": {"enabled": False}}},
    "file_inj": {"label": "Replay: FM capture with injected AM voices",
                 "sdr": {"source": "file", "center_hz": 96.6e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8"}},
                 "checks": {"ring": {"enabled": True}}},
    "file_air": {"label": "Replay: airband capture",
                 "sdr": {"source": "file", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/air_119.5MHz_2.4Msps_g29.7.cu8"}},
                 "checks": {"ring": {"enabled": False}}},
    "file_am": {"label": "Replay: AM band capture (HackRF)",
                "sdr": {"source": "file", "center_hz": 2.2e6, "sample_rate": 4e6,
                        "fmin_hz": 520e3, "fmax_hz": 1720e3,
                        "file": {"path": "captures/am_2.2MHz_4Msps_hackrf_a1l16g20.cs8"}},
                "checks": {"ring": {"enabled": False}}},
}


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
        self.version = 0

    def get(self):
        with self._lock:
            return copy.deepcopy(self._cfg)

    def update(self, patch):
        with self._lock:
            before = copy.deepcopy(self._cfg["sdr"])
            deep_merge(self._cfg, patch)
            self.version += 1
            sdr_changed = before != self._cfg["sdr"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._cfg, indent=1))
            return copy.deepcopy(self._cfg), sdr_changed

    def reset(self):
        return self.update(copy.deepcopy(DEFAULT))

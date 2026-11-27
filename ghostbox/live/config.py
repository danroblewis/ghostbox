"""Live-engine configuration and the registry of detection checks.

Every check has a weight and its own parameters. Weight 0 = off: the check is
not computed at all. `enabled` is kept in the config but is always derived from
the weight (weight > 0), so the two can never disagree. The UI is generated from
CHECKS, so adding a check here (plus its implementation in stage1.py /
stage2.py) is all it takes to expose it.

Confidence is a weighted geometric mean of the active checks' scores (each in
[0, 1]):  exp( sum w_i log max(s_i, eps) / sum w_i ). A heavily weighted check
acts as a veto when it scores low.
"""
import copy
import json
import threading
from pathlib import Path

# family: "dsb" (ordinary AM) and "ssb" are alternative modulation hypotheses: within a family the
# checks combine as usual, and the better-fitting family counts. Checks without a family always count.
# stage 1 = whole-band IQ checks (every frequency at once; each has a waterfall view, computed only when
# enabled). stage 2 = per-detection checks (run on each candidate track).
CHECKS = [
    # ---------- stage 1: modulation (alternatives) ----------
    {"id": "twinning", "group": "modulation", "stage": 1, "family": "dsb", "label": "Sideband twinning",
     "desc": "AM: upper and lower sidebands rise and fall together (mirror-image sidebands).",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
    {"id": "slide", "group": "modulation", "stage": 1, "family": "dsb", "label": "Sliding twinning",
     "desc": "AM twinning measured along slanted lines, so a voice drifting in frequency builds evidence instead of smearing. Costs several times the CPU of plain twinning.",
     "default": {"enabled": False, "weight": 0.0, "max_khz_s": 100, "steps": 3},
     "params": [{"key": "max_khz_s", "label": "fastest slide (kHz/s)", "min": 10, "max": 300, "step": 10},
                {"key": "steps", "label": "slide rates each way", "min": 1, "max": 6, "step": 1}]},
    {"id": "mirror", "group": "modulation", "stage": 1, "family": "dsb", "label": "Fine mirror",
     "desc": "AM: bin-by-bin mirror image of the sideband spectrum around the carrier. Only matches at the exact centre.",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
    {"id": "carrier", "group": "modulation", "stage": 1, "family": "dsb", "label": "Carrier present",
     "desc": "AM: a carrier line stands above its own sideband level. Set to 0 to allow carrierless (DSB-SC) voices.",
     "default": {"enabled": True, "weight": 0.7, "full_db": 15.0},
     "params": [{"key": "full_db", "label": "dB above sidebands for full score", "min": 3, "max": 30, "step": 1}]},
    {"id": "ssb", "group": "modulation", "stage": 1, "family": "ssb", "label": "Single-sideband voice",
     "desc": "SSB: a voice-band rhythm on one side of a (suppressed) carrier only. The alternative to the AM checks; whichever fits better counts.",
     "default": {"enabled": True, "weight": 1.0, "full_db": 8.0},
     "params": [{"key": "full_db", "label": "one-sidedness for full score (dB)", "min": 3, "max": 20, "step": 1}]},
    # ---------- stage 1: speech evidence ----------
    {"id": "syllabic", "group": "speech", "stage": 1, "label": "Syllable rhythm",
     "desc": "Sideband loudness pulses in the syllable band (2-10 Hz) above the noise level.",
     "default": {"enabled": True, "weight": 1.0, "band_lo": 2.0, "band_hi": 10.0, "full_scale": 8.0},
     "params": [{"key": "band_lo", "label": "band low (Hz)", "min": 0.5, "max": 6, "step": 0.5},
                {"key": "band_hi", "label": "band high (Hz)", "min": 6, "max": 12, "step": 0.5},
                {"key": "full_scale", "label": "x noise for full score", "min": 2, "max": 30, "step": 1}]},
    {"id": "rhythm", "group": "speech", "stage": 1, "label": "4 Hz syllable peak",
     "desc": "Loudness changes at the syllable rate (3-6 Hz) but hardly faster (8-12 Hz): true of speech, not of noise, which changes at every rate.",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "comb", "group": "speech", "stage": 1, "label": "Pitch comb",
     "desc": "Evenly spaced harmonic lines around a carrier, spaced by a human voice pitch: measured from the raw IQ at fine resolution, for every frequency at once.",
     "default": {"enabled": True, "weight": 1.0, "f0_min": 75, "f0_max": 350},
     "params": [{"key": "f0_min", "label": "pitch min (Hz)", "min": 50, "max": 150, "step": 5},
                {"key": "f0_max", "label": "pitch max (Hz)", "min": 200, "max": 500, "step": 10}]},
    {"id": "pitch_motion", "group": "speech", "stage": 1, "label": "Pitch movement",
     "desc": "The comb's pitch glides smoothly over a few semitones (speech), not fixed (tones) or stepped (music).",
     "default": {"enabled": True, "weight": 0.7, "range_st": 4.0},
     "params": [{"key": "range_st", "label": "semitone range for full score", "min": 1, "max": 12, "step": 0.5}]},
    {"id": "voicing", "group": "speech", "stage": 1, "label": "Voiced/unvoiced alternation",
     "desc": "The pitch comb comes and goes several times a second (vowels vs consonants and gaps).",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "shape", "group": "speech", "stage": 1, "label": "Speech spectral shape",
     "desc": "Sideband energy is strongest near the carrier and falls toward 3.4 kHz (formant-shaped), unlike flat noise.",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "formants", "group": "speech", "stage": 2, "label": "Two moving tones (formants)",
     "desc": "Two tones at once that glide independently, like a vowel's first two formants (250-1000 Hz and 850-3000 Hz). Two varying tones over a little noise is what the ear hears as a voice, even without a pitch comb. Scored on each detection's demodulated audio.",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
    {"id": "resonances", "group": "speech", "stage": 1, "label": "Two moving resonances (whole band)",
     "desc": "At every frequency, the sidebands hold two peaks at once, one at 250-1000 Hz and one at 850-3000 Hz from it, and both move independently over the last second: two gliding tones, like a vowel's first two formants, seen straight from the IQ. Mainly a layer for the composed voice; off for detection by default.",
     "default": {"enabled": False, "weight": 0.0}, "params": []},
    {"id": "envstat", "group": "speech", "stage": 1, "label": "Loudness range",
     "desc": "Syllables and pauses make the sideband loudness swing by tens of dB; noise swings a few. Strong for SSB and loud voices, but weak AM voices (carrier present) don't swing enough, so it's off by default.",
     "default": {"enabled": False, "weight": 0.0, "full_db": 12.0},
     "params": [{"key": "full_db", "label": "dB above 8 dB for full score", "min": 4, "max": 30, "step": 1}]},
    # ---------- stage 1: rejection ----------
    {"id": "ring", "group": "rejection", "stage": 1, "label": "FM rejection (ring)",
     "desc": "Rejects signals whose loudness also pulses 4.5-8 kHz out (wideband FM). Off in the AM broadcast band.",
     "default": {"enabled": False, "weight": 0.0, "lo_hz": 4500, "hi_hz": 8000},
     "params": [{"key": "lo_hz", "label": "ring inner (Hz)", "min": 3500, "max": 20000, "step": 500},
                {"key": "hi_hz", "label": "ring outer (Hz)", "min": 5000, "max": 40000, "step": 500}]},
    {"id": "broadband", "group": "rejection", "stage": 1, "label": "Broadband-pulse rejection",
     "desc": "Rejects candidates whose loudness pulses in step with the wide surrounding band (local electrical noise).",
     "default": {"enabled": True, "weight": 1.0, "lo_hz": 20000, "hi_hz": 60000},
     "params": [{"key": "lo_hz", "label": "inner (Hz)", "min": 8000, "max": 50000, "step": 1000},
                {"key": "hi_hz", "label": "outer (Hz)", "min": 20000, "max": 200000, "step": 5000}]},
    {"id": "tone", "group": "rejection", "stage": 2, "label": "Steady-tone rejection",
     "desc": "Rejects steady tones and keyed carriers: audio that is mostly a few fixed tones (beat notes, test tones, FSK data), or a carrier switched on and off several times a second (Morse, data). Speech harmonics move with the pitch, and an AM voice's carrier stays on between words.",
     "default": {"enabled": True, "weight": 2.0, "voice_frac": 0.22, "tone_frac": 0.45},
     "params": [{"key": "voice_frac", "label": "tonality still fine (speech)", "min": 0.05, "max": 0.6, "step": 0.01},
                {"key": "tone_frac", "label": "tonality fully rejected", "min": 0.2, "max": 0.95, "step": 0.01}]},
    # ---------- stage 1: long-term ----------
    {"id": "profile", "group": "longterm", "stage": 1, "label": "Unusual for this band",
     "desc": "Learns what each frequency normally looks like (over hours, saved per band, no IQ kept) and scores how unusual the current evidence is. Neutral until it has learned for a while.",
     "default": {"enabled": True, "weight": 0.5, "tau_h": 2.0, "warmup_min": 10, "full_z": 4.0},
     "params": [{"key": "tau_h", "label": "memory (hours)", "min": 0.25, "max": 24, "step": 0.25},
                {"key": "warmup_min", "label": "learn before scoring (min)", "min": 1, "max": 120, "step": 1},
                {"key": "full_z", "label": "standard deviations for full score", "min": 1, "max": 10, "step": 0.5}]},
    # ---------- stage 2: per detection ----------
    {"id": "comod", "group": "detection", "stage": 2, "label": "Cross-channel co-modulation",
     "desc": "Another frequency pulses with the same syllable rhythm (at once or shortly after): one voice spread across, or hopping between, channels. Co-modulated detections are joined into one chain.",
     "default": {"enabled": True, "weight": 0.5, "min_khz": 20, "max_lag_s": 0.5},
     "params": [{"key": "min_khz", "label": "minimum separation (kHz)", "min": 5, "max": 500, "step": 5},
                {"key": "max_lag_s", "label": "largest delay (s)", "min": 0, "max": 2, "step": 0.1}]},
    {"id": "hop", "group": "detection", "stage": 2, "label": "Hop continuity",
     "desc": "When a voice hops, the new channel continues the old one's pitch. Scores only detections that start a hop.",
     "default": {"enabled": True, "weight": 0.7}, "params": []},
    {"id": "am_phase", "group": "detection", "stage": 2, "family": "dsb", "label": "AM phase coherence",
     "desc": "AM: carrier/sideband pairs in phase (AM) rather than anti-phase (FM) or random (noise).",
     "default": {"enabled": True, "weight": 0.7, "full_z": 8.0},
     "params": [{"key": "full_z", "label": "noise sigmas for full score", "min": 2, "max": 30, "step": 1}]},
    {"id": "vad", "group": "detection", "stage": 2, "label": "ML voice detector (Silero)",
     "desc": "Neural voice-activity detector run only on the demodulated candidates.",
     "default": {"enabled": True, "weight": 1.0}, "params": []},
]
GROUPS = [{"id": "modulation", "label": "Modulation", "note": "AM and SSB are alternatives: whichever fits a signal better counts."},
          {"id": "speech", "label": "Speech evidence"},
          {"id": "rejection", "label": "Rejection"},
          {"id": "longterm", "label": "Long-term"},
          {"id": "detection", "label": "Per detection", "note": "Run on each candidate track rather than the whole band."}]
FAMILIES = {}
for _c in CHECKS:
    if _c.get("family"):
        FAMILIES.setdefault(_c["family"], []).append(_c["id"])

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
    # Playback mode. Off: every track that sounds like a voice on its own plays (individual voices).
    # On (composed voice): one voice assumed split across several channels. No tracks: layers (whole-band
    # filter maps) give every frequency a confidence, 1 - prod(1 - weight x layer); its peaks above
    # candidate_floor are candidate channels. Each channel splits into low / mid / high keys; each hop the
    # keys (from at least min_channels channels) that together best make a voice are played: each
    # channel demodulated, each key band-passed and levelled, never shifted. The set's confidence
    # (structure blended with the ML voice detector on the mix, vad_weight) must reach min_conf (the Min
    # confidence slider) to be heard.
    "compose": {"enabled": False,
                "layers": [{"id": "comb", "weight": 1.0}, {"id": "pitch_motion", "weight": 1.0},
                           {"id": "resonances", "weight": 1.0}, {"id": "rhythm", "weight": 0.8}],
                "candidate_floor": 0.15, "min_conf": 0.3, "vad_weight": 0.5,
                "candidates": 24, "min_sep_hz": 7000,
                "max_components": 6, "min_channels": 2, "window_s": 3.0, "hold_s": 2.0,
                "w_voiced": 1.5, "probe": 8,     # the top `probe` candidates demodulated each hop to hear their pitch
                "w_pitch": 1.0, "w_formants": 1.0, "w_syllables": 1.0, "w_together": 1.0, "w_coverage": 0.5,
                "cost": 0.01,                    # score each extra key must earn
                # voice shaping of the mix, from its own pitch and envelope (voicefx): period reinforced,
                # envelope peaks sharpened, hiss between voiced stretches turned down (0 = off each)
                "shape_pitch": 0.6, "shape_formant": 0.6, "shape_duck_db": 12.0,
                "segment_s": 8.0,                # the mix is recorded and transcribed in pieces at most this long,
                "overlap_s": 0.5},               # each starting with this much of the piece before
    # Frequency ranges neither mode uses (known stations, carriers): [{"lo": Hz, "hi": Hz, "label": ""}]
    "blocked": [],
    "audio": {"delay_s": 4.0, "bandwidth_hz": 3400,      # (unused: the channel follows voice_hi_hz)
              "demod": "auto",                          # auto (sync AM with a carrier, suppressed-carrier loop without;
                                                        # product detector for SSB) | sync | suppressed | sync_sb | envelope
              "voice_band": True, "voice_lo_hz": 300, "voice_hi_hz": 3000,   # hard band-pass for playback
              "preroll_s": 0.5, "postroll_s": 0.5, "fade_s": 0.15,      # ease detections in and out
              "denoise": True, "denoise_strength": 1.0, "denoise_floor_db": -18.0},
    "record": {"enabled": True, "dir": "recordings", "save_iq": True, "max_s": 60.0,
               "max_mb": 2000,                  # ring buffer: oldest recordings deleted beyond this
               "min_save_conf": 0.70,           # only detections at least this confident are written to disk
               "memory_mb": 300},               # audio of the rest is kept in memory (oldest dropped first)
    "iq_ring": {"enabled": False, "dir": "iq_ring", "max_gb": 4.0, "segment_s": 10,
                "record_replays": False},       # continuous raw IQ, oldest segments deleted beyond max_gb
    "storage": {"min_free_gb": 10.0},           # disk guard: below this, trim the IQ ring, then pause writing
    "autotune": {"auto": False,                 # recalibrate automatically: after any receiver change (once enough
                                                # IQ is recorded since) and whenever nothing is caught for `minutes`
                 "minutes": 2, "per_min": 6,    # analyse this much; aim for at most this many detections/min,
                 "floor": 0.3},                 # counting only candidates the ML judge finds this speech-like
    "display": {"width": 1200, "dyn_range_db": 60.0, "history_min": 20,
                # Power change view: each column's power minus its own running average (memory change_tau_s),
                # so steady carriers vanish; rises under change_floor_db (noise) are hidden, full colour at
                # change_floor_db + change_range_db
                "change_tau_s": 30.0, "change_floor_db": 6.0, "change_range_db": 24.0},
    "whisper": {
        "enabled": True,
        "model": "base",                    # an id from asr.MODELS (Whisper, Moonshine, wav2vec2, Vosk)
        "language": "auto",                 # auto or an ISO code (en, es, ...)
        "min_confidence": 0.55,             # only transcribe detections that reached this confidence
        "chains": True,                     # also transcribe a whole hop chain stitched together
        "workers": 2,                       # worker threads sharing one resident model
        "batch_size": 16,                   # clips per batch (each in its own 30 s window)
        "cpu_threads": 8,
        "max_queue": 64,                    # beyond this the least confident clip is dropped
        "max_age_s": 120,                   # clips waiting longer than this are skipped
        "filter_phrases": [                 # removed wherever they appear (on top of the built-in list)
            "Thank you very much for your continued support.", "dragonball", "Piaw", "Click on the top",
            "discontinue eligibility", "I'm gonna be the one who's gonna be the one"],
    },
    "checks": {c["id"]: dict(c["default"]) for c in CHECKS},
}

# Waterfall views: power, the combined score, novelty, and one per stage-1 check (plus the sparse
# co-modulation map). Only enabled checks are computed, so the UI lists only those.
VIEWS = ([{"id": "power", "label": "Power (dB)"},
          {"id": "power_change", "label": "Power change (above usual)"},
          {"id": "score", "label": "AM-voice score (combined)"},
          {"id": "novelty", "label": "Novelty evidence (what triggers candidates)"}]
         + [{"id": c["id"], "label": c["label"], "check": True} for c in CHECKS if c["stage"] == 1]
         + [{"id": "comod", "label": "Cross-channel co-modulation", "check": True}])

PRESETS = {
    "am_hackrf": {"label": "AM broadcast (HackRF)",
                  "sdr": {"source": "hackrf", "center_hz": 2.2e6, "sample_rate": 4e6,
                          "fmin_hz": 520e3, "fmax_hz": 1720e3, "gain_db": 36, "amp": True},
                  "checks": {"ring": {"weight": 0.0}}},
    "air_hackrf": {"label": "Airband 118-122 MHz (HackRF)",
                   "sdr": {"source": "hackrf", "center_hz": 120.0e6, "sample_rate": 4e6,
                           "fmin_hz": 118.0e6, "fmax_hz": 121.9e6, "gain_db": 48, "amp": True},
                   "checks": {"ring": {"weight": 0.0}}},
    "fm_hackrf": {"label": "FM band 96-104 MHz (HackRF, ghosts only)",
                  "sdr": {"source": "hackrf", "center_hz": 100.0e6, "sample_rate": 8e6,
                          "fmin_hz": 96.2e6, "fmax_hz": 103.8e6, "gain_db": 24, "amp": False},
                  "checks": {"ring": {"weight": 1.0}}},
    "air_rtl": {"label": "Airband 119.5 MHz (RTL-SDR)",
                "sdr": {"source": "rtlsdr", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                        "fmin_hz": None, "fmax_hz": None, "gain_db": 29.7},
                "checks": {"ring": {"weight": 0.0}}},
    "file_inj": {"label": "Replay: FM capture with injected AM voices",
                 "sdr": {"source": "file", "center_hz": 96.6e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8"}},
                 "checks": {"ring": {"weight": 1.0}}},
    "file_air": {"label": "Replay: airband capture",
                 "sdr": {"source": "file", "center_hz": 119.5e6, "sample_rate": 2.4e6,
                         "fmin_hz": None, "fmax_hz": None,
                         "file": {"path": "captures/air_119.5MHz_2.4Msps_g29.7.cu8"}},
                 "checks": {"ring": {"weight": 0.0}}},
    "file_am": {"label": "Replay: AM band capture (HackRF)",
                "sdr": {"source": "file", "center_hz": 2.2e6, "sample_rate": 4e6,
                        "fmin_hz": 520e3, "fmax_hz": 1720e3,
                        "file": {"path": "captures/am_2.2MHz_4Msps_hackrf_a1l16g20.cs8"}},
                "checks": {"ring": {"weight": 0.0}}},
}


# One-click detector sensitivity. "Ghost hunter" trades precision for many more candidates;
# stage-2 voice checks are softened so more human-ish fragments get through.
SENSITIVITY = {
    "strict": {"label": "Strict", "detect": {"candidate_threshold": 0.35, "threshold": 0.65, "min_duration_s": 0.5,
               "max_candidates": 6, "max_tracks": 12, "novelty": 0.8, "prominence": 0.5},
               "checks": {"vad": {"weight": 1.5}, "comb": {"weight": 1.0}}},
    "balanced": {"label": "Balanced", "detect": {"candidate_threshold": 0.30, "threshold": 0.55, "min_duration_s": 0.4,
                 "max_candidates": 6, "max_tracks": 12, "novelty": 0.8, "prominence": 0.5},
                 "checks": {"vad": {"weight": 1.0}, "comb": {"weight": 1.0}}},
    "ghost": {"label": "Ghost hunter", "detect": {"candidate_threshold": 0.22, "threshold": 0.42, "min_duration_s": 0.25,
              "max_candidates": 12, "max_tracks": 24, "novelty": 0.6, "prominence": 0.3},
              "checks": {"vad": {"weight": 0.5}, "comb": {"weight": 0.7}}},
    "everything": {"label": "Everything", "detect": {"candidate_threshold": 0.15, "threshold": 0.30, "min_duration_s": 0.2,
                   "max_candidates": 20, "max_tracks": 40, "novelty": 0.4, "prominence": 0.2},
                   "checks": {"vad": {"weight": 0.25}, "comb": {"weight": 0.5}}},
}


def hackrf_gains(gain_db):
    """Split one gain figure into HackRF LNA (0-40, step 8) + VGA (0-62, step 2)."""
    g = max(0.0, float(gain_db))
    lna = int(min(40, 8 * int(g / 2 / 8)))
    vga = int(min(62, max(0, 2 * round((g - lna) / 2))))
    return lna, vga


def normalize_checks(cfg, patch=None):
    """Weight is the only switch: `enabled` = weight > 0.
    Loading (patch=None): an older config's enabled: false becomes weight 0.
    A patch that sets enabled without a weight (older clients, presets) switches by weight:
    false -> 0, true on a check at 0 -> its default weight (or 1)."""
    pc = (patch or {}).get("checks") or {}
    for c in CHECKS:
        ch = cfg["checks"].setdefault(c["id"], dict(c["default"]))
        if patch is None:
            if ch.get("enabled") is False:
                ch["weight"] = 0.0
        else:
            p = pc.get(c["id"])
            if isinstance(p, dict) and "enabled" in p and "weight" not in p:
                if not p["enabled"]:
                    ch["weight"] = 0.0
                elif float(ch.get("weight", 0)) <= 0:
                    ch["weight"] = float(c["default"]["weight"]) or 1.0
        ch["weight"] = max(0.0, float(ch.get("weight", 0)))
        ch["enabled"] = ch["weight"] > 0
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
        normalize_checks(self._cfg)
        cc = self._cfg.get("compose", {})
        if "candidate_threshold" in cc or "max_tracks" in cc:     # the earlier track-based composed voice
            self._cfg["compose"] = dict(copy.deepcopy(DEFAULT["compose"]), enabled=bool(cc.get("enabled")))
        elif "min_channels" not in cc:                            # before keys: new choosing settings, keep the rest
            keep = {k: cc[k] for k in ("enabled", "layers", "min_conf", "window_s", "hold_s", "candidates",
                                       "segment_s", "overlap_s", "w_pitch", "w_formants", "w_syllables") if k in cc}
            self._cfg["compose"] = dict(copy.deepcopy(DEFAULT["compose"]), **keep)
        self.version = 0

    def get(self):
        with self._lock:
            return copy.deepcopy(self._cfg)

    def update(self, patch):
        with self._lock:
            before = copy.deepcopy(self._cfg["sdr"])
            deep_merge(self._cfg, patch)
            normalize_checks(self._cfg, patch)
            self.version += 1
            sdr_changed = before != self._cfg["sdr"]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._cfg, indent=1))
            return copy.deepcopy(self._cfg), sdr_changed

    def reset(self):
        return self.update(copy.deepcopy(DEFAULT))

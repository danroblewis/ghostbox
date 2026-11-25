"""Tune the demodulation and filtering for one detection: the one you believe in.

Works on that detection's IQ only (read once from the IQ ring). Each setting is judged by the ML
voice detector on the audio you would actually hear (the mean of the more speech-like half of its
32 ms frames, as in the detector's own voice check). Stages, each keeping the previous choices:

1. detector: Auto, Envelope, Sync AM, Suppressed carrier, Sync sideband, and USB / LSB (the
   AM/SSB classification can be wrong);
2. for USB / LSB the presumed carrier is first fine-tuned from the voice's harmonics (they sit at
   exact multiples of the pitch only when the carrier is right), which removes the pitch shift /
   warble of a mistuned SSB voice; the voice detector alone barely notices a mistuning;
3. voice band edges (or no band filter);
4. noise reduction (off / gentle / normal / strong).

Simpler options come first and win ties, so nothing is added that doesn't help.
"""
import numpy as np

from .audio import render_clip
from .engine import baseband, detect_clip
from .stage2 import vad_probs

DETECTORS = [("auto", 0, "Auto"), ("envelope", 0, "Envelope"), ("sync", 0, "Sync AM"),
             ("suppressed", 0, "Suppressed carrier"), ("sync_sb", 0, "Sync sideband"),
             ("auto", 1, "USB"), ("auto", -1, "LSB")]
BANDS = [(None, None)] + [(lo, hi) for lo in (200, 300, 400, 500) for hi in (2400, 2800, 3200, 3800)]
DENOISE = [(False, 1.0, -18.0, "off"), (True, 0.5, -24.0, "gentle"), (True, 1.0, -18.0, "normal"),
           (True, 1.5, -12.0, "strong")]
TIE = 0.002


def judge(audio):
    """Voice score of what you'd hear: the voice detector's log-odds over its more speech-like half
    of frames (log-odds keep telling 0.99 from 0.999 apart, where a plain probability saturates),
    mapped back to 0..1 for display."""
    p = np.sort(vad_probs(np.asarray(audio, np.float32)))
    if not len(p):
        return 0.0
    q = np.clip(p[len(p) // 2:], 1e-4, 1 - 1e-4)
    return float(1 / (1 + np.exp(-np.mean(np.log(q / (1 - q))))))


def harmonic_offset(audio, fs=16000, max_hz=200.0):
    """How far a single-sideband voice is mistuned, from its harmonics: a voiced frame has energy
    at k*F0 when the presumed carrier is right and at k*F0 + d when it is d Hz off. Each loud frame
    is fitted with combs over pitch (80-350 Hz) and shift d; the shift that aligns best across all
    frames (whose pitches differ, which settles which harmonic is which) is the mistuning.
    Returns (d in Hz, confidence 0..1)."""
    x = np.asarray(audio, np.float32)
    N, hop = 2048, 512
    if len(x) < N * 2:
        return 0.0, 0.0
    w = np.hanning(N)
    fr = np.stack([x[i:i + N] * w for i in range(0, len(x) - N, hop)])
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 0.5                   # compressed magnitude
    df = fs / N
    e = (S ** 2).sum(axis=1)
    loud = S[e >= np.percentile(e, 50)]
    loud = loud / (loud.mean(axis=1, keepdims=True) + 1e-12)
    f0s = np.arange(80.0, 351.0, 2.0)
    ds = np.arange(-max_hz, max_hz + 0.1, 2.0)
    total = np.zeros(len(ds))
    for j, d in enumerate(ds):
        best = np.zeros(len(loud))
        for f0 in f0s:
            k = np.arange(1, int(3200 // f0) + 1)
            f = k * f0 + d
            f = f[(f > 150) & (f < 3400)]
            if len(f) < 4:
                continue
            idx = np.round(f / df).astype(int)
            sc = loud[:, idx].mean(axis=1)                       # mean: sub-harmonic combs don't win
            best = np.maximum(best, sc)
        total[j] = best.mean()
    j = int(np.argmax(total))
    z = (total[j] - np.median(total)) / (total.std() + 1e-12)
    return float(ds[j]), float(np.clip(z / 6, 0, 1))


def tune(loaded, pts, au0, mode0, progress=None):
    """Returns the best settings for this clip, its voice score, the score with the current
    settings, and every trial."""
    trials = []
    wide = dict(au0, voice_hi_hz=3800)                       # demodulate wide; the band filter cuts later
    bb_cache = {}

    def bb(offset):
        if offset not in bb_cache:
            bb_cache[offset] = baseband(loaded, pts, wide, offset)
        return bb_cache[offset]

    def hear(raw, s):
        a = dict(au0, voice_band=s["lo"] is not None, voice_lo_hz=s["lo"] or 300, voice_hi_hz=s["hi"] or 3000,
                 denoise=s["denoise"], denoise_strength=s["strength"], denoise_floor_db=s["floor"])
        return np.clip(render_clip(raw, a), -1, 1)

    def run(stage, s, raw, label):
        sc = judge(hear(raw, s))
        trials.append({"stage": stage, "label": label, "score": round(sc, 4), **s})
        return sc

    cur = {"demod": au0.get("demod", "auto"), "mode": mode0, "offset": 0.0,
           "lo": au0.get("voice_lo_hz", 300) if au0.get("voice_band", True) else None,
           "hi": au0.get("voice_hi_hz", 3000) if au0.get("voice_band", True) else None,
           "denoise": au0.get("denoise", True), "strength": au0.get("denoise_strength", 1.0),
           "floor": au0.get("denoise_floor_db", -18.0)}
    base_raw = detect_clip(bb(0.0), dict(wide, demod=cur["demod"]), mode0)
    baseline = run("current", cur, base_raw, "Current settings")
    total = len(DETECTORS) + len(BANDS) + len(DENOISE)
    done = [0]

    def tick():
        done[0] += 1
        if progress:
            progress(min(0.99, done[0] / total))

    # 1. detector (with the current band and noise reduction); sideband trials are fine-tuned first
    best, best_sc, raws = None, -1.0, {}
    for demod, mode, label in DETECTORS:
        off, conf = 0.0, None
        if mode:
            off, conf = harmonic_offset(detect_clip(bb(0.0), wide, mode))
            if conf < 0.15:                                  # no clear harmonics: don't move the carrier
                off = 0.0
        s = dict(cur, demod=demod, mode=mode, offset=off)
        raw = detect_clip(bb(off), dict(wide, demod=demod), mode)
        sc = run("detector", s, raw, label + (f" ({off:+.0f} Hz)" if off else ""))
        tick()
        # a sideband reading with clearly aligned harmonics wins ties: the voice detector can't hear
        # a pitch shift, the harmonics can
        aligned = bool(mode) and conf is not None and conf >= 0.3
        if aligned and best is not None and best["mode"] == 0 and sc >= best_sc - TIE and sc <= best_sc + TIE:
            # only if the AM reading is itself pitch-shifted (an SSB voice through an AM detector);
            # an AM voice through an AM detector is already at the right pitch
            d0, c0 = harmonic_offset(raws[best["offset"]])
            aligned = not (c0 >= 0.3 and abs(d0) < 10)
        if sc > best_sc + TIE or (aligned and sc >= best_sc - TIE):
            best, best_sc, raws = s, sc, {off: raw}
    raw = raws.get(best["offset"]) if raws else None
    if raw is None:
        raw = detect_clip(bb(best["offset"]), dict(wide, demod=best["demod"]), best["mode"])
    # 3. voice band, 4. noise reduction (re-use the demodulated audio)
    for lo, hi in BANDS:
        s = dict(best, lo=lo, hi=hi)
        sc = run("band", s, raw, "no band filter" if lo is None else f"{lo}-{hi} Hz")
        tick()
        if sc > best_sc + TIE:
            best, best_sc = s, sc
    for on, strength, floor, label in DENOISE:
        s = dict(best, denoise=on, strength=strength, floor=floor)
        sc = run("denoise", s, raw, f"noise reduction {label}")
        tick()
        if sc > best_sc + TIE:
            best, best_sc = s, sc
    label = next((l for d, m, l in DETECTORS if d == best["demod"] and m == best["mode"]),
                 {1: "USB", -1: "LSB"}.get(best["mode"], best["demod"]))
    return {"best": dict(best, label=label), "score": round(best_sc, 4), "baseline": round(baseline, 4),
            "trials": trials}


def as_audio_settings(best):
    """The global playback settings a tuning result maps to (the carrier offset and a forced
    USB/LSB only make sense for that one signal, so they are not part of it)."""
    out = {"voice_band": best["lo"] is not None, "denoise": bool(best["denoise"]),
           "denoise_strength": best["strength"], "denoise_floor_db": best["floor"]}
    if best["lo"] is not None:
        out.update(voice_lo_hz=best["lo"], voice_hi_hz=best["hi"])
    if best["mode"] == 0:
        out["demod"] = best["demod"]
    return out

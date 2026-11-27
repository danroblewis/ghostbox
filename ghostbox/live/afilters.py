"""Audio filters for listening back to a clip (detail dialog): applied to finished 16 kHz audio, in a
fixed order, each optional. They only change what you hear; detection never sees them.

  declick   impulse noise (pulses, clicks) cut out and bridged
  notch     steady tones (carrier whistles, hum, data tones) found and removed
  hp / lp   high-pass / low-pass (rumble / hiss)
  nr        noise reduction (spectral gating, the same as live playback)
  voice     voice shaper: the clip's own pitch period and formant peaks reinforced, hiss between ducked
  presence  speech-presence boost around 2.5 kHz (dB)
  gate      mutes what stays near the noise floor between sounds (dB above it)
  comp      evens out loud and quiet parts (0..1)
  echo      a short room echo (0..1 mix)
  speed     playback speed (pitch moves with it, like a tape)
  reverse   backwards
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft

from .audio import StreamingDenoiser
from .voicefx import VoiceShaper

FS = 16000

# id -> (label, default value, min, max, step, unit). A filter is on when its id is in the request.
FILTERS = [
    ("declick", "Click / pulse removal", 6.0, 3.0, 12.0, 0.5, "x"),
    ("notch", "Remove steady tones", 6.0, 3.0, 20.0, 1.0, "dB"),
    ("hp", "High-pass", 250.0, 50.0, 1000.0, 10.0, "Hz"),
    ("lp", "Low-pass", 3200.0, 1000.0, 7000.0, 100.0, "Hz"),
    ("nr", "Noise reduction", 1.0, 0.2, 3.0, 0.1, "x"),
    ("voice", "Voice shaper (pitch + formants from the clip itself)", 0.6, 0.1, 1.0, 0.05, ""),
    ("presence", "Speech presence boost", 6.0, 1.0, 15.0, 1.0, "dB"),
    ("gate", "Noise gate", 6.0, 0.0, 20.0, 1.0, "dB"),
    ("comp", "Leveller (compressor)", 0.5, 0.1, 0.9, 0.05, ""),
    ("echo", "Room echo", 0.3, 0.05, 0.8, 0.05, ""),
    ("speed", "Speed", 0.8, 0.5, 1.5, 0.05, "x"),
    ("reverse", "Reverse", 1.0, 1.0, 1.0, 1.0, ""),
]


def _declick(y, k):
    """Samples k x above the local typical level (20 ms), widened by 2 ms, replaced by a straight line."""
    a = np.abs(y)
    ref = median_filter(a, size=int(0.02 * FS)) + 1e-6
    bad = maximum_filter1d((a > k * ref).astype(np.uint8), int(0.004 * FS)) > 0
    if not bad.any() or bad.mean() > 0.3:
        return y
    idx = np.arange(len(y))
    out = y.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return out


def _notch(y, db):
    """Bins whose average level stands db above their neighbourhood and stays steady (a tone, not a
    voice harmonic, which moves) are turned down to the neighbourhood level."""
    f, t, Z = stft(y, FS, nperseg=2048, noverlap=1536)
    if Z.shape[1] < 4:
        return y
    L = 20 * np.log10(np.abs(Z) + 1e-9)
    mean = L.mean(axis=1)
    around = median_filter(mean, size=31)
    steady = np.median(L, axis=1) > around + db               # there most of the time, not just loud once
    tone = (mean - around > db) & steady
    tone = maximum_filter1d(tone.astype(np.uint8), 5) > 0           # the window spreads a tone over ~5 bins
    if not tone.any():
        return y
    g = np.ones(len(f))
    g[tone] = 10 ** ((around[tone] - mean[tone]) / 20)
    _, out = istft(Z * g[:, None], FS, nperseg=2048, noverlap=1536)
    return out[: len(y)].astype(np.float32) if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out))).astype(np.float32)


def _peak(y, f0, gain_db, q=0.9):
    """RBJ peaking EQ."""
    A, w = 10 ** (gain_db / 40), 2 * np.pi * f0 / FS
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    from scipy.signal import filtfilt
    return filtfilt(b / a[0], a / a[0], y).astype(np.float32)


def _env_db(y, win_s):
    return 10 * np.log10(uniform_filter1d(y * y, max(1, int(win_s * FS))) + 1e-12)


def _gate(y, db):
    e = _env_db(y, 0.02)
    floor = np.percentile(e, 20)
    g = np.clip((e - floor - db) / 6, 0, 1)                    # 6 dB soft knee above the threshold
    g = uniform_filter1d(g, int(0.015 * FS))                    # no zipper noise
    return (y * (0.05 + 0.95 * g)).astype(np.float32)


def _comp(y, amount):
    e = _env_db(y, 0.05)
    ref = np.percentile(e, 95)
    gain_db = np.clip((ref - e) * amount, 0, 30)                # quiet parts up, by up to 30 dB
    gain_db = uniform_filter1d(gain_db, int(0.03 * FS))
    return (y * 10 ** (gain_db / 20)).astype(np.float32)


def _echo(y, mix):
    out = y.copy()
    for d, g in ((0.043, 0.6), (0.071, 0.45), (0.113, 0.3), (0.167, 0.2)):
        n = int(d * FS)
        out[n:] += mix * g * y[:-n]
    return out


def apply(y, params):
    """params: {filter id: value}; only the filters present are applied."""
    y = np.asarray(y, np.float32).copy()
    if not len(y):
        return y
    p = {k: float(v) for k, v in params.items()}
    if "declick" in p:
        y = _declick(y, p["declick"])
    if "notch" in p:
        y = _notch(y, p["notch"])
    if "hp" in p:
        y = sosfiltfilt(butter(4, p["hp"], btype="highpass", fs=FS, output="sos"), y).astype(np.float32)
    if "lp" in p and p["lp"] < FS / 2 - 100:
        y = sosfiltfilt(butter(6, p["lp"], btype="lowpass", fs=FS, output="sos"), y).astype(np.float32)
    if "nr" in p:
        y = StreamingDenoiser(p["nr"], -18.0).process(np.concatenate([y, np.zeros(512, np.float32)]))[512:]
    if "voice" in p:
        y = VoiceShaper(p["voice"], p["voice"], 12.0).process(y)
    if "presence" in p:
        y = _peak(y, 2500.0, p["presence"])
    if "gate" in p:
        y = _gate(y, p["gate"])
    if "comp" in p:
        y = _comp(y, p["comp"])
    if "echo" in p:
        y = _echo(y, p["echo"])
    if "speed" in p and abs(p["speed"] - 1) > 0.01:
        from fractions import Fraction
        fr = Fraction(1 / p["speed"]).limit_denominator(40)
        y = resample_poly(y, fr.numerator, fr.denominator).astype(np.float32)
    if "reverse" in p:
        y = y[::-1].copy()
    # level: speech-like loudness, soft-limited
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    y = np.tanh(1.3 * y * min(2000.0, 0.18 / rms)) / np.tanh(1.3)
    return np.clip(y, -1, 1).astype(np.float32)

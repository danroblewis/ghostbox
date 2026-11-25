"""Stage 2: per-track checks on the demodulated candidate (16 kHz audio) and its
narrowband IQ. Only run on the handful of tracks stage 1 proposes.

Each check returns a score in [0, 1]. `info` carries extras for the UI/metadata
(pitch, voicing fraction, ...).
"""
import numpy as np
from scipy.signal import welch

STAGE2 = ("comb", "pitch_motion", "shape", "voicing", "rhythm", "am_phase", "vad")
FS = 16000

import threading

_vad = {"model": None, "error": None}
_local = threading.local()          # one Silero model per worker thread: it carries per-clip state


def vad_available():
    if _vad["model"] is None and _vad["error"] is None:
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad
            torch.set_num_threads(1)
            _vad["model"] = load_silero_vad()
        except Exception as e:  # pragma: no cover
            _vad["error"] = str(e)
    return _vad["model"] is not None


def _frames(x, n, hop):
    if len(x) < n:
        return np.zeros((0, n), np.float32)
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i + n] for i in idx])


def _pitch_track(audio, f0_min, f0_max):
    """Normalised autocorrelation pitch per 40 ms frame (20 ms hop). Returns (voicing r, f0, energy)."""
    fr = _frames(audio, 640, 320)
    if len(fr) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    fr = fr - fr.mean(axis=1, keepdims=True)
    energy = np.mean(fr ** 2, axis=1)
    lo, hi = int(FS / f0_max), int(FS / f0_min)
    spec = np.fft.rfft(fr * np.hanning(640), n=2048, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, : hi + 1]
    ac /= ac[:, :1] + 1e-12
    seg = ac[:, lo: hi + 1]
    j = seg.argmax(axis=1)
    r = seg[np.arange(len(seg)), j]
    f0 = FS / (lo + j)
    return r, f0, energy


def vad_probs(audio):
    """Silero speech probability per 32 ms block (512 samples at 16 kHz)."""
    if not vad_available():
        return np.zeros(0, np.float32)
    import torch
    n = len(audio) // 512 * 512
    if n == 0:
        return np.zeros(0, np.float32)
    a = np.clip(audio[:n] / (np.abs(audio[:n]).max() + 1e-9), -1, 1).astype(np.float32)
    with torch.no_grad():
        model = getattr(_local, "vad", None)
        if model is None:
            from silero_vad import load_silero_vad
            model = _local.vad = load_silero_vad()
        model.reset_states()
        return np.array([float(model(torch.from_numpy(a[i:i + 512]), FS)) for i in range(0, n, 512)], np.float32)


def run(audio, iq, cfg_checks, enabled, vad=None):
    """vad: precomputed speech probabilities for this audio (incremental, per track); else computed here."""
    """audio: float32 16 kHz demodulated voice; iq: complex64 16 kHz narrowband around the carrier."""
    out, info = {}, {}
    if len(audio) < FS // 2:
        return out, info
    c = cfg_checks
    if not any(enabled.get(k) for k in ("comb", "pitch_motion", "voicing")):
        # the pitch checks now run on the IQ in stage 1; only AM phase and the ML detector remain here
        if enabled.get("am_phase"):
            z = am_cross_z(iq)
            info["am_phase_z"] = float(z)
            out["am_phase"] = float(np.clip(z / c["am_phase"]["full_z"], 0, 1))
        if enabled.get("vad") and vad_available():
            probs = np.sort(vad if vad is not None else vad_probs(audio))
            out["vad"] = float(probs[len(probs) // 2:].mean()) if len(probs) else 0.0
        return out, info
    r, f0, energy = _pitch_track(audio, c["comb"]["f0_min"], c["comb"]["f0_max"])
    loud = energy > np.percentile(energy, 30) if len(energy) else energy.astype(bool)
    voiced = (r > 0.45) & loud
    info["voiced_frac"] = float(voiced.mean()) if len(voiced) else 0.0
    if voiced.any():
        info["f0_median_hz"] = float(np.median(f0[voiced]))

    if enabled.get("comb"):
        top = np.sort(r[loud])[-max(1, int(0.3 * loud.sum())):] if loud.any() else np.zeros(1)
        out["comb"] = float(np.clip((top.mean() - 0.3) / 0.5, 0, 1))

    if enabled.get("pitch_motion"):
        if voiced.sum() >= 5:
            st = 12 * np.log2(f0[voiced])
            rng = np.percentile(st, 90) - np.percentile(st, 10)
            # smoothness: within consecutive voiced frames, steps under 2 semitones
            steps = np.abs(np.diff(12 * np.log2(f0)))[(voiced[1:] & voiced[:-1])]
            smooth = float(np.mean(steps < 2.0)) if len(steps) else 0.0
            moving = float(np.mean(steps > 0.05)) if len(steps) else 0.0  # not a fixed tone
            out["pitch_motion"] = float(np.clip(rng / c["pitch_motion"]["range_st"], 0, 1) * smooth * moving)
            info["pitch_range_st"] = float(rng)
        else:
            out["pitch_motion"] = 0.0

    if enabled.get("shape"):
        f, p = welch(audio, FS, nperseg=1024)
        band = lambda a, b: 10 * np.log10(p[(f >= a) & (f < b)].mean() + 1e-20)
        e1, e2, e3 = band(200, 1000), band(1000, 2000), band(2000, 3400)
        out["shape"] = float(0.5 * np.clip((e1 - e2) / 6, 0, 1) + 0.5 * np.clip((e2 - e3) / 6, 0, 1))

    if enabled.get("voicing"):
        dur = len(audio) / FS
        rate = np.count_nonzero(np.diff(voiced.astype(int))) / max(dur, 1e-3) if len(voiced) > 1 else 0
        info["voicing_switch_hz"] = float(rate)
        out["voicing"] = float(np.clip(rate / 3, 0, 1) if rate <= 10 else np.clip((16 - rate) / 6, 0, 1))

    if enabled.get("rhythm"):
        env = np.sqrt(np.mean(_frames(audio, 160, 160) ** 2, axis=1) + 1e-12)  # 100 Hz envelope
        le = 20 * np.log10(env)
        e = le - le.mean()
        E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
        fe = np.fft.rfftfreq(len(e), 0.01)
        ratio = E[(fe >= 2.5) & (fe <= 7)].sum() / (E[(fe >= 0.5) & (fe <= 16)].sum() + 1e-12)
        dips = np.mean(le < np.percentile(le, 90) - 10)
        dip_score = float(np.clip(1 - abs(dips - 0.3) / 0.3, 0, 1))
        out["rhythm"] = float(0.6 * np.clip((ratio - 0.2) / 0.3, 0, 1) + 0.4 * dip_score)

    if enabled.get("am_phase"):
        z = am_cross_z(iq)
        info["am_phase_z"] = float(z)
        out["am_phase"] = float(np.clip(z / c["am_phase"]["full_z"], 0, 1))

    if enabled.get("vad") and vad_available():
        probs = np.sort(vad if vad is not None else vad_probs(audio))
        out["vad"] = float(probs[len(probs) // 2:].mean()) if len(probs) else 0.0
    return out, info


def am_cross_z(iq, nseg=128, alpha=(300, 3400), step=3):
    """Cross-term AM statistic (see ghostbox.envspec) for one window centred on the carrier.

    Bins pairs (j, j+d) inside +-alpha_hi; keeps only cross terms between pairs; residue
    classes of 3 avoid Hann-window neighbour correlation. Noise ~ N(0,1), AM >> 0, FM < 0.
    """
    m = len(iq) // nseg
    if m < 4:
        return 0.0
    X = np.fft.fftshift(np.fft.fft(iq[: m * nseg].reshape(m, nseg) * np.hanning(nseg), axis=1), axes=1)
    df = FS / nseg
    c = nseg // 2
    w = int(alpha[1] / df)
    X = X[:, c - w: c + w + 1]
    num = var = 0.0
    for d in range(step, 2 * w + 1, step):
        if d * df < alpha[0] or d * df > alpha[1]:
            continue
        a = X[:, d:] * np.conj(X[:, :-d])
        for r in range(step):
            ar = a[:, r::step]
            E = ar.sum(axis=1)
            S = (np.abs(ar) ** 2).sum(axis=1)
            num += float(np.sum(np.abs(E) ** 2 - S))
            var += float(np.sum(S ** 2))
    return num / np.sqrt(var + 1e-30)

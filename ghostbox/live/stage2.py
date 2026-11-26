"""Stage 2: per-track checks on the demodulated candidate (16 kHz audio) and its
narrowband IQ. Only run on the handful of tracks stage 1 proposes.

Each check returns a score in [0, 1]. `info` carries extras for the UI/metadata
(pitch, voicing fraction, ...).
"""
import numpy as np
from scipy.signal import welch

STAGE2 = ("comb", "pitch_motion", "shape", "voicing", "rhythm", "am_phase", "tone", "vad")
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


def tonality(audio, lo=250.0, hi=3400.0):
    """How tone-like the audio is, 0..1: (share of the loud frames whose strongest line sits on one of
    the three most common lines) x (share of the voice-band energy on those lines). A steady tone,
    Morse (a keyed tone), a beat note or FSK data (a few tones) keeps its lines put: near 1 x most
    of the energy. Speech harmonics follow the pitch, so the strongest line wanders and the energy
    spreads. Returns (tonality, frequency of the main line) or (None, None) if too short."""
    n = 4096                                             # 256 ms frames, ~4 Hz bins at 16 kHz
    if len(audio) < 4 * n:
        return None, None
    fr = _frames(audio, n, n // 2)
    P = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    b = (f >= lo) & (f <= hi)
    P, f = P[:, b], f[b]
    e = P.sum(axis=1)
    P = P[e > np.percentile(e, 50)]                      # the louder half: keyed tones are off half the time
    if len(P) < 4 or P.sum() <= 0:
        return None, None
    k = P.argmax(axis=1)
    h = np.convolve(np.bincount(k, minlength=P.shape[1]).astype(float), [1, 1, 1], "same")
    lines, mask = [], np.zeros(P.shape[1], bool)
    for _ in range(3):                                   # the three most common strongest lines
        j = int(h.argmax())
        lines.append(j)
        h[max(0, j - 2): j + 3] = 0
        mask[max(0, j - 1): j + 2] = True
    fixed = float(np.mean(np.min(np.abs(k[:, None] - np.array(lines)[None]), axis=1) <= 1))
    share = float(P[:, mask].sum() / P.sum())
    return fixed * share, float(f[lines[0]])


def keying(iq, min_share=0.15):
    """Sudden carrier steps per second: the channel's strongest line jumping by more than 12 dB within
    20 ms while near full strength. Morse and FSK data key their carrier in a few ms (10+ per second);
    an AM voice's carrier stays on between words, and even fast HF fading takes tens to hundreds of
    ms to swing that far. iq: 16 kHz narrowband containing the carrier. None without a dominant line."""
    n = min(len(iq), 32768)
    if n < 4096:
        return None
    S = np.abs(np.fft.fft(iq[-n:] * np.hanning(n))) ** 2
    f = np.fft.fftfreq(n, 1 / FS)
    ch = np.abs(f) < 3300
    k = int(np.argmax(np.where(ch, S, 0)))
    share = S[np.arange(k - 3, k + 4) % n].sum() / (S[ch].sum() + 1e-30)
    if share < min_share:
        return None
    # the line's strength per 10 ms: +-150 Hz around it, as a magnitude, so the small frequency and
    # phase jumps between demodulated chunks don't read as steps
    m = len(iq) // 160
    x = iq[: m * 160] * np.exp(-2j * np.pi * f[k] * np.arange(m * 160) / FS)
    x = np.convolve(x, np.ones(53, np.float32) / 53, mode="same")
    db = 20 * np.log10(np.abs(x).reshape(m, 160).mean(axis=1) + 1e-12)
    if m < 12:
        return None
    ref = np.percentile(db, 90)
    # a step = the level over the 30 ms before vs the 30 ms after a 20 ms gap differs by > 12 dB, one
    # side near full strength. Holding for 30 ms+ on both sides (a Morse dot is 40 ms+) leaves out the
    # one-frame dips a chunk boundary in the demodulator can cause.
    c = np.concatenate([[0.0], np.cumsum(db)])
    i = np.arange(3, m - 4)
    before, after = (c[i] - c[i - 3]) / 3, (c[i + 5] - c[i + 2]) / 3
    hit = (np.abs(after - before) > 12) & (np.maximum(before, after) > ref - 6)
    edges = np.count_nonzero(np.diff(np.concatenate([[0], hit.astype(int)])) == 1)   # one per step
    return float(edges / (m * 0.01))


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
    if enabled.get("tone"):
        conc, f_tone = tonality(audio)
        if conc is not None:
            info["tonality"], info["tone_hz"] = conc, f_tone
            t = c.get("tone", {})
            lo_, hi_ = float(t.get("voice_frac", 0.22)), float(t.get("tone_frac", 0.45))
            out["tone"] = float(np.clip((hi_ - conc) / max(1e-3, hi_ - lo_), 0, 1))
        steps = keying(iq)
        if steps is not None:
            info["carrier_steps_hz"] = steps                 # faded AM voices: 0-1.5 per second; Morse: 4+
            out["tone"] = min(out.get("tone", 1.0), float(np.clip((4.0 - steps) / 2.5, 0, 1)))
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

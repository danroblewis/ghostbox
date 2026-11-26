"""Composed voice: play several demodulations at once, chosen so that *together* they sound like a voice.

No tracks. The candidates are frequencies, found on the layered confidence map (engine); each one's
"audio" is read straight off the fine IQ spectrum: the power at offsets 0..3.4 kHz above and below it,
added (what an AM or SSB demodulator there would turn into sound). Each hop:

  1. the mix is scored on the *sum* of its members' sideband spectrograms (sounds that don't share a
     phase add in power), so trying a candidate costs one addition, not a demodulation or an ML pass:
       pitch      - a harmonic comb (lines on one fundamental) whose pitch moves smoothly
       voice2     - a second comb on another note (a second vocal cord)
       formants   - two independently gliding resonances (F1 250-1000, F2 850-3000 Hz)
       syllables  - loudness pulsing at syllable rate (2-8 Hz) with real depth
  2. greedy: from the set already playing, add whichever candidate best fills what's missing, drop one
     that no longer helps; members stay at least hold_s. One that has every part plays alone.
     No target words or melody: candidates are only picked and levelled, never shifted or reshaped,
     so whatever comb or rhythm is heard came from the IQ.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .stage2 import formant_score

PARTS = ("pitch", "voice2", "formants", "syllables")


class Grid:
    """Frequency / time layout of the spectrograms being scored: bins df Hz apart from 0 Hz, frames hop_s apart."""

    def __init__(self, df, hop_s, comb=(4.0, 8.0), top_hz=3400.0):
        self.df, self.hop = float(df), float(hop_s)
        self.cmb0, self.cmb1 = comb                     # comb salience (dB) where the pitch score starts / its span
                                                        # (noise reaches ~cmb0 at its 90th percentile on this grid)
        self.top = int(top_hz / df) + 1
        self.F = np.arange(self.top) * self.df
        self.v1, self.v2 = int(np.searchsorted(self.F, 300)), int(np.searchsorted(self.F, 3000))
        # harmonic comb lookup: fundamentals 80-400 Hz, their first 8 harmonics below 3 kHz, and the
        # midpoints between harmonics (where a comb has troughs)
        self.F0 = np.geomspace(80, 400, 56)
        H = np.arange(1, 9)
        valid = np.outer(self.F0, H + 0.5) <= 3000
        self.hb = np.where(valid, np.rint(np.outer(self.F0, H) / df), 0).astype(int)
        self.mb = np.where(valid, np.rint(np.outer(self.F0, H + 0.5) / df), 0).astype(int)
        self.valid, self.nh = valid, valid.sum(axis=1)
        self.smooth = max(1, int(round(0.06 / hop_s)))      # comb: power averaged over ~60 ms
        self.drift = max(3, int(round(0.5 / hop_s)))        # syllables: slow drift removed (> 0.5 s)


AUDIO = Grid(16000 / 1024, 320 / 16000)                 # demodulated 16 kHz audio (tests, replays)


def iq_grid(df, hop_s):
    """Sidebands read off the receiver's fine IQ spectrum (25 Hz bins, 40 ms frames): coarser, so plain
    noise's best comb sits lower (~2.2 dB at its 90th percentile) and real voices reach 3-7 dB."""
    return Grid(df, hop_s, comb=(2.8, 5.0))


def spectrogram(x, fs=16000, nfft=1024, hop=320):
    """Power spectrogram of audio on the AUDIO grid, scaled to unit mean power in 300-3000 Hz."""
    if len(x) < nfft + 10 * hop:
        return None
    idx = np.arange(0, len(x) - nfft + 1, hop)
    fr = np.stack([x[i:i + nfft] for i in idx]) * np.hanning(nfft).astype(np.float32)
    P = (np.abs(np.fft.rfft(fr, axis=1)[:, :AUDIO.top]) ** 2).astype(np.float32)
    return normalise(P, AUDIO)


def normalise(P, g):
    lvl = float(P[:, g.v1:g.v2].sum(axis=1).mean())
    return None if lvl <= 1e-12 else (P / lvl).astype(np.float32)


def _combs(M, loud, g):
    """Per frame: (best comb salience dB, its f0, second comb salience dB, its f0). Salience = mean over a
    fundamental's harmonics of (level at the harmonic - level midway to the next). Power is averaged over
    ~60 ms first and both sides take the best of 3 bins, so plain noise averages ~0 dB."""
    L = 10 * np.log10(uniform_filter1d(M, g.smooth, axis=0) + 1e-9)
    Lm = maximum_filter1d(L, 3, axis=1)[loud]
    c = np.clip(Lm[:, g.hb] - Lm[:, g.mb], -15, 15) * g.valid  # [frames, f0, harmonic]
    sal = c.sum(axis=2) / g.nh                                  # [frames, f0]
    j1 = sal.argmax(axis=1)
    s1 = sal[np.arange(len(sal)), j1]
    # second comb: a different note, not an octave or a fifth-ish alias of the first
    r = np.log2(g.F0[None, :] / g.F0[j1][:, None])
    near = (np.abs(r) < 0.09) | (np.abs(np.abs(r) - 1) < 0.09) | (np.abs(np.abs(r) - 0.585) < 0.05)
    sal2 = np.where(near, -99, sal)
    j2 = sal2.argmax(axis=1)
    s2 = np.maximum(0, sal2[np.arange(len(sal2)), j2])
    return s1, g.F0[j1], s2, g.F0[j2]


def _motion(f0, ok):
    """How much a pitch track moves *smoothly*, 0..1: its spread in semitones over the frames where it's
    present x the share of neighbouring present frames that stay within a semitone. A steady buzz or a
    fixed set of tones has no spread; noise has spread but jumps about."""
    if ok.sum() < 6:
        return 0.0
    st = 12 * np.log2(f0)
    pair = ok[1:] & ok[:-1]
    if pair.sum() < 3:
        return 0.0
    cont = float(np.mean(np.abs(np.diff(st))[pair] < 1.0))
    return float(np.clip((np.std(st[ok]) - 0.3) / 1.2, 0, 1) * cont)


def score(M, g=AUDIO):
    """Voice parts of a (summed) spectrogram M [frames, g.top], each 0..1, plus info."""
    M = np.maximum(M, 0)                                 # (a member taken away can leave rounding below 0)
    e = M[:, g.v1:g.v2].sum(axis=1)
    loud = e > np.percentile(e, 40)
    out, info = {}, {}
    if loud.sum() < 6:
        return {p: 0.0 for p in PARTS}, info
    s1, f1, s2, f2 = _combs(M, loud, g)
    p1, p2 = np.clip((s1 - g.cmb0) / g.cmb1, 0, 1), np.clip((s2 - g.cmb0) / g.cmb1, 0, 1)
    out["pitch"] = float(p1.mean() * (0.25 + 0.75 * _motion(f1, p1 > 0.3)))
    out["voice2"] = float(p2.mean() * (0.25 + 0.75 * _motion(f2, p2 > 0.3)))
    on1, on2 = p1 > 0.3, p2 > 0.3
    if on1.sum() >= 3:
        info["f0_hz"] = float(np.median(f1[on1]))
    if on2.sum() >= 3:
        info["f0b_hz"] = float(np.median(f2[on2]))
    fs_, finfo = formant_score(M, g.F, g.hop)
    out["formants"] = float(fs_ or 0.0)
    info.update({k: v for k, v in finfo.items() if k in ("f1_hz", "f2_hz")})
    # syllables: the loudness envelope pulsing at 2-8 Hz, with some depth
    le = 10 * np.log10(e + 1e-9)
    depth = float(np.percentile(le, 90) - np.percentile(le, 10))
    x = le - np.convolve(le, np.ones(g.drift) / g.drift, "same")
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fm = np.fft.rfftfreq(len(x), g.hop)
    band, tot = sp[(fm >= 2) & (fm <= 8)].sum(), sp[(fm >= 0.5) & (fm <= 20)].sum() + 1e-12
    out["syllables"] = float(np.clip((band / tot - 0.35) / 0.35, 0, 1) * np.clip((depth - 4) / 8, 0, 1))
    return out, info


def total(parts, n, cfg):
    w = {p: float(cfg.get("w_" + p, 1.0)) for p in PARTS}
    base = w["pitch"] + w["formants"] + w["syllables"] or 1.0  # voice2 is a bonus on top
    return sum(w[p] * parts[p] for p in PARTS) / base - float(cfg.get("cost", 0.02)) * n


class Composer:
    """Chooses, hop by hop, which candidates play. Holds no audio itself."""

    def __init__(self):
        self.members = {}                    # candidate key -> stream time it joined

    def choose(self, specs, t, cfg, g=AUDIO):
        """specs: {key: spectrogram} (same frames, grid g). Returns (members {key: role}, parts, total, info)."""
        mx = int(cfg.get("max_components", 4))
        mn = min(int(cfg.get("min_components", 1)), mx)
        hold = float(cfg.get("hold_s", 2.0))
        self.members = {k: v for k, v in self.members.items() if k in specs}
        if not specs:
            return {}, {p: 0.0 for p in PARTS}, 0.0, {}
        M = np.zeros(next(iter(specs.values())).shape, np.float32)
        for k in self.members:
            M = M + specs[k]

        def value(Mx, n):
            return -1.0 if n == 0 else total(score(Mx, g)[0], n, cfg)

        cur = value(M, len(self.members))
        margin = float(cfg.get("margin", 0.005))
        for _ in range(int(cfg.get("moves", 3))):
            n, moves = len(self.members), []
            if n < mx:
                adds = [(value(M + S, n + 1), k) for k, S in specs.items() if k not in self.members]
                if adds:
                    v, k = max(adds)
                    moves.append((v, "add", k, n < mn))       # below the minimum: add the best regardless
            if n > mn:
                drops = [(value(M - specs[k], n - 1), k) for k, since in self.members.items() if t - since >= hold]
                if drops:
                    v, k = max(drops)
                    moves.append((v, "drop", k, False))
            forced = [m for m in moves if m[3]]
            if forced:
                v, kind, k, _ = forced[0]
            elif moves and max(moves)[0] > cur + margin:
                v, kind, k, _ = max(moves)
            else:
                break
            if kind == "add":
                self.members[k] = t
                M = M + specs[k]
            else:
                del self.members[k]
                M = M - specs[k]
            cur = v
        parts, info = score(M, g) if self.members else ({p: 0.0 for p in PARTS}, {})
        tot = total(parts, len(self.members), cfg) if self.members else 0.0
        # what each member gives: the part that falls most without it
        roles = {}
        for k in self.members:
            if len(self.members) == 1:
                roles[k] = max(PARTS, key=lambda p: parts[p]) if max(parts.values()) > 0.02 else "texture"
                continue
            without = score(M - specs[k], g)[0]
            d = {p: parts[p] - without[p] for p in PARTS}
            p = max(d, key=d.get)
            roles[k] = p if d[p] > 0.02 else "texture"
        return roles, parts, max(0.0, tot), info

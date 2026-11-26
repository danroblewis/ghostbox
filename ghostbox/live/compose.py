"""Composed voice: one voice assumed split across several channels, put back together.

No tracks. Candidates are frequencies found on the layered confidence map (engine). Each candidate
channel's sound is read straight off the fine IQ spectrum (power 0..3.4 kHz above and below it,
added) and split into *keys*: its low (250-900 Hz, around F1), mid (900-1800, F2) and high
(1800-3400) bands. Playing a key = that channel's demodulated audio band-passed to that band, set to
a speech-like level for the band. That is the only processing: filtered and levelled, never shifted.

A set of keys is scored on what one voice needs, using the sum of their spectrograms (sounds that
don't share a phase add in power, so trying a key costs one addition):
  pitch      a harmonic comb whose pitch moves smoothly (the keys' harmonics line up on one f0)
  formants   two independently gliding resonances (F1 and F2)
  syllables  loudness pulsing at syllable rate (2-8 Hz) with depth
  together   keys from *different* channels pulsing together: the parts of one voice move at once
  coverage   low, mid and high bands all present, as in speech
Each hop a greedy search starts from the keys already playing, adds the key that helps most (until
at least min_channels channels play), drops ones that stopped helping (after hold_s). No target
words or melody are involved anywhere.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .stage2 import formant_score

PARTS = ("pitch", "formants", "syllables", "together", "coverage")
# (name, low Hz, high Hz, level dB relative to the low band: a speech-like spectral tilt)
BANDS = [("low", 250.0, 900.0, 0.0), ("mid", 900.0, 1800.0, -4.0), ("high", 1800.0, 3400.0, -9.0)]


class Grid:
    """Frequency / time layout of the spectrograms being scored: bins df Hz apart from 0 Hz, frames hop_s apart."""

    def __init__(self, df, hop_s, comb=(4.0, 8.0), top_hz=3400.0):
        self.df, self.hop = float(df), float(hop_s)
        self.cmb0, self.cmb1 = comb                     # comb salience (dB) where the pitch score starts / its span
                                                        # (noise reaches ~cmb0 at its 90th percentile on this grid)
        self.top = int(top_hz / df) + 1
        self.F = np.arange(self.top) * self.df
        self.v1, self.v2 = int(np.searchsorted(self.F, 250)), int(np.searchsorted(self.F, 3400))
        # harmonic comb lookup: fundamentals 80-400 Hz, their first 8 harmonics below 3 kHz, and the
        # midpoints between harmonics (where a comb has troughs)
        self.F0 = np.geomspace(80, 400, 56)
        H = np.arange(1, 9)
        valid = np.outer(self.F0, H + 0.5) <= 3000
        self.hb = np.where(valid, np.rint(np.outer(self.F0, H) / df), 0).astype(int)
        self.mb = np.where(valid, np.rint(np.outer(self.F0, H + 0.5) / df), 0).astype(int)
        self.valid, self.nh = valid, valid.sum(axis=1)
        self.smooth = max(1, int(round(0.06 / hop_s)))      # comb: power averaged over ~60 ms
        self.drift = max(3, int(round(0.5 / hop_s)))        # envelopes: slow drift removed (> 0.5 s)
        # band masks with soft (~60 Hz) edges
        self.masks = []
        for _, lo, hi, _ in BANDS:
            m = np.clip((self.F - lo) / 60 + 0.5, 0, 1) * np.clip((hi - self.F) / 60 + 0.5, 0, 1)
            self.masks.append(m.astype(np.float32))


AUDIO = Grid(16000 / 1024, 320 / 16000)                 # demodulated 16 kHz audio (tests, replays)


def iq_grid(df, hop_s):
    """Sidebands read off the receiver's fine IQ spectrum (25 Hz bins, 40 ms frames): coarser, so plain
    noise's best comb sits lower (~2.2 dB at its 90th percentile) and real voices reach 3-7 dB."""
    return Grid(df, hop_s, comb=(2.8, 5.0))


def spectrogram(x, fs=16000, nfft=1024, hop=320):
    """Power spectrogram of audio on the AUDIO grid, scaled to unit mean power in 250-3400 Hz."""
    if len(x) < nfft + 10 * hop:
        return None
    idx = np.arange(0, len(x) - nfft + 1, hop)
    fr = np.stack([x[i:i + nfft] for i in idx]) * np.hanning(nfft).astype(np.float32)
    P = (np.abs(np.fft.rfft(fr, axis=1)[:, :AUDIO.top]) ** 2).astype(np.float32)
    return normalise(P, AUDIO)


def normalise(P, g):
    lvl = float(P[:, g.v1:g.v2].sum(axis=1).mean())
    return None if lvl <= 1e-12 else (P / lvl).astype(np.float32)


def _envelope(P, g):
    """Syllable-rate loudness of a spectrogram: log power per frame minus its slow drift, z-scored."""
    le = 10 * np.log10(P.sum(axis=1) + 1e-9)
    x = le - np.convolve(le, np.ones(g.drift) / g.drift, "same")
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-6 else np.zeros_like(x)


def keys_of(A, g, channel):
    """A channel's sideband spectrogram A -> its keys {band index: key}; a key carries its spectrogram
    (band only, at the band's speech level), its envelope and its channel."""
    out = {}
    for i, (name, lo, hi, lvl) in enumerate(BANDS):
        S = A * g.masks[i]
        p = float(S.sum(axis=1).mean())
        if p <= 1e-12:
            continue
        S = (S * (10 ** (lvl / 10) / p)).astype(np.float32)
        out[i] = {"S": S, "env": _envelope(S, g), "band": i, "ch": channel}
    return out


def _combs(M, loud, g):
    """Per frame: (best comb salience dB, its f0). Salience = mean over a fundamental's harmonics of (level
    at the harmonic - level midway to the next). Power is averaged over ~60 ms first and both sides take
    the best of 3 bins, so plain noise averages ~0 dB."""
    L = 10 * np.log10(uniform_filter1d(M, g.smooth, axis=0) + 1e-9)
    Lm = maximum_filter1d(L, 3, axis=1)[loud]
    c = np.clip(Lm[:, g.hb] - Lm[:, g.mb], -15, 15) * g.valid  # [frames, f0, harmonic]
    sal = c.sum(axis=2) / g.nh                                  # [frames, f0]
    j = sal.argmax(axis=1)
    return sal[np.arange(len(sal)), j], g.F0[j]


def _motion(f0, ok):
    """How much a pitch track moves *smoothly*, 0..1: its spread in semitones over the frames where it's
    present x the share of neighbouring present frames that stay within a semitone."""
    if ok.sum() < 6:
        return 0.0
    st = 12 * np.log2(f0)
    pair = ok[1:] & ok[:-1]
    if pair.sum() < 3:
        return 0.0
    cont = float(np.mean(np.abs(np.diff(st))[pair] < 1.0))
    return float(np.clip((np.std(st[ok]) - 0.3) / 1.2, 0, 1) * cont)


def score(M, g=AUDIO, members=()):
    """Voice parts of a set of keys: M is the sum of their spectrograms [frames, g.top]; members the keys
    themselves (for together / coverage). Each part 0..1, plus info."""
    M = np.maximum(M, 0)                                 # (a member taken away can leave rounding below 0)
    e = M[:, g.v1:g.v2].sum(axis=1)
    out = {p: 0.0 for p in PARTS}
    info = {}
    loud = e > np.percentile(e, 40)
    if loud.sum() >= 6:
        s1, f1 = _combs(M, loud, g)
        p1 = np.clip((s1 - g.cmb0) / g.cmb1, 0, 1)
        out["pitch"] = float(p1.mean() * (0.25 + 0.75 * _motion(f1, p1 > 0.3)))
        if (p1 > 0.3).sum() >= 3:
            info["f0_hz"] = float(np.median(f1[p1 > 0.3]))
        fs_, finfo = formant_score(M, g.F, g.hop)
        out["formants"] = float(fs_ or 0.0)
        info.update({k: v for k, v in finfo.items() if k in ("f1_hz", "f2_hz")})
        le = 10 * np.log10(e + 1e-9)
        depth = float(np.percentile(le, 90) - np.percentile(le, 10))
        x = le - np.convolve(le, np.ones(g.drift) / g.drift, "same")
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        fm = np.fft.rfftfreq(len(x), g.hop)
        band, tot = sp[(fm >= 2) & (fm <= 8)].sum(), sp[(fm >= 0.5) & (fm <= 20)].sum() + 1e-12
        out["syllables"] = float(np.clip((band / tot - 0.35) / 0.35, 0, 1) * np.clip((depth - 4) / 8, 0, 1))
    # together: pairs of keys from different channels whose syllable-rate loudness moves together
    cors = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            if a["ch"] != b["ch"]:
                cors.append(float(np.mean(a.get("envr", a["env"]) * b.get("envr", b["env"]))))
    out["together"] = float(np.clip((np.mean(cors) - 0.15) / 0.45, 0, 1)) if cors else 0.0
    out["coverage"] = len({m["band"] for m in members}) / len(BANDS)
    return out, info


def total(parts, n, cfg):
    w = {p: float(cfg.get("w_" + p, 1.0)) for p in PARTS}
    base = sum(w.values()) or 1.0
    return sum(w[p] * parts[p] for p in PARTS) / base - float(cfg.get("cost", 0.01)) * n


class Composer:
    """Chooses, hop by hop, which keys play. Holds no audio itself."""

    def __init__(self):
        self.members = {}                    # key id -> stream time it joined

    def choose(self, keys, t, cfg, g=AUDIO):
        """keys: {key id: key} (same frames, grid g). Returns (members {id: role}, parts, total, info)."""
        mx = int(cfg.get("max_components", 6))
        min_ch = int(cfg.get("min_channels", 2))
        hold = float(cfg.get("hold_s", 2.0))
        self.members = {k: v for k, v in self.members.items() if k in keys}
        none = ({}, {p: 0.0 for p in PARTS}, 0.0, {})
        if not keys:
            return none
        M = np.zeros(next(iter(keys.values()))["S"].shape, np.float32)
        for k in self.members:
            M = M + keys[k]["S"]
        n_ch = len({keys[k]["ch"] for k in keys})
        # "together" must be specific: interference pulses and receiver gain changes move every channel at
        # once. Take out what the whole band does (the median envelope) before correlating.
        E = np.stack([key["env"] for key in keys.values()])
        common = np.median(E, axis=0)
        for key in keys.values():
            r = key["env"] - common
            sd = r.std()
            key["envr"] = (r - r.mean()) / sd if sd > 1e-6 else np.zeros_like(r)
        by_ch = {}
        for k, key in keys.items():
            by_ch.setdefault(key["ch"], []).append(k)

        def value(Mx, mem):
            return -1.0 if not mem else total(score(Mx, g, [keys[k] for k in mem])[0], len(mem), cfg)

        def chans(mem):
            return {keys[k]["ch"] for k in mem}

        cur = value(M, list(self.members))
        margin = float(cfg.get("margin", 0.005))
        for _ in range(int(cfg.get("moves", 4))):
            mem = list(self.members)
            need = len(chans(mem)) < min(min_ch, n_ch)           # too few channels: must add one from a new channel
            moves = []
            if len(mem) < mx or need:
                have = chans(mem)
                adds = [(value(M + key["S"], mem + [k]), (k,)) for k, key in keys.items()
                        if k not in self.members and (not need or key["ch"] not in have)]
                # a whole channel at once (all its bands): a voice that is all there in one channel
                # only looks good complete, so one band at a time would never bring it in
                for ch, ks in by_ch.items():
                    ks = tuple(k for k in ks if k not in self.members)
                    if len(ks) > 1 and len(mem) + len(ks) <= max(mx, len(mem) + 1) and (not need or ch not in have):
                        adds.append((value(M + sum(keys[k]["S"] for k in ks), mem + list(ks)), ks))
                if adds:
                    v, ks = max(adds, key=lambda x: x[0])
                    moves.append((v, "add", ks, need))
            if len(mem) > 1 and not need:
                drops = []
                for k, since in self.members.items():
                    rest = [m for m in mem if m != k]
                    if t - since >= hold and len(chans(rest)) >= min(min_ch, n_ch):
                        drops.append((value(M - keys[k]["S"], rest), k))
                if drops:
                    v, k = max(drops)
                    moves.append((v, "drop", k, False))
            forced = [m for m in moves if m[3]]
            best = max(moves, key=lambda x: x[0]) if moves else None
            if forced:
                v, kind, k, _ = forced[0]
            elif best and best[0] > cur + margin:
                v, kind, k, _ = best
            else:
                break
            if kind == "add":
                for k_ in k:
                    self.members[k_] = t
                    M = M + keys[k_]["S"]
            else:
                del self.members[k]
                M = M - keys[k]["S"]
            cur = v
        if not self.members:
            return none
        mem = list(self.members)
        parts, info = score(M, g, [keys[k] for k in mem])
        tot = total(parts, len(mem), cfg)
        # what each key gives: the part that falls most without it
        roles = {}
        for k in mem:
            rest = [m for m in mem if m != k]
            without = score(M - keys[k]["S"], g, [keys[m] for m in rest])[0] if rest else {p: 0.0 for p in PARTS}
            d = {p: parts[p] - without[p] for p in PARTS}
            p = max(d, key=d.get)
            roles[k] = p if d[p] > 0.01 else "fill"
        info["channels"] = len(chans(mem))
        return roles, parts, max(0.0, tot), info

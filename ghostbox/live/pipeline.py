"""The signal pipeline: IQ in, radio steps, the sum into sound, audio steps, out.

    raw IQ ──► radio steps ──► SUM ──► audio steps ──► level ──► speaker
               (IQ in, IQ out:      (sound in, sound out)
                the whole band)

Radio steps act on the IQ of the whole band and return IQ of the same length; audio steps act on
the sound and return sound of the same length. Steps are added one at a time (STEPS, below).

SUM (SumToAudio) is the one step that crosses from radio to sound: the total power of all the IQ,
moment by moment (by Parseval's theorem, the same as adding up every column of a spectrum), as
16,000 samples a second. Whatever the radio steps leave in the band is what is heard. It is
averaged in fine slices and low-passed to what 16 kHz sound can hold, so faster power changes are
removed rather than folding into the sound; its steady part (the band's average power: silent, but
it would use up the speaker's whole range) is taken out with a one-pole high-pass at 20 Hz.

Level (audio.SmoothGain) comes last and is always there: the sum's size depends on the gain and the
band, so it is scaled slowly toward a comfortable loudness.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import butter, firwin, sosfilt, sosfilt_zi

from .audio import SmoothGain

AUDIO_FS = 16000


# ---------------- radio steps (IQ in, IQ out) ----------------
class _Spectral:
    """Base for radio steps that work frequency by frequency: overlapping FFT blocks (sqrt-Hann
    windows at half overlap, which add back to exactly the input when every gain is 1), each block's
    spectrum multiplied by a per-frequency gain (gains(), in FFT order), then back to IQ. The output is
    delayed by one block, so it has the same length as the input. bin_hz sets the frequency resolution
    (the block is fs / bin_hz samples, rounded up to a power of two)."""

    bin_hz = 500.0

    def __init__(self, fs, fc=0.0):
        self.fs, self.fc = float(fs), float(fc)
        self.N = int(2 ** np.ceil(np.log2(self.fs / self.bin_hz)))
        self.H = self.N // 2
        self.df = self.fs / self.N
        self.f = np.fft.fftfreq(self.N, 1 / self.fs)            # each bin's offset from the centre (FFT order)
        self.w = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.N) / self.N)).astype(np.float32)
        self.inb = np.zeros(0, np.complex64)                    # input not yet in a full block
        self.ola = np.zeros(self.N - self.H, np.complex64)      # overlap-add carried to the next call
        self.fifo = np.zeros(self.N, np.complex64)              # output, one block behind the input
        self.g_last = None                                      # the gains last applied (FFT order)

    def curve(self, off):
        """This step's mask: the amplitude gain it applies at frequency offsets `off` (Hz from the
        centre), as last applied (None before the first block)."""
        g = getattr(self, "mask", None) if getattr(self, "mask", None) is not None else self.g_last
        if g is None:
            return None
        return np.interp(off, np.fft.fftshift(self.f), np.fft.fftshift(np.broadcast_to(g, (self.N,))))

    def gains(self, F):
        """Per-bin amplitude gain for these frames' spectra F [frames, N] (FFT order)."""
        return np.ones(self.N, np.float32)

    def process(self, x):
        N, H = self.N, self.H
        inb = np.concatenate([self.inb, x.astype(np.complex64)])
        nfr = (len(inb) - N) // H + 1 if len(inb) >= N else 0
        if nfr:
            frames = sliding_window_view(inb[:(nfr - 1) * H + N], N)[::H]     # views, no copy
            F = sfft.fft(frames * self.w, axis=1, workers=-1)                   # single precision, all cores
            self.g_last = self.gains(F)
            F *= self.g_last
            Y = sfft.ifft(F, axis=1, workers=-1, overwrite_x=True)
            Y *= self.w
            # overlap-add at half overlap: each output half-block = this frame's first half + the
            # previous frame's second half (the first one continues the last call's tail)
            prev = np.concatenate([self.ola[None, :], Y[:-1, H:]])
            done = (Y[:, :H] + prev).reshape(-1)
            self.ola = Y[-1, H:].copy()
            self.inb = inb[nfr * H:]
            self.fifo = np.concatenate([self.fifo, done])
        else:
            self.inb = inb
        out, self.fifo = self.fifo[:len(x)], self.fifo[len(x):]
        if len(out) < len(x):                                    # (can't happen once running)
            out = np.concatenate([out, np.zeros(len(x) - len(out), np.complex64)])
        return out


class Flatten(_Spectral):
    """Flatten receiver (an equalizer on the IQ).

    A receiver doesn't pass every frequency in its span equally: the edges roll off and the middle
    ripples. Signals that move in frequency (FM) then change in measured power as they move across
    those slopes, and the sum hears them ("slope detection"). This step learns the receiver's own
    response and evens it out:

      1. each frequency's power is averaged over the last `learn_s` seconds;
      2. the receiver's response is taken as the noise floor under that: the 20th percentile over a
         sliding window `window_khz` wide (several stations wide, so the gaps between stations set
         it and stations themselves don't count), smoothed;
      3. every frequency is boosted or cut by the difference between that response and its median,
         times `strength` (0 = off, 1 = fully flat), never more than `max_db` either way.

    Bins of ~250-500 Hz; the output is about 2 ms behind the input."""

    domain, label = "radio", "Flatten receiver"
    desc = ("Learns how the receiver's sensitivity varies across the span (from the noise floor, so stations "
            "don't count) and evens it out. Signals that move in frequency, like FM, are heard by the sum because "
            "they move across the receiver's slopes; flattening the slopes fades them.")
    params = [{"key": "strength", "label": "Strength", "min": 0, "max": 1, "step": 0.05, "unit": "%"},
              {"key": "learn_s", "label": "Learn over", "min": 1, "max": 60, "step": 1, "unit": "s"},
              {"key": "max_db", "label": "Boost / cut at most", "min": 3, "max": 30, "step": 1, "unit": "dB"},
              {"key": "window_khz", "label": "Ignore signals narrower than", "min": 100, "max": 3000, "step": 50, "unit": "kHz"}]
    default = {"strength": 1.0, "learn_s": 10.0, "max_db": 20.0, "window_khz": 1000.0}

    def __init__(self, fs, fc=0.0, **p):
        super().__init__(fs, fc)
        self.M = None                                           # per-bin average power (FFT order)
        self.set(**p)

    def set(self, strength=1.0, learn_s=10.0, max_db=20.0, window_khz=1000.0):
        self.strength, self.learn_s, self.max_db = float(strength), float(learn_s), float(max_db)
        self.window_khz = float(window_khz)

    def response(self):
        """The receiver's learned response in dB across the span (low to high frequency)."""
        db = 10 * np.log10(np.fft.fftshift(self.M) + 1e-20)
        win = max(15, int(self.window_khz * 1e3 / self.df) | 1)
        floor = percentile_filter(db, 20, size=win, mode="nearest")
        return uniform_filter1d(floor, size=max(3, win // 4), mode="nearest")

    def gains(self, F):
        # learn: these frames' power folds into the running average with weight (their length) / (learn_s * fs)
        P = (F.real ** 2 + F.imag ** 2).mean(axis=0)
        a = min(1.0, len(F) * self.H / (self.learn_s * self.fs))
        self.M = P if self.M is None else self.M + a * (P - self.M)
        if self.strength <= 0:
            return np.ones(self.N, np.float32)
        R = self.response()
        g_db = np.clip(np.median(R) - R, -self.max_db, self.max_db) * self.strength
        return np.fft.ifftshift(10 ** (g_db / 20)).astype(np.float32)


def ranges_of(p):
    """A step's ranges as [(lo, hi)] absolute Hz, from its "ranges" list (or an older single centre/width)."""
    out = []
    for r in p.get("ranges") or []:
        try:
            lo, hi = float(r["lo"]), float(r["hi"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi > lo:
            out.append((lo, hi))
    if not out and p.get("center_hz"):
        c, w = float(p["center_hz"]), float(p.get("width_hz") or 6000)
        out.append((c - w / 2, c + w / 2))
    return out


class _Ranges(_Spectral):
    """Base for the range masks: a list of frequency ranges, each 1 inside, falling to 0 over `edge_hz`
    outside (raised cosine, so the edges don't ring); the ranges together are their union (a frequency
    is in the mask if it is in any range). Bins of ~60-120 Hz, so even narrow ranges have clean edges;
    the output is 15-30 ms behind the input."""

    bin_hz = 100.0
    params = [{"key": "ranges", "label": "Ranges", "kind": "ranges"},
              {"key": "edge_hz", "label": "Soft edges", "min": 0, "max": 5000, "step": 50, "unit": "Hz"},
              {"key": "click_width_hz", "label": "Width of a clicked range", "min": 500, "max": 50000, "step": 250, "unit": "Hz"}]
    default = {"ranges": [], "edge_hz": 300.0, "click_width_hz": 6000.0}

    def __init__(self, fs, fc=0.0, **p):
        super().__init__(fs, fc)
        self.key = None
        self.set(**p)

    def union(self, ranges, edge_hz):
        m = np.zeros(self.N, np.float32)
        edge = max(float(edge_hz), self.df)
        for lo, hi in ranges:
            c, half = (lo + hi) / 2 - self.fc, (hi - lo) / 2
            t = np.clip((np.abs(self.f - c) - half) / edge, 0, 1)    # 0 inside, 1 beyond the soft edge
            m = np.maximum(m, (0.5 + 0.5 * np.cos(np.pi * t)).astype(np.float32))
        return m

    def set(self, **p):
        key = json_key(p)
        if key == self.key:
            return
        self.key = key
        self.mask = self.shape(self.union(ranges_of(p), p.get("edge_hz", 300.0)))

    def gains(self, F):
        return self.mask


class BandPass(_Ranges):
    """Band-pass: keep only the ranges (all of them: one window or several) and set every other
    frequency to zero. SUM then plays what is in them: how their energy rises and falls."""

    domain, label = "radio", "Band-pass"
    desc = ("Keeps only its ranges and removes everything else, so the sound is those ranges alone. Several ranges "
            "in one Band-pass are all kept (two Band-pass steps in a row would keep only what is in both). On an AM "
            "signal it plays the voice; on the edge of an FM station it plays the station too, because the station "
            "swings in and out of the range (slope detection); across a whole FM station it is almost silent.")

    def shape(self, union):
        return union


class BandStop(_Ranges):
    """Band-stop: remove the ranges (all of them) and keep everything else."""

    domain, label = "radio", "Band-stop"
    desc = ("Removes its ranges and keeps everything else: known stations, steady lines, interference. Drag across "
            "the waterfall with Band-stop (or shift-drag) to add a range.")

    def shape(self, union):
        return (1 - union).astype(np.float32)


def json_key(p):
    import json
    return json.dumps(p, sort_keys=True, default=float)


# ---------------- audio steps (sound in, sound out) ----------------
EQ_BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 6300)


def _peaking(f0, gain_db, q=1.41):
    """One band of a graphic equalizer: a peaking filter (RBJ audio-EQ cookbook) as a biquad."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / AUDIO_FS
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return np.concatenate([b / a[0], a / a[0]])


class AudioEQ:
    """Audio EQ: a graphic equalizer on the sound, one band per octave from 63 Hz to 6.3 kHz, each
    a peaking filter that boosts or cuts around its frequency (0 dB = unchanged)."""

    domain, label = "audio", "Audio EQ"
    desc = "A graphic equalizer on the sound: boost or cut each octave band, 63 Hz to 6.3 kHz."
    params = [{"key": f"g{f}", "label": (f"{f // 1000:g} kHz" if f >= 1000 and f % 1000 == 0 else f"{f / 1000:g} kHz" if f >= 1000 else f"{f} Hz"),
               "min": -15, "max": 15, "step": 0.5, "unit": "dB"} for f in EQ_BANDS]
    default = {f"g{f}": 0.0 for f in EQ_BANDS}

    def __init__(self, fs, fc=0.0, **p):
        self.zi = None
        self.set(**p)

    def set(self, **p):
        g = {f: float(p.get(f"g{f}", 0.0)) for f in EQ_BANDS}
        self.sos = np.array([_peaking(f, g[f]) for f in EQ_BANDS])
        if self.zi is None:
            self.zi = np.zeros((len(EQ_BANDS), 2))

    def process(self, y):
        out, self.zi = sosfilt(self.sos, y, zi=self.zi)
        return out.astype(np.float32)


class LowPass:
    """Low-pass: removes everything in the sound above `cutoff_hz`, steeply (a Butterworth filter of
    `order`; each order adds 6 dB per octave of cut beyond the cutoff: order 8 = 48 dB per octave)."""

    domain, label = "audio", "Low-pass"
    desc = ("Cuts everything in the sound above the cutoff, steeply. The sound can hold frequencies up to 8 kHz; "
            "a low-pass at 7 kHz removes a whine at the very top.")
    params = [{"key": "cutoff_hz", "label": "Cutoff", "min": 200, "max": 7900, "step": 50, "unit": "Hz"},
              {"key": "order", "label": "Steepness (order)", "min": 2, "max": 12, "step": 2, "unit": ""}]
    default = {"cutoff_hz": 7000.0, "order": 8}

    def __init__(self, fs, fc=0.0, **p):
        self.zi, self.key = None, None
        self.set(**p)

    def set(self, cutoff_hz=7000.0, order=8):
        key = (float(cutoff_hz), int(order))
        if key == self.key:
            return
        self.key = key
        self.sos = butter(int(order), min(float(cutoff_hz), 0.49 * AUDIO_FS), btype="lowpass", fs=AUDIO_FS, output="sos")
        if self.zi is None or self.zi.shape != (len(self.sos), 2):
            self.zi = np.zeros((len(self.sos), 2))

    def process(self, y):
        out, self.zi = sosfilt(self.sos, y, zi=self.zi)
        return out.astype(np.float32)


STEPS = {"flatten": Flatten, "bandpass": BandPass, "bandstop": BandStop, "audio_eq": AudioEQ, "lowpass": LowPass}


def catalog():
    """For the page: every step type with its domain, label, description, parameters and defaults."""
    return [{"type": k, "domain": c.domain, "label": c.label, "desc": c.desc, "params": c.params, "default": c.default}
            for k, c in STEPS.items()]


# ---------------- the sum: radio -> sound ----------------
class _Decimate:
    """Streaming low-pass FIR + keep every D-th sample, with the kept samples on a fixed grid of
    absolute indices (so consecutive calls join without a seam)."""

    def __init__(self, taps, D, i0=0):
        self.h, self.D = np.asarray(taps, np.float64)[::-1].copy(), int(D)
        self.hist = np.zeros(len(taps) - 1, np.float64)   # the last inputs, for the filter's memory
        self.i = int(i0)                                    # absolute index of the next input sample

    def process(self, v):
        L = len(self.h)
        buf = np.concatenate([self.hist, v])
        first = self.i                                     # absolute index of v[0] (= buf[L - 1])
        self.i += len(v)
        self.hist = buf[len(buf) - (L - 1):] if L > 1 else buf[:0]
        j0 = -(-first // self.D) * self.D                  # first kept absolute index >= first
        if j0 >= self.i:
            return j0 // self.D, np.zeros(0, np.float64)
        win = sliding_window_view(buf, L)[(j0 - first)::self.D]   # windows ending at each kept sample
        return j0 // self.D, win @ self.h


class SumToAudio:
    """The power of all the IQ, as sound, 16,000 samples a second, without folding.

    1. fine slices: the power of the IQ averaged over each 1/512,000 s (blocks cut on a fixed grid of
       absolute sample positions, so consecutive calls join without a seam);
    2. low-pass to the sound's range, in two stages: 512 k -> 64 k samples/s (a short filter that
       removes what would fold into 0-8 kHz), then 64 k -> 16 k (passes up to 7 kHz untouched, removes
       everything above 8 kHz, so nothing folds at all). Power changes faster than the sound can hold are removed here
       instead of folding back into it (two stations 40 kHz apart make the band's power pulse at
       40 kHz, which plain 1/16,000 s averages would turn into a tone at 8 kHz);
    3. the steady part (below 20 Hz) taken out."""

    R1 = AUDIO_FS * 32                         # fine slices: 512 kHz

    def __init__(self, fs, n0=0):
        self.fs = float(fs)
        self.r1 = float(min(self.R1, self.fs))
        self.n = int(n0)                       # absolute index of the next IQ sample expected
        self.k = int(np.ceil(n0 * self.r1 / self.fs - 1e-9))     # next fine slice (absolute)
        self.k += (-self.k) % 32               # start on the audio grid
        self.rest = np.zeros(0, np.float32)    # power of IQ samples from the last edge on, not used yet
        f2 = firwin(63, 24000, fs=self.R1)                                     # 512 k -> 64 k
        f3 = firwin(383, 7500, width=1000, window=("kaiser", 8.0), fs=self.R1 / 8)   # 64 k -> 16 k
        self.d2, self.d3 = _Decimate(f2, 8, self.k), _Decimate(f3, 4, self.k // 8)
        self.hp = butter(1, 20, btype="highpass", fs=AUDIO_FS, output="sos")
        self.zi = None

    def fine(self, x):
        """Average power per fine slice for IQ block x (which starts at absolute sample self.n).
        Slices have exact, fractional edges at k * fs / r1: a sample cut by an edge counts in
        proportion to how much of it falls inside (the IQ's power taken as constant over each
        sample). So every slice is exactly 1/512,000 s long, with no jitter from rounding the
        edges to whole samples. All whole-array operations: no Python loop over samples."""
        p = x.real * x.real + x.imag * x.imag                    # power of each IQ sample (float32)
        buf = np.concatenate([self.rest, p]) if len(self.rest) else p
        start = self.n - len(self.rest)                          # absolute index of buf[0]
        self.n += len(x)
        step = self.fs / self.r1                                 # IQ samples per slice (>= 1)
        k1 = int(np.ceil(self.n / step)) - 1                     # last edge with a sample under it
        if k1 <= self.k:
            self.rest = buf
            return np.zeros(0, np.float64)
        E = np.arange(self.k, k1 + 1) * step - start             # edges, in samples into buf
        fl = np.floor(E).astype(np.int64)
        fr = E - fl
        B = np.add.reduceat(buf[:fl[-1]], fl[:-1]).astype(np.float64)   # whole samples between edges
        pe = buf[fl].astype(np.float64)
        vals = (B + fr[1:] * pe[1:] - fr[:-1] * pe[:-1]) / step
        self.k, self.rest = k1, buf[fl[-1]:].copy()
        return vals

    def process(self, x):
        """Sound for IQ block x: (absolute index of its first audio sample, float32)."""
        v = self.fine(x)
        _, v = self.d2.process(v)
        k0, v = self.d3.process(v)
        if not len(v):
            return k0, np.zeros(0, np.float32)
        if self.zi is None:
            self.zi = sosfilt_zi(self.hp) * v[0]
        y, self.zi = sosfilt(self.hp, v, zi=self.zi)             # take out the steady part (< 20 Hz)
        return k0, y.astype(np.float32)


class Pipeline:
    """Radio steps in order, the sum, audio steps in order, then the level. configure() changes the
    steps while running: a step of the same type in the same place keeps its state (what it has
    learned, its filters' memory) and only takes the new settings."""

    def __init__(self, fs, n0=0, steps=(), fc=0.0):
        self.fs, self.fc = float(fs), float(fc)
        self.radio, self.audio = [], []         # [(type, on, instance, index in the step list)]
        self.sum = SumToAudio(fs, n0)
        self.gain = SmoothGain()
        self.iq = None                          # the last block of IQ after the radio steps
        self.configure(steps)

    def configure(self, steps):
        old = {"radio": list(self.radio), "audio": list(self.audio)}
        new = {"radio": [], "audio": []}
        for idx, s in enumerate(steps):
            cls = STEPS.get(s.get("type"))
            if cls is None:
                continue
            params = {**cls.default, **(s.get("params") or {})}
            pool = old[cls.domain]
            j = next((i for i, e in enumerate(pool) if e[0] == s["type"]), None)
            if j is not None:
                inst = pool.pop(j)[2]
                inst.set(**params)
            else:
                inst = cls(self.fs, self.fc, **params)
            new[cls.domain].append((s["type"], bool(s.get("on", True)), inst, idx))
        self.radio, self.audio = new["radio"], new["audio"]

    @property
    def radio_on(self):
        return any(e[1] for e in self.radio)

    def masks(self, off):
        """The radio steps' masks at frequency offsets `off`: ({index in the step list: amplitude gain},
        their product over the steps that are on: what reaches the sum)."""
        each, total = {}, np.ones(len(off))
        for _, on, s, idx in self.radio:
            c = s.curve(off) if hasattr(s, "curve") else None
            if c is None:
                continue
            each[idx] = c
            if on:
                total = total * c
        return each, total

    def process(self, x):
        for _, on, s, _ in self.radio:
            if on:
                x = s.process(x)
        self.iq = x
        k0, y = self.sum.process(x)
        if not len(y):
            return k0, y
        for _, on, s, _ in self.audio:
            if on:
                y = s.process(y)
        return k0, self.gain.process(y)

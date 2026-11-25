"""Playback audio processing for detections (what you hear and what is saved).

The detector itself always judges the raw demodulated audio; nothing here changes detection.

StreamingDenoiser  spectral gating with a minimum-statistics noise estimate, processed in
                   chunks of any size with exact overlap-add continuity between chunks.
SmoothGain         slow automatic level control: the gain glides across each chunk instead
                   of jumping between chunks, with a soft limiter for peaks.
fade()             raised-cosine fade in / out.
"""
import numpy as np
from scipy.signal import lfilter

FS = 16000


class StreamingDenoiser:
    """Per-bin gain = max(1 - strength * noise / power, floor), smoothed over time.

    Noise is tracked as the slowly-rising running minimum of each bin's smoothed power
    (minimum statistics), so speech pauses teach it the background without any labelling.
    sqrt-Hann analysis/synthesis windows at 75% overlap reconstruct exactly when gain = 1.
    Output lags input by N samples (32 ms at 16 kHz)."""

    def __init__(self, strength=1.0, floor_db=-18.0, n=512, hop=128):
        self.n, self.hop = n, hop
        self.strength, self.floor = float(strength), 10 ** (floor_db / 20)
        if n % hop:
            raise ValueError("n must be a multiple of hop")
        self.win = np.sqrt(np.hanning(n + 1)[:n]).astype(np.float32)
        self.inbuf = np.zeros(n, np.float32)            # last n input samples
        self.ola = np.zeros(n, np.float32)              # overlap-add accumulator
        self.pending = np.zeros(0, np.float32)          # input not yet consumed by a hop
        self.fifo = np.zeros(hop, np.float32)           # output queue; the prefill makes every call return len(x)
        self.ps = None                                  # smoothed power per bin
        self.nmin = None                                # noise estimate per bin
        self.gs = None                                  # smoothed gain per bin
        self.frames = 0

    def process(self, x):
        x = np.asarray(x, np.float32)
        if self.strength <= 0:
            return x
        n, hop = self.n, self.hop
        data = np.concatenate([self.pending, x])
        nh = len(data) // hop
        self.pending = data[nh * hop:]
        if nh == 0:
            out = np.zeros(0, np.float32)
        else:
            out = self._frames(data[:nh * hop], nh)
        self.fifo = np.concatenate([self.fifo, out])
        res, self.fifo = self.fifo[:len(x)], self.fifo[len(x):]
        return res

    def _frames(self, data, nh):
        """All complete hops of one call at once: one batched rfft/irfft, the sequential
        per-frame recursions as short loops / filters over 257-bin vectors, vectorised OLA."""
        n, hop = self.n, self.hop
        buf = np.concatenate([self.inbuf[hop:], data])            # frame i = buf[i*hop : i*hop+n]
        frames = np.lib.stride_tricks.sliding_window_view(buf, n)[::hop]
        self.inbuf = buf[-n:].copy()
        X = np.fft.rfft(frames * self.win, axis=1)
        P = (X.real ** 2 + X.imag ** 2).astype(np.float32)
        if self.ps is None:
            self.ps, self.nmin, self.gs = P[0].copy(), P[0].copy(), np.ones_like(P[0])
        f0 = self.frames
        # noise tracker (conditional recursion, inherently sequential): see the class docstring
        N = np.empty_like(P)
        nmin = self.nmin
        for i in range(nh):
            f = f0 + i
            Pi = P[i]
            if f < 20:                                             # seed: plain average over the first ~0.16 s
                nmin = Pi.copy() if f <= 4 else nmin + (Pi - nmin) / (f + 1 - 4)
            else:                                                  # average over noise-like frames, else creep up
                nmin = np.where(Pi < 2.5 * nmin, 0.97 * nmin + 0.03 * Pi, nmin * 1.002)
            N[i] = nmin
        self.nmin = nmin
        # gain from power smoothed over neighbouring bins (3-tap) and frames (one-pole IIR)
        Pf = 0.5 * P.astype(np.float64)
        Pf[:, 1:] += 0.25 * P[:, :-1]
        Pf[:, :-1] += 0.25 * P[:, 1:]
        ps, _ = lfilter([0.35], [1.0, -0.65], Pf, axis=0, zi=(0.65 * np.asarray(self.ps, np.float64))[None, :])
        self.ps = ps[-1]
        G = np.maximum(1 - self.strength * 1.65 * N / (ps + 1e-12), self.floor)
        if f0 < 20:                                                # no estimate yet (~0.16 s): pass through
            G[:max(0, min(nh, 20 - f0))] = 1.0
        self.gs = G[-1]
        self.frames += nh
        Y = np.fft.irfft(X * G, n, axis=1).astype(np.float32) * self.win * 0.5
        # overlap-add: frame i lands on hop blocks i .. i+r-1; block i is complete after frame i
        r = n // hop
        acc = np.zeros((nh + r, hop), np.float32)
        acc[:r] = self.ola.reshape(r, hop)
        Y = Y.reshape(nh, r, hop)
        for k in range(r - 1, -1, -1):                             # oldest contribution first, as streamed
            acc[k:k + nh] += Y[:, k]
        self.ola = acc[nh:].reshape(-1).copy()
        return acc[:nh].reshape(-1)


class SmoothGain:
    """Automatic level control that glides (time constant ~tau) and soft-limits peaks."""

    def __init__(self, target_rms=0.18, tau_s=1.2, max_gain=20000.0):   # raw AM audio is often ~1e-3 FS
        self.target, self.tau, self.max_gain = target_rms, tau_s, max_gain
        self.g = None

    def process(self, x):
        x = np.asarray(x, np.float32)
        if not len(x):
            return x
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        want = min(self.max_gain, self.target / rms)
        if self.g is None:
            self.g = want
        a = 1 - np.exp(-len(x) / FS / self.tau)
        g_new = self.g + a * (want - self.g)
        ramp = np.linspace(self.g, g_new, len(x), dtype=np.float32)
        self.g = g_new
        return np.tanh(1.3 * x * ramp) / np.tanh(1.3)              # soft limiter, ~linear below 0.5


def fade(x, fs=FS, fade_in_s=0.0, fade_out_s=0.0):
    x = np.array(x, np.float32, copy=True)
    for secs, start in ((fade_in_s, True), (fade_out_s, False)):
        n = min(len(x), int(secs * fs))
        if n > 1:
            r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
            if start:
                x[:n] *= r
            else:
                x[-n:] *= r[::-1]
    return x


def render_clip(audio, cfg_audio):
    """Whole-clip version of the playback chain (for clips rebuilt from the IQ ring)."""
    y = np.asarray(audio, np.float32)
    if cfg_audio.get("denoise", True):
        y = StreamingDenoiser(cfg_audio.get("denoise_strength", 1.0), cfg_audio.get("denoise_floor_db", -18)).process(y)
    y = SmoothGain().process(y)
    return fade(y, FS, cfg_audio.get("fade_s", 0.15), cfg_audio.get("fade_s", 0.15))

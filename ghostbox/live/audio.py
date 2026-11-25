"""Playback audio processing for detections (what you hear and what is saved).

The detector itself always judges the raw demodulated audio; nothing here changes detection.

StreamingDenoiser  spectral gating with a minimum-statistics noise estimate, processed in
                   chunks of any size with exact overlap-add continuity between chunks.
SmoothGain         slow automatic level control: the gain glides across each chunk instead
                   of jumping between chunks, with a soft limiter for peaks.
fade()             raised-cosine fade in / out.
"""
import numpy as np

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
        data = np.concatenate([self.pending, x])
        nh = len(data) // self.hop
        out = np.empty(nh * self.hop, np.float32)
        for i in range(nh):
            hopd = data[i * self.hop:(i + 1) * self.hop]
            self.inbuf = np.concatenate([self.inbuf[self.hop:], hopd])
            X = np.fft.rfft(self.inbuf * self.win)
            P = (X.real ** 2 + X.imag ** 2).astype(np.float32)
            if self.ps is None:
                self.ps, self.nmin, self.gs = P.copy(), P.copy(), np.ones_like(P)
            # noise tracker: average each bin only over frames where it looks like noise (< 2.5x the
            # estimate); otherwise let the estimate creep up slowly so a rising background is followed
            if self.frames < 20:                                   # seed: plain average over the first ~0.16 s
                k = self.frames + 1
                self.nmin = P.copy() if self.frames <= 4 else self.nmin + (P - self.nmin) / (k - 4)
            else:
                quiet = P < 2.5 * self.nmin
                self.nmin = np.where(quiet, 0.97 * self.nmin + 0.03 * P, self.nmin * 1.002)
            noise = (1.0 if self.frames < 20 else 1.65) * self.nmin  # the truncated average sits ~0.6x the mean
            # decide the gain from power smoothed over neighbouring frames and bins: a single frame's
            # noise power fluctuates too much (that is what leaves "musical noise" behind)
            Pf = np.convolve(P, (0.25, 0.5, 0.25), mode="same")
            self.ps = 0.65 * self.ps + 0.35 * Pf
            self.gs = np.maximum(1 - self.strength * noise / (self.ps + 1e-12), self.floor)
            if self.frames < 20:                                   # no estimate yet (~0.16 s): pass through
                self.gs[:] = 1.0
            self.frames += 1
            y = np.fft.irfft(X * self.gs, self.n).astype(np.float32) * self.win * 0.5
            self.ola += y
            out[i * self.hop:(i + 1) * self.hop] = self.ola[:self.hop]
            self.ola = np.concatenate([self.ola[self.hop:], np.zeros(self.hop, np.float32)])
        self.pending = data[nh * self.hop:]
        self.fifo = np.concatenate([self.fifo, out])
        res, self.fifo = self.fifo[:len(x)], self.fifo[len(x):]
        return res


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

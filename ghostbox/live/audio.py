"""Level and fades for what you hear. Nothing here filters the sound.

SmoothGain   slow automatic level control: the gain glides across each chunk instead of jumping
             between chunks, with a soft limiter for peaks (faint ranges would be inaudible).
fade()       raised-cosine fade in / out.
"""
import numpy as np

FS = 16000


class SmoothGain:
    """Automatic level control that glides (time constant ~tau) and soft-limits peaks."""

    def __init__(self, target_rms=0.18, tau_s=1.2, max_gain=20000.0):   # a raw envelope is often ~1e-3 FS
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

"""Learned band profile: what each frequency normally looks like, learned as it goes.

Per analysis bin it keeps an exponentially weighted mean and variance of the combined voice
evidence (time constant: hours), so no IQ is stored. It is saved per band (centre, span, bin
width, analysis range) every few minutes and on retune, and loaded again when you come back,
so it keeps learning across sessions. The "Unusual for this band" score is how many standard
deviations the current evidence sits above that frequency's normal.
"""
import time
from pathlib import Path

import numpy as np


class BandProfile:
    def __init__(self, path, n_bins):
        self.path = Path(path)
        self.mean = np.zeros(n_bins, np.float32)
        self.var = np.full(n_bins, 1e-3, np.float32)
        self.learned_s = 0.0
        self.saved_at = time.time()
        if self.path.exists():
            try:
                d = np.load(self.path)
                if d["mean"].shape == (n_bins,):
                    self.mean, self.var = d["mean"].astype(np.float32), d["var"].astype(np.float32)
                    self.learned_s = float(d["learned_s"])
            except Exception:
                pass

    @staticmethod
    def key(fc, fs, bin_hz, f_lo, f_hi):
        return f"{fc:.0f}_{fs:.0f}_{bin_hz:g}_{f_lo:.0f}_{f_hi:.0f}.npz"

    def score(self, x, full_z):
        """[frames, bins] evidence -> unusualness in [0, 1] against the learned profile."""
        z = (x - self.mean) / np.sqrt(self.var + 1e-4)
        return np.clip(z / full_z, 0, 1).astype(np.float32)

    def update(self, x, frame_rate, tau_h):
        """Fold in [frames, bins] evidence (exponential forgetting with time constant tau_h)."""
        a = 1.0 / max(1.0, tau_h * 3600 * frame_rate)
        for row in x:                                   # cheap: a few frames per second
            d = row - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
        self.learned_s += len(x) / frame_rate

    def maybe_save(self, every_s=300, force=False):
        if not force and time.time() - self.saved_at < every_s:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez(tmp, mean=self.mean, var=self.var, learned_s=self.learned_s)
            tmp.replace(self.path)
        except Exception:
            pass
        self.saved_at = time.time()

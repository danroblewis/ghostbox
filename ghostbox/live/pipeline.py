"""The signal pipeline: IQ in, a chain of IQ steps, then the one final step that turns IQ into sound.

    raw IQ ──► step 1 ──► step 2 ──► … ──► SUM ──► audio
               (IQ in, IQ out: filters and transforms on the whole band)

There are no steps yet: they are added one at a time (STEPS, below). Every step takes a block of
complex IQ at the receiver's rate and returns a block of the same length, so the chain can always
end in the same final step.

The final step, SumToAudio, is the simplest way there is to make sound out of IQ: at every audio
sample (16,000 per second) it adds up the power of all the IQ that arrived in that sliver of time.
That is the total power of the whole band, moment by moment (by Parseval's theorem, the same as
adding up every column of a spectrum taken over that sliver). Whatever the steps before it leave
in the band is what is heard.

Two more things happen to that sum so it can be played, and nothing else:
  - the steady part is taken out: the band's average power is a large constant, which is silent
    but would use up the speaker's whole range; a one-pole high-pass at 20 Hz (below hearing)
    removes it;
  - the level is evened out (audio.SmoothGain): the sum's size depends on the gain and the band,
    so it is scaled slowly toward a comfortable loudness.
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .audio import SmoothGain

AUDIO_FS = 16000

# IQ steps: type -> class with process(x) -> x (same length). None yet.
STEPS = {}


class SumToAudio:
    """Power of all the IQ in each 1/16000 s, as audio. Stateful: blocks are cut on a fixed grid of
    absolute sample positions, so audio from consecutive calls joins without a seam, and a call may
    end mid-block (the rest waits for the next call)."""

    def __init__(self, fs, n0=0):
        self.fs = float(fs)
        self.n = int(n0)                       # absolute index of the next IQ sample expected
        self.k = int(np.ceil(n0 * AUDIO_FS / self.fs - 1e-9))   # next audio sample (absolute)
        self.rest = np.zeros(0, np.float64)    # power of IQ samples not yet in a finished block
        self.hp = butter(1, 20, btype="highpass", fs=AUDIO_FS, output="sos")
        self.zi = None
        self.gain = SmoothGain()

    def edge(self, k):
        return int(round(k * self.fs / AUDIO_FS))

    def raw(self, x):
        """Power sum per audio sample (divided by the samples in it) for IQ block x (which starts at absolute sample self.n).
        Returns (absolute index of the first audio sample, values)."""
        start = self.n - len(self.rest)                          # absolute index of buf[0]
        buf = np.concatenate([self.rest, (x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2)])
        self.n += len(x)
        k1 = self.k
        while self.edge(k1 + 1) <= self.n:
            k1 += 1
        if k1 == self.k:
            self.rest = buf
            return self.k, np.zeros(0, np.float64)
        e = np.array([self.edge(k) for k in range(self.k, k1 + 1)]) - start
        cs = np.concatenate([[0.0], np.cumsum(buf)])
        # sum of power over each block, per IQ sample: at rates that aren't a multiple of 16 kHz the
        # blocks alternate in length by one sample, and a plain sum would wobble with that pattern
        vals = (cs[e[1:]] - cs[e[:-1]]) / np.diff(e)
        k0, self.k, self.rest = self.k, k1, buf[e[-1]:]
        return k0, vals

    def process(self, x):
        """Audio for IQ block x: (stream time of its first sample in samples/AUDIO_FS units, float32)."""
        k0, v = self.raw(x)
        if not len(v):
            return k0, np.zeros(0, np.float32)
        if self.zi is None:
            self.zi = sosfilt_zi(self.hp) * v[0]
        y, self.zi = sosfilt(self.hp, v, zi=self.zi)             # take out the steady part (< 20 Hz)
        return k0, self.gain.process(y.astype(np.float32))       # even out the level


class Pipeline:
    """The IQ steps in order (none yet), then SumToAudio."""

    def __init__(self, fs, n0=0, steps=()):
        self.steps = [STEPS[s["type"]](fs, **(s.get("params") or {})) for s in steps
                      if s.get("on", True) and s.get("type") in STEPS]
        self.out = SumToAudio(fs, n0)

    def process(self, x):
        for s in self.steps:
            x = s.process(x)
        return self.out.process(x)

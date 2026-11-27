"""Voice shaper for the composed mix: makes what is already voice-like in the mix stand out, the way
a speech codec's postfilter does. Everything is measured from the mix itself, frame by frame (20 ms),
causally; no target pitch, formant or rhythm is put in.

  pitch    long-term postfilter: where the mix repeats at a pitch period (70-400 Hz, found on the
           LPC residual so a narrow band's ringing is not mistaken for pitch), that period is
           reinforced, so harmonics already there ring out (0..1).
  formant  short-term postfilter A(z/0.55) / A(z/g): the mix's own spectral peaks (its LPC envelope)
           are sharpened, like formants; tilt compensated (0..1).
  duck_db  between voiced stretches (low periodicity), the mix is turned down by this much, with a
           150 ms hang so consonants at the edges of a voiced stretch stay.
"""
import numpy as np
from scipy.signal import butter, lfilter, sosfilt

FS = 16000
FRAME = 320
LAG_MIN, LAG_MAX = FS // 400, FS // 70
WIN = 640                     # periodicity measured over the last 40 ms
HIST = WIN + LAG_MAX + 32
_LP = butter(4, 1000, "lowpass", fs=FS, output="sos")


def _lpc(x, order):
    r = np.correlate(x, x, "full")[len(x) - 1: len(x) + order]
    if r[0] <= 1e-10:
        return np.r_[1.0, np.zeros(order)], 0.0
    r = r * np.exp(-0.5 * (2 * np.pi * 60 / FS * np.arange(order + 1)) ** 2)    # lag window
    r[0] *= 1.0001
    a, e = np.zeros(order + 1), r[0]
    a[0] = 1.0
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / e
        a[1:i + 1] = a[1:i + 1] + k * np.r_[a[i - 1:0:-1], 1.0]
        e *= 1 - k * k
        if e <= 0:
            break
    return a, r[1] / r[0]


def analyse(buf, order=12):
    """The last 20 ms of buf (HIST + frame samples): LPC envelope a, first autocorrelation r1, and
    periodicity v (0..1) at pitch period T samples, measured on the whitened signal below 1 kHz (where
    the harmonics are strongest; whitening keeps a narrow band's ringing from passing as pitch)."""
    a, r1 = _lpc(buf[-480:] * np.hanning(480), order)
    e = sosfilt(_LP, lfilter(a, [1.0], buf))
    S0 = len(buf) - WIN
    cur = e[S0:]
    ee = float(cur @ cur)
    if ee <= 1e-12 or S0 < LAG_MAX:
        return a, r1, 0.0, 0
    lags = np.arange(LAG_MIN, LAG_MAX + 1)
    past = e[S0 - lags[:, None] + np.arange(WIN)[None, :]]
    r = (past @ cur) / (np.sqrt(ee * np.einsum("ij,ij->i", past, past)) + 1e-12)
    j = int(np.argmax(r))
    for jj in range(j):                         # the shortest lag nearly as good: no octave-down errors
        q = lags[j] % lags[jj]
        if r[jj] > 0.85 * r[j] and min(q, lags[jj] - q) < 3:
            j = jj
            break
    return a, r1, float(max(0.0, r[j])), int(lags[j])


class Voicing:
    """Streaming periodicity only (for judging a channel): process(y) -> v per 20 ms frame."""

    def __init__(self):
        self.hist = np.zeros(HIST, np.float32)

    def process(self, x):
        out = []
        for i in range(0, len(x), FRAME):
            buf = np.concatenate([self.hist, x[i:i + FRAME]]).astype(np.float64)
            self.hist = buf[-HIST:].astype(np.float32)
            if len(x[i:i + FRAME]) >= 80:
                out.append(analyse(buf)[2])
        return out


def voiced_score(v):
    """Frame periodicity -> how voiced, 0 (noise stays below ~0.42) .. 1 (clearly pitched)."""
    return float(np.mean(np.clip((np.asarray(v) - 0.42) / 0.25, 0, 1))) if len(v) else 0.0


class VoiceShaper:
    def __init__(self, pitch=0.6, formant=0.6, duck_db=12.0, order=12):
        self.set(pitch, formant, duck_db)
        self.order = order
        self.hist = np.zeros(HIST, np.float32)
        self.zi = np.zeros(order)
        self.tilt_prev = 0.0
        self.g_prev, self.hang, self.lvl = self.duck, 0, None
        self.voicing = []                       # per frame, for inspection

    def set(self, pitch, formant, duck_db):
        self.pitch, self.formant, self.duck = float(pitch), float(formant), 10 ** (-float(duck_db) / 20)

    def process(self, x):
        x = np.asarray(x, np.float32)
        out = np.empty_like(x)
        for i in range(0, len(x), FRAME):
            out[i:i + FRAME] = self._frame(x[i:i + FRAME])
        return out

    def _frame(self, x):
        L = len(x)
        buf = np.concatenate([self.hist, x]).astype(np.float64)
        self.hist = buf[-HIST:].astype(np.float32)
        H = len(buf) - L
        a, r1, v, T = analyse(buf, self.order)
        self.voicing.append(v)
        y = buf[H:].copy()
        # long-term: reinforce the period (on the input, FIR: always stable)
        if T and v > 0.4 and self.pitch > 0:
            g = self.pitch * min(1.0, (v - 0.4) / 0.3)
            y = (y + g * buf[H - T: H - T + L]) / (1 + g)
        # short-term: sharpen the mix's own envelope peaks
        if self.formant > 0:
            gd = 0.55 + 0.35 * self.formant
            num_ = a * 0.55 ** np.arange(self.order + 1)
            den_ = a * gd ** np.arange(self.order + 1)
            y, self.zi = lfilter(num_, den_, y, zi=self.zi)
            mu = 0.5 * self.formant * max(0.0, r1)
            y = y - mu * np.r_[self.tilt_prev, y[:-1]] if L else y
            self.tilt_prev = float(y[-1]) if L else self.tilt_prev
            ein, eout = float(np.mean(buf[H:] ** 2)), float(np.mean(y ** 2)) + 1e-12
            y = y * np.sqrt(ein / eout) if ein > 0 else y
        # ducking between voiced stretches, with a hang for consonants
        if v > 0.45:
            self.hang = int(0.15 * FS / FRAME) + 1
        target = 1.0 if self.hang > 0 else self.duck + (1 - self.duck) * min(1.0, max(0.0, (v - 0.35) / 0.1))
        self.hang = max(0, self.hang - 1)
        g = np.linspace(self.g_prev, target, L, endpoint=False) if target < self.g_prev else \
            np.linspace(self.g_prev, target, min(L, 48), endpoint=False).tolist() + [target] * (L - min(L, 48))
        self.g_prev = target
        return (y * np.asarray(g)).astype(np.float32)


def shape(y, params):
    """The shaper with compose config params (shape_pitch, shape_formant, shape_duck_db) over a whole clip."""
    return VoiceShaper(params.get("shape_pitch", 0.6), params.get("shape_formant", 0.6),
                       params.get("shape_duck_db", 12.0)).process(y)

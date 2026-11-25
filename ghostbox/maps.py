"""Whole-band AM-voice evidence maps, computed directly on wideband IQ.

No channelizer and no demodulation: every candidate carrier frequency is tested
at once with vectorised operations over a fine (time x frequency) power grid.

For each candidate carrier bin k, three envelopes are formed from box sums along
frequency (bin offsets relative to k), log-compressed and band-passed to the
syllabic band (2-10 Hz):

  u  upper sideband   k+inner .. k+outer        (~250 Hz .. 3.4 kHz)
  l  lower sideband   k-outer .. k-inner
  r  ring             k +- (ring_lo .. ring_hi) (~4.5 .. 8 kHz, both sides;
                    ring_hz=None disables it, e.g. in the AM broadcast band where
                    there is no FM to reject and station audio reaches 10+ kHz)

  sym   = corr(u, l)        AM double sideband: mirror-image sidebands -> ~+1
  mod   = rms(u,l) / noise  how strongly the sidebands pulse at syllable rate
  ring  = corr(u+l, r)      a 200 kHz FM station pulses everywhere at once -> ~+1;
                            a narrow AM voice has only noise in its ring -> ~0
  raw   = sym+ * log(mod)+ * (1 - ring+)
  score = raw minus each bin's own running mean of raw over baseline_s
          (novelty: always-on emitters such as FM station centres cancel out;
          a voice that appears, hops in or slides through stands out)

All correlations use a short sliding window (stat_win_s) so hopping voices with
a few hundred ms dwell can still be caught.
"""
from dataclasses import dataclass

import numpy as np
from scipy import fft as sp_fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt


@dataclass
class Maps:
    t: np.ndarray       # frame times (s)
    f: np.ndarray       # bin offsets from center (Hz)
    P: np.ndarray       # power spectrogram
    mod: np.ndarray     # sideband syllabic modulation, relative to noise
    sym: np.ndarray     # upper/lower sideband envelope correlation
    ring: np.ndarray    # sideband/ring envelope correlation (FM-likeness)
    raw: np.ndarray     # AM-voice evidence before novelty normalisation
    score: np.ndarray   # novelty-normalised AM-voice evidence


def power_frames(x, fs, nfft=19200, frame_rate=25, chunk_s=1.0):
    """Welch-averaged spectrum per frame: frame = fs/frame_rate samples, split into nfft segments."""
    hop = int(fs // frame_rate)
    nseg = hop // nfft
    win = np.hanning(nfft).astype(np.float32)
    win /= np.sqrt(np.sum(win ** 2))
    nframes = len(x) // hop
    P = np.empty((nframes, nfft), dtype=np.float32)
    per_chunk = max(1, int(chunk_s * frame_rate))
    for i in range(0, nframes, per_chunk):
        j = min(nframes, i + per_chunk)
        seg = x[i * hop: j * hop].reshape(j - i, hop)[:, : nseg * nfft].reshape(j - i, nseg, nfft)
        S = sp_fft.fft(seg * win, axis=-1, workers=-1)
        P[i:j] = np.fft.fftshift(np.mean(np.abs(S) ** 2, axis=1), axes=-1)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)).astype(np.float32)
    t = ((np.arange(nframes) + 0.5) / frame_rate).astype(np.float32)
    return t, f, P


class OffsetSummer:
    """offset_sum(a, b)[:, k] = sum of P[:, k+a .. k+b] (zero outside the band)."""

    def __init__(self, P, max_off):
        T, K = P.shape
        self.m = max_off + 1
        pad = np.zeros((T, self.m), dtype=np.float64)
        self.cs = np.cumsum(np.concatenate([pad, P.astype(np.float64), pad], axis=1), axis=1)
        self.k = np.arange(K) + self.m

    def __call__(self, a, b):
        return (self.cs[:, self.k + b] - self.cs[:, self.k + a - 1]).astype(np.float32)


def compute_maps(x, fs, bin_hz=125, frame_rate=25, dc_guard_hz=2000, **kw):
    nfft = int(round(fs / bin_hz))
    t, f, P = power_frames(x, fs, nfft, frame_rate)
    dc = np.abs(f) <= dc_guard_hz
    P[:, dc] = np.median(P, axis=1, keepdims=True)  # drop DC/LO leakage spike
    return maps_from_power(t, f, P, frame_rate=frame_rate, **kw)


def maps_from_power(t, f, P, frame_rate=25, inner_hz=250, outer_hz=3400,
                    ring_hz=(4500, 8000), band=(2.0, 10.0), stat_win_s=0.5, baseline_s=15.0):
    """All maps from a precomputed power spectrogram (lets callers build P in chunks or crop it)."""
    df = float(f[1] - f[0])
    inner, outer = round(inner_hz / df), round(outer_hz / df)
    S = OffsetSummer(P, round(ring_hz[1] / df) if ring_hz else outer)
    U, L = S(inner, outer), S(-outer, -inner)

    sos = butter(4, band, btype="bandpass", fs=frame_rate, output="sos")
    bp = lambda a: sosfiltfilt(sos, np.log(a + 1e-12), axis=0).astype(np.float32)
    u, l = bp(U), bp(L)

    n = max(3, int(stat_win_s * frame_rate))
    avg = lambda a: uniform_filter1d(a, size=n, axis=0, mode="nearest")
    corr = lambda a, b: avg(a * b) / np.sqrt(avg(a * a) * avg(b * b) + 1e-12)
    sym = corr(u, l)
    if ring_hz:  # FM rejection; pointless (and harmful to wideband AM) where there is no FM
        r0, r1 = round(ring_hz[0] / df), round(ring_hz[1] / df)
        ring = corr(u + l, bp(S(r0, r1) + S(-r1, -r0)))
    else:
        ring = np.zeros_like(u)
    rms = np.sqrt(0.5 * (avg(u * u) + avg(l * l)))
    mod = rms / np.median(rms)
    raw = np.clip(sym, 0, 1) * np.clip(np.log(mod), 0, None) * (1 - np.clip(ring, 0, 1))
    nb = min(len(t), int(baseline_s * frame_rate))
    baseline = uniform_filter1d(raw, size=nb, axis=0, mode="nearest")
    score = np.clip(raw - baseline, 0, None)
    return Maps(t, f, P, mod.astype(np.float32), sym.astype(np.float32), ring.astype(np.float32),
                raw.astype(np.float32), score.astype(np.float32))

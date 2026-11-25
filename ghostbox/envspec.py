"""Audio-rate AM detection for every frequency at once, via spectral correlation.

Split each frame into M segments of N samples (bin spacing df = fs/N). Within
one segment, the sum over a frequency window of bin-pair products

    a_j    = X_m(j+d) conj(X_m(j))
    E_d,m  = sum_{j in window} a_j

is the spectrum of that window's amplitude envelope |x|^2 at alpha = d*df.
In other words: "AM-demodulate the band and take its audio spectrum", done for
every window center at once with cumulative sums along frequency.

|E|^2 = sum_j |a_j|^2 + cross terms. Independent noise only produces the self
terms, so the statistic keeps the cross terms and expresses them in noise
standard deviations:

    num = sum_{d,m} ( |sum_j a_j|^2 - sum_j |a_j|^2 )
    z   = num / sqrt( sum_{d,m} (sum_j |a_j|^2)^2 )

  noise -> z ~ N(0, 1)
  AM    -> z >> 0  (the (lower sideband, carrier) and (carrier, upper sideband)
                    pairs carry the same envelope phase)
  FM    -> z << 0  when the window holds the whole signal: constant envelope,
                    so the sum cancels and only the subtracted self terms remain

Hann-windowed bins are correlated with neighbours up to 2 bins away, which would
bias the cross terms. So the sum runs over one residue class j = r (mod 3) at a
time, with d a multiple of 3: every bin involved is then 3+ bins from the others.
A carrier at c and its sidebands c +- d always share a class, so nothing is lost.

Scale matters:
  narrow (~3.5 kHz) : a ghost voice on its own carrier.
  wide (~260 kHz):   a whole FM broadcast station; a voice amplitude-modulating
                     the station pushes its strongly negative z upward.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

STEP = 3  # Hann bins this far apart are uncorrelated for white noise


@dataclass
class EnvMap:
    scale_hz: float
    t: np.ndarray
    f: np.ndarray      # window centers (Hz offset)
    z: np.ndarray      # AM cross-term statistic in noise sigmas, [frames, centers]
    score: np.ndarray  # novelty: (z - running mean) / running spread per center, clipped at 0


def _class_box(cs_list, lo, hi):
    """Sum over j in [lo, hi] of a quantity stored per residue class.

    cs_list[r] is a zero-prefixed cumsum along the last axis of the values at j = r, r+3, r+6, ...
    Returns the sum for every class separately, stacked on a new leading axis.
    """
    out = []
    for r, cs in enumerate(cs_list):
        n = cs.shape[-1] - 1
        qa = np.clip(-((r - lo) // STEP), 0, n)          # ceil((lo - r)/3)
        qb = np.clip((hi - r) // STEP + 1, 0, n)          # floor((hi - r)/3) + 1
        qb = np.maximum(qa, qb)
        out.append(cs[..., qb] - cs[..., qa])
    return np.stack(out)


def _class_cumsums(v, dtype):
    z = np.zeros(v.shape[:-1] + (1,), dtype)
    return [np.concatenate([z, np.cumsum(v[..., r::STEP], axis=-1, dtype=dtype)], axis=-1)
            for r in range(STEP)]


def envelope_maps(x, fs, scales_hz=(3500, 260000), nfft=19200, frame_rate=25,
                  alpha_hz=(300, 3400), dc_guard_hz=2000, chunk_frames=25, baseline_s=15.0,
                  stride_div=4, smooth_s=0.2):
    hop = int(fs // frame_rate)
    M = hop // nfft
    df = fs / nfft
    ds = [d for d in range(STEP, round(alpha_hz[1] / df) + 1, STEP) if d * df >= alpha_hz[0]]
    nframes = len(x) // hop
    win = np.hanning(nfft).astype(np.float32)
    f_all = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)).astype(np.float32)
    dc = np.abs(f_all) <= dc_guard_hz

    specs = []
    for s in scales_hz:
        w = max(1, round(s / 2 / df))
        stride = max(1, w // stride_div)
        kc = np.arange(0, nfft, stride)
        specs.append((s, w, kc, np.empty((nframes, len(kc)), np.float32)))

    for i in range(0, nframes, chunk_frames):
        j = min(nframes, i + chunk_frames)
        seg = x[i * hop: j * hop].reshape(j - i, hop)[:, : M * nfft].reshape(j - i, M, nfft)
        X = np.fft.fftshift(np.fft.fft(seg * win, axis=-1), axes=-1).astype(np.complex64)
        X[:, :, dc] = 0
        num = [np.zeros((j - i, len(kc))) for _, _, kc, _ in specs]
        var = [np.zeros((j - i, len(kc))) for _, _, kc, _ in specs]
        for d in ds:
            a = X[:, :, d:] * np.conj(X[:, :, :-d])            # pair (j, j+d) stored at j
            csA = _class_cumsums(a, np.complex128)
            csS = _class_cumsums(np.abs(a) ** 2, np.float64)
            for (s, w, kc, _), nu, va in zip(specs, num, var):
                lo, hi = kc - w, kc + w - d                     # pairs fully inside [k-w, k+w]
                E = _class_box(csA, lo, hi)                     # [class, frame, seg, center]
                S = _class_box(csS, lo, hi)
                nu += (np.abs(E) ** 2 - S).sum(axis=(0, 2))
                va += (S ** 2).sum(axis=(0, 2))
        for (s, w, kc, out), nu, va in zip(specs, num, var):
            out[i:j] = nu / np.sqrt(va + 1e-60)

    t = ((np.arange(nframes) + 0.5) / frame_rate).astype(np.float32)
    nb = min(nframes, int(baseline_s * frame_rate))
    ns = max(1, int(smooth_s * frame_rate))
    maps = []
    for s, w, kc, Z in specs:
        zs = uniform_filter1d(Z, size=ns, axis=0, mode="nearest")
        base = uniform_filter1d(zs, size=nb, axis=0, mode="nearest")
        spread = np.sqrt(uniform_filter1d((zs - base) ** 2, size=nb, axis=0, mode="nearest"))
        # each center's own units: an FM station whose z swings with its programme audio
        # needs a proportionally larger jump than a quiet stretch of band
        score = np.clip((zs - base) / (spread + 1.0), 0, None).astype(np.float32)
        maps.append(EnvMap(s, t, f_all[kc], Z, score))
    return maps

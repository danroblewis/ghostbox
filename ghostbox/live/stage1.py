"""Stage 1: whole-band checks, every frequency bin at once, from the power spectrogram.

Each check returns a [frames, bins] score in [0, 1] (1 = looks like an AM voice
centred on this bin). Only the requested (settled) frame slice is returned.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..maps import OffsetSummer

STAGE1 = ("twinning", "syllabic", "mirror", "carrier", "ring", "broadband")


def geo_mean(scores, weights, eps=0.02):
    """Weighted geometric mean of {id: array}; weight 0 -> ignored. Returns None if nothing enabled."""
    num, den = None, 0.0
    for k, s in scores.items():
        w = weights.get(k, 0.0)
        if w <= 0:
            continue
        term = w * np.log(np.maximum(s, eps))
        num = term if num is None else num + term
        den += w
    return None if num is None else np.exp(num / den)


def run(P, df, frame_rate, cfg, sl, force=()):
    """force: check ids to compute even if disabled (for waterfall views); they don't count
    toward the combined score unless enabled."""
    an, ch = cfg["analysis"], cfg["checks"]
    on = {k: ch[k]["enabled"] or k in force for k in STAGE1}
    inner = max(1, round(an["sideband_lo_hz"] / df))
    outer = max(inner + 2, round(an["sideband_hi_hz"] / df))
    max_off = outer
    for k in ("ring", "broadband"):
        if on[k]:
            max_off = max(max_off, round(ch[k]["hi_hz"] / df))
    S = OffsetSummer(P, max_off)
    U, L = S(inner, outer), S(-outer, -inner)

    syl = ch["syllabic"]
    lo, hi = syl["band_lo"], min(syl["band_hi"], 0.45 * frame_rate)
    sos = butter(4, (lo, hi), btype="bandpass", fs=frame_rate, output="sos")
    bp = lambda a: sosfiltfilt(sos, np.log(a + 1e-12), axis=0).astype(np.float32)
    n = max(3, int(an["stat_win_s"] * frame_rate))
    avg = lambda a: uniform_filter1d(a, size=n, axis=0, mode="nearest")
    corr = lambda a, b: avg(a * b) / np.sqrt(avg(a * a) * avg(b * b) + 1e-12)

    u, l = bp(U), bp(L)
    out = {}
    if on["twinning"]:
        out["twinning"] = np.clip(corr(u, l), 0, 1)[sl]
    if on["syllabic"]:
        rms = np.sqrt(0.5 * (avg(u * u) + avg(l * l)))
        mod = rms / (np.median(rms) + 1e-12)
        out["syllabic"] = np.clip(np.log(np.maximum(mod, 1e-6)) / np.log(syl["full_scale"]), 0, 1)[sl]
    if on["mirror"]:
        out["mirror"] = _fine_mirror(P, inner, outer, avg)[sl]
    if on["carrier"]:
        pk = maximum_filter1d(P, 3, axis=1)
        dens = (U + L) / (2 * (outer - inner + 1))
        cdb = 10 * np.log10((pk + 1e-12) / (dens + 1e-12))
        out["carrier"] = np.clip(avg(cdb) / ch["carrier"]["full_db"], 0, 1)[sl]
    for k in ("ring", "broadband"):
        if on[k]:
            r0, r1 = round(ch[k]["lo_hz"] / df), round(ch[k]["hi_hz"] / df)
            r = bp(S(r0, r1) + S(-r1, -r0))
            out[k] = 1 - np.clip(corr(u + l, r), 0, 1)[sl]

    weights = {k: ch[k]["weight"] if ch[k]["enabled"] else 0.0 for k in out}
    combined = geo_mean(out, weights)
    if combined is None:
        combined = np.zeros((sl.stop - sl.start, P.shape[1]), np.float32)
    return out, combined.astype(np.float32)


def _fine_mirror(P, inner, outer, avg):
    """Per frame, correlation across offsets d of A[k+d] and A[k-d], where A is log power with
    each bin's own time-mean removed (receiver ripple cancels, changing voice spectra remain)."""
    A = np.log(P + 1e-12)
    A -= A.mean(axis=0, keepdims=True)
    T, K = A.shape
    pad = np.zeros((T, outer), np.float32)
    Ap = np.concatenate([pad, A.astype(np.float32), pad], axis=1)
    sx = np.zeros((T, K), np.float32)
    sy, sxx, syy, sxy = (np.zeros_like(sx) for _ in range(4))
    for d in range(inner, outer + 1):
        x = Ap[:, outer + d: outer + d + K]
        y = Ap[:, outer - d: outer - d + K]
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    m = outer - inner + 1
    cov = sxy - sx * sy / m
    var = (sxx - sx * sx / m) * (syy - sy * sy / m)
    c = cov / np.sqrt(np.maximum(var, 1e-12))
    return np.clip(avg(c), 0, 1)

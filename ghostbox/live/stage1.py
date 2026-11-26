"""Stage 1: whole-band IQ checks, every frequency bin at once.

Each check returns a [frames, bins] score in [0, 1] (1 = looks like a voice on this bin) for the
requested (settled) frame slice. Inputs are the power spectrogram P (coarse bins, e.g. 125-250 Hz)
and, for the pitch checks, per-frame pitch maps computed from a fine (25 Hz) spectrum (pitch_frames).

Modulation checks come in families that are *alternatives* ("dsb" = ordinary AM, "ssb" =
single sideband): combine() takes the better-fitting family, so an SSB voice is not punished for
failing the AM twinning test and vice versa.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ..maps import OffsetSummer
from .config import CHECKS, FAMILIES

STAGE1 = tuple(c["id"] for c in CHECKS if c["stage"] == 1 and c["id"] != "profile")
PITCH = ("comb", "pitch_motion", "voicing")             # need the fine-spectrum pitch maps
FINE_HZ = 25.0                                          # fine spectrum resolution for the pitch comb


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


def combine(scores, checks, eps=0.02):
    """Confidence from check scores (arrays or scalars) with modulation families as alternatives.

    Shared checks and the best family enter one weighted geometric mean; the family enters with
    the larger of the families' total weights. Returns None if nothing is enabled."""
    w = {k: (checks[k]["weight"] if checks.get(k, {}).get("enabled") else 0.0) for k in scores}
    fam_ids = {i for ids in FAMILIES.values() for i in ids}
    shared = {k: v for k, v in scores.items() if k not in fam_ids and w.get(k, 0) > 0}
    weights = {k: w[k] for k in shared}
    best, best_w = None, 0.0
    for ids in FAMILIES.values():
        sub = {k: scores[k] for k in ids if k in scores and w.get(k, 0) > 0}
        if not sub:
            continue
        g = geo_mean(sub, w, eps)
        best = g if best is None else np.maximum(best, g)
        best_w = max(best_w, sum(w[k] for k in sub))
    if best is not None:
        shared["_modulation"], weights["_modulation"] = best, best_w
    return geo_mean(shared, weights, eps)


def family_scores(scores, checks, eps=0.02):
    w = {k: (checks[k]["weight"] if checks.get(k, {}).get("enabled") else 0.0) for k in scores}
    out = {}
    for fam, ids in FAMILIES.items():
        sub = {k: scores[k] for k in ids if k in scores and w.get(k, 0) > 0}
        if sub:
            out[fam] = geo_mean(sub, w, eps)
    return out


# ---------------- pitch comb from a fine spectrum ----------------
def pitch_frames(Pf, f0_min, f0_max, side_hz=3400.0, df=FINE_HZ):
    """Harmonic-comb strength and pitch at every fine bin, for each row of a fine power spectrum.

    Around an AM/SSB voice, voiced speech puts lines every F0 Hz. With the slowly varying spectral
    envelope removed, the log spectrum's autocorrelation along frequency at lag F0/df peaks; we
    require the second harmonic lag to agree (min of lag L and 2L) to reject random peaks.
    Returns (strength [rows, bins], f0_hz [rows, bins])."""
    A = np.log(Pf + 1e-12).astype(np.float32)
    A -= uniform_filter1d(A, size=9, axis=1, mode="nearest")      # remove the formant/tilt envelope
    np.clip(A, -2.5, 2.5, out=A)                                  # a carrier spike mustn't dominate the ripple
    R, K = A.shape
    W = int(side_hz / df)
    lmin = max(2, int(round(f0_min / df)))
    lmax = max(lmin + 1, int(round(f0_max / df)))
    lags = range(lmin, 2 * lmax + 1)

    def box(x):                                                   # sum over [k-W, k+W] along bins
        c = np.concatenate([np.zeros((R, 1), np.float32), np.cumsum(x, axis=1, dtype=np.float32)], axis=1)
        a = np.clip(np.arange(K) - W, 0, K)
        b = np.clip(np.arange(K) + W + 1, 0, K)
        return c[:, b] - c[:, a]

    e = box(A * A) + 1e-6
    corr = {}
    for L in lags:
        prod = np.zeros_like(A)
        prod[:, :K - L] = A[:, :K - L] * A[:, L:]
        corr[L] = box(prod) / e
    best = np.full((R, K), -1.0, np.float32)
    best_l = np.zeros((R, K), np.float32)
    for L in range(lmin, lmax + 1):
        c = np.minimum(corr[L], corr[2 * L])                      # harmonic consistency
        better = c > best
        best = np.where(better, c, best)
        # parabolic refinement of the peak lag
        if L - 1 in corr and L + 1 in corr:
            y0, y1, y2 = corr[L - 1], corr[L], corr[L + 1]
            den = y0 - 2 * y1 + y2
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / den, 0.0)
            best_l = np.where(better, L + np.clip(frac, -0.5, 0.5), best_l)
        else:
            best_l = np.where(better, L, best_l)
    return np.clip(best, 0, 1), (best_l * df).astype(np.float32)


# ---------------- the checks ----------------
def run(P, df, frame_rate, cfg, sl, force=(), pitch=None):
    """P: [T, K] coarse power; pitch: (C, F0) [T, K] maps (or None).
    force: check ids to compute even if disabled; they don't count toward the score unless enabled.
    Returns (scores {id: [n, K]}, combined [n, K], extra) with extra = {mode, env, f0}."""
    an, ch = cfg["analysis"], cfg["checks"]
    on = {k: ch[k]["enabled"] or k in force for k in STAGE1}
    inner = max(1, round(an["sideband_lo_hz"] / df))
    outer = max(inner + 2, round(an["sideband_hi_hz"] / df))
    S = OffsetSummer(P, max_offset(df, cfg, force))
    U, L = S(inner, outer), S(-outer, -inner)
    n_sb = outer - inner + 1

    syl = ch["syllabic"]
    lo, hi = syl["band_lo"], min(syl["band_hi"], 0.45 * frame_rate)
    sos = butter(4, (lo, hi), btype="bandpass", fs=frame_rate, output="sos")
    bp = lambda a: sosfiltfilt(sos, np.log(a + 1e-12), axis=0).astype(np.float32)
    n = max(3, int(an["stat_win_s"] * frame_rate))
    avg = lambda a, m=n: uniform_filter1d(a, size=m, axis=0, mode="nearest")
    corr = lambda a, b: avg(a * b) / np.sqrt(avg(a * a) * avg(b * b) + 1e-12)
    T, K = P.shape

    u, l = bp(U), bp(L)
    out, extra = {}, {}
    rms_u, rms_l = np.sqrt(avg(u * u)), np.sqrt(avg(l * l))
    noise_rms = np.median(0.5 * (rms_u + rms_l)) + 1e-12
    if on["twinning"]:
        out["twinning"] = np.clip(corr(u, l), 0, 1)[sl]
    if on["slide"]:
        out["slide"] = _sliding_twinning(U, L, bp, corr, df, frame_rate, ch["slide"])[sl]
    if on["syllabic"]:
        mod = np.sqrt(0.5 * (rms_u ** 2 + rms_l ** 2)) / noise_rms
        out["syllabic"] = np.clip(np.log(np.maximum(mod, 1e-6)) / np.log(syl["full_scale"]), 0, 1)[sl]
    if on["mirror"]:
        out["mirror"] = _fine_mirror(P, inner, outer, avg)[sl]
    if on["carrier"]:
        pk = maximum_filter1d(P, 3, axis=1)
        dens = (U + L) / (2 * n_sb)
        cdb = 10 * np.log10((pk + 1e-12) / (dens + 1e-12))
        out["carrier"] = np.clip(avg(cdb) / ch["carrier"]["full_db"], 0, 1)[sl]
    ssb_side = None
    if on["ssb"]:
        # one sideband carries a syllable rhythm, the other is (much) quieter ...
        asym = 10 * np.log10((avg(U) + 1e-12) / (avg(L) + 1e-12))
        full = ch["ssb"]["full_db"]
        lm = np.log(syl["full_scale"])
        mu = np.clip(np.log(np.maximum(rms_u / noise_rms, 1e-6)) / lm, 0, 1)
        ml = np.clip(np.log(np.maximum(rms_l / noise_rms, 1e-6)) / lm, 0, 1)
        # ... seen from its true carrier: voice energy is strongest near the carrier and falls away
        # (from the far edge it would rise instead, which would demodulate to reversed audio) ...
        mid = max(inner + 1, round(1200 / df))
        def falloff(near, far):
            return np.clip(10 * np.log10((avg(near) + 1e-12) / (avg(far) + 1e-12)) / 6, 0, 1)
        fu = falloff(S(inner, mid), S(mid + 1, outer))
        fl = falloff(S(-mid, -inner), S(-outer, -mid - 1))
        # ... and a real SSB voice has no carrier line inside its active side (an AM voice seen from
        # one of its own sidebands would)
        nb = outer - inner + 1
        M = maximum_filter1d(P, nb, axis=1)                    # centred: covers [j - nb//2, j + (nb-1)//2]
        peak_u = _shift_cols(M, inner + nb // 2)                # -> max over [k+inner, k+outer]
        peak_l = _shift_cols(M, -(outer - nb // 2))             # -> max over [k-outer, k-inner]
        def no_carrier(peak, side_sum):
            pdb = 10 * np.log10((avg(peak) + 1e-12) / (avg(side_sum) / nb + 1e-12))
            return np.clip((15 - pdb) / 6, 0, 1)
        # ... and the other side is just noise: from inside the voice band both sides are the same
        # voice and rise and fall together, which a true carrier edge never shows
        apart = 1 - np.clip(corr(u, l), 0, 1)
        usb = np.clip(asym / full, 0, 1) * mu * fu * no_carrier(peak_u, U) * apart
        lsb = np.clip(-asym / full, 0, 1) * ml * fl * no_carrier(peak_l, L) * apart
        out["ssb"] = np.maximum(usb, lsb).astype(np.float32)[sl]
        ssb_side = np.where(usb >= lsb, 1, -1).astype(np.int8)[sl]
    if on["rhythm"]:
        # loudness modulation per Hz at the syllable rate (3-6 Hz) vs above it (8-12 Hz): speech has
        # syllable rhythm but almost nothing faster; noise is spread evenly (ratio ~1.4).
        # Measured: noise median 1.4 (95th pct 4.4), synthetic AM / SSB voices ~6.
        e = np.log(U + L + 1e-12)
        sA = butter(3, (3.0, 6.0), btype="bandpass", fs=frame_rate, output="sos")
        sC = butter(3, (8.0, min(12.0, 0.45 * frame_rate)), btype="bandpass", fs=frame_rate, output="sos")
        m2 = max(3, int(2.0 * frame_rate))
        a = avg(sosfiltfilt(sA, e, axis=0) ** 2, m2) / 3.0
        c = avg(sosfiltfilt(sC, e, axis=0) ** 2, m2) / (min(12.0, 0.45 * frame_rate) - 8.0) + 1e-12
        out["rhythm"] = np.clip(np.log(np.maximum(a / c, 1e-6) / 2.0) / np.log(4.0), 0, 1).astype(np.float32)[sl]
    if on["shape"]:
        # sideband power density near / mid / far from the carrier (both sides): speech falls off
        b1, b2, b3 = (max(inner, round(x / df)) for x in (1000, 2000, outer * df))
        def dens(a, b):
            return (S(a, b) + S(-b, -a)) / max(1, 2 * (b - a + 1))
        near, mid, far = dens(inner, b1), dens(b1 + 1, b2), dens(b2 + 1, max(b2 + 1, b3))
        d1 = 10 * np.log10((avg(near) + 1e-12) / (avg(mid) + 1e-12))
        d2 = 10 * np.log10((avg(mid) + 1e-12) / (avg(far) + 1e-12))
        out["shape"] = (0.5 * np.clip(d1 / 6, 0, 1) + 0.5 * np.clip(d2 / 6, 0, 1)).astype(np.float32)[sl]
    if on["envstat"]:
        # loudness range of the sidebands over ~1.5 s (~2.56 sigma of dB = 10th..90th percentile for
        # a normal spread): syllables vs pauses swing by tens of dB, noise by a few
        e = 10 * np.log10(U + L + 1e-12)
        m15 = max(3, int(1.5 * frame_rate))
        sd = np.sqrt(np.maximum(avg(e * e, m15) - avg(e, m15) ** 2, 0))
        out["envstat"] = np.clip((2.56 * sd - 8) / ch["envstat"]["full_db"], 0, 1).astype(np.float32)[sl]
    if pitch is not None and any(on[k] for k in PITCH):
        C, F0 = pitch
        if on["comb"]:
            out["comb"] = np.clip((avg(C) - 0.08) / 0.35, 0, 1).astype(np.float32)[sl]
        if on["pitch_motion"]:
            w = np.clip(C - 0.1, 0, None)
            st = 12 * np.log2(np.maximum(F0, 1.0))
            m20 = max(3, int(0.8 * frame_rate))
            sw = avg(w, m20) + 1e-6
            mean = avg(w * st, m20) / sw
            var = np.maximum(avg(w * st * st, m20) / sw - mean ** 2, 0)
            step = np.abs(np.diff(st, axis=0, prepend=st[:1]))
            jump = avg(w * step, m20) / sw                       # big frame-to-frame jumps = not a glide
            support = np.clip(sw / 0.08, 0, 1)
            score = (np.clip(np.sqrt(var) / ch["pitch_motion"]["range_st"], 0, 1)
                     * np.clip(1 - jump / 4.0, 0, 1) * support)
            out["pitch_motion"] = score.astype(np.float32)[sl]
        if on["voicing"]:
            sv = butter(3, (2.0, min(8.0, 0.45 * frame_rate)), btype="bandpass", fs=frame_rate, output="sos")
            r = np.sqrt(avg(sosfiltfilt(sv, C, axis=0) ** 2, frame_rate))
            out["voicing"] = np.clip((r - 0.02) / 0.1, 0, 1).astype(np.float32)[sl]
        extra["f0"] = F0[sl]
        extra["c"] = C[sl]
    if on.get("resonances"):
        out["resonances"] = _resonances(S, df, inner, outer, avg, frame_rate)[sl]
    for k in ("ring", "broadband"):
        if on[k]:
            r0, r1 = round(ch[k]["lo_hz"] / df), round(ch[k]["hi_hz"] / df)
            r = bp(S(r0, r1) + S(-r1, -r0))
            out[k] = 1 - np.clip(corr(u + l, r), 0, 1)[sl]

    combined = combine(out, ch)
    if combined is None:
        combined = np.zeros((sl.stop - sl.start, K), np.float32)
    # demodulation mode per bin: +1 USB / -1 LSB where the SSB family fits better than AM, else 0
    mode = np.zeros((sl.stop - sl.start, K), np.int8)
    if ssb_side is not None:
        fams = family_scores(out, ch)
        if "ssb" in fams:
            dsb = fams.get("dsb", np.zeros_like(fams["ssb"]))
            mode = np.where(fams["ssb"] > dsb, ssb_side, 0).astype(np.int8)
    extra["mode"] = mode
    extra["env"] = (u + l).astype(np.float32)                    # syllable-band loudness, full window
    return out, combined.astype(np.float32), extra


def _shift_cols(M, s):
    """out[:, k] = M[:, k + s] (zero where that falls outside)."""
    out = np.zeros_like(M)
    K = M.shape[1]
    if s >= 0:
        out[:, :K - s] = M[:, s:]
    else:
        out[:, -s:] = M[:, :K + s]
    return out


def _sliding_twinning(U, L, bp, corr, df, frame_rate, prm):
    """Twinning along slanted lines: for each slide rate, shear the sideband envelopes so a voice
    drifting at that rate stays in one column, test, and shear back. Best rate wins."""
    T, K = U.shape
    rmax, steps = prm["max_khz_s"] * 1e3, max(1, int(prm["steps"]))
    t = (np.arange(T) - T / 2) / frame_rate
    best = np.zeros((T, K), np.float32)
    cols = np.arange(K)
    for r in np.linspace(-rmax, rmax, 2 * steps + 1):
        shift = np.round(r * t / df).astype(int)
        idx = np.clip(cols[None, :] + shift[:, None], 0, K - 1)
        rows = np.arange(T)[:, None]
        su, slw = U[rows, idx], L[rows, idx]
        c = np.clip(corr(bp(su), bp(slw)), 0, 1)
        back = np.zeros_like(c)
        np.put_along_axis(back, idx, c, axis=1)                  # undo the shear
        best = np.maximum(best, back)
    return best


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


def _resonances(S, df, inner, outer, avg, frame_rate):
    """Two moving resonances in the sidebands of every bin (both sides added): the strongest offset in
    250-1000 Hz and in 850-3000 Hz, each 3 dB over the sideband mean and 350 Hz+ apart, present for a
    good part of the last second, the upper one moving and the two moving independently (their gap
    changes). The coarse bins (~125 Hz) are about a formant's width."""
    lo1, hi1 = max(inner, round(250 / df)), round(1000 / df)
    lo2, hi2 = round(850 / df), min(outer, round(3000 / df))
    tot, n = 0.0, 0
    v1 = k1 = v2 = k2 = None
    for k in range(lo1, hi2 + 1):
        a = 10 * np.log10(S(k, k) + S(-k, -k) + 1e-12).astype(np.float32)
        tot, n = tot + a, n + 1
        if k <= hi1:
            if v1 is None:
                v1, k1 = a.copy(), np.full(a.shape, k, np.float32)
            else:
                m = a > v1
                v1[m], k1[m] = a[m], k
        if k >= lo2:
            if v2 is None:
                v2, k2 = a.copy(), np.full(a.shape, k, np.float32)
            else:
                m = a > v2
                v2[m], k2[m] = a[m], k
    mean = tot / n
    w = (((v1 - mean) > 3) & ((v2 - mean) > 3) & ((k2 - k1) * df >= 350)).astype(np.float32)
    m = max(3, int(frame_rate))
    sw = avg(w, m) + 1e-6

    def wsd(x):
        mu = avg(w * x, m) / sw
        return np.sqrt(np.maximum(avg(w * x * x, m) / sw - mu * mu, 0))
    sd2, sdd = wsd(k2 * df), wsd((k2 - k1) * df)
    return (np.clip((sw - 0.1) / 0.4, 0, 1) * np.clip((sd2 - 100) / 250, 0, 1)
            * np.clip((sdd - 80) / 250, 0, 1)).astype(np.float32)


def max_offset(df, cfg, force=()):
    """Largest frequency offset (bins) any enabled/forced check reaches; blocks need this much overlap."""
    an, ch = cfg["analysis"], cfg["checks"]
    inner = max(1, round(an["sideband_lo_hz"] / df))
    m = max(inner + 2, round(an["sideband_hi_hz"] / df))
    for k in ("ring", "broadband"):
        if ch[k]["enabled"] or k in force:
            m = max(m, round(ch[k]["hi_hz"] / df))
    if ch["slide"]["enabled"] or "slide" in force:                # sheared lines reach sideways
        m += int(ch["slide"]["max_khz_s"] * 1e3 * cfg["analysis"]["window_s"] / 2 / df) + 1
    return m + 2


def run_blocks(P, df, frame_rate, cfg, sl, force=(), pool=None, pitch=None, nblocks=16, min_block=2000):
    """run() split into frequency blocks processed in parallel (numpy releases the GIL).

    Each block is padded by the checks' maximum frequency reach, so results are identical to
    run() except the noise references (median over the block), which become local."""
    K = P.shape[1]
    nb = max(1, min(nblocks, K // min_block))
    if pool is None or nb == 1:
        return run(P, df, frame_rate, cfg, sl, force, pitch)
    pad = max_offset(df, cfg, force)
    edges = np.linspace(0, K, nb + 1).astype(int)

    def job(a, b):
        lo, hi = max(0, a - pad), min(K, b + pad)
        pb = (pitch[0][:, lo:hi], pitch[1][:, lo:hi]) if pitch is not None else None
        out, comb, ex = run(P[:, lo:hi], df, frame_rate, cfg, sl, force, pb)
        return a, b, lo, out, comb, ex

    parts = list(pool.map(lambda ab: job(*ab), zip(edges[:-1], edges[1:])))
    n = sl.stop - sl.start
    comb = np.empty((n, K), np.float32)
    out = {k: np.empty((n, K), np.float32) for k in parts[0][3]}
    extra = {k: np.empty((v.shape[0], K), v.dtype) for k, v in parts[0][5].items()}
    for a, b, lo, o, c, ex in parts:
        comb[:, a:b] = c[:, a - lo:b - lo]
        for k, v in o.items():
            out[k][:, a:b] = v[:, a - lo:b - lo]
        for k, v in ex.items():
            extra[k][:, a:b] = v[:, a - lo:b - lo]
    return out, comb, extra

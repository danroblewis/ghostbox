"""Playback audio processing for detections (what you hear and what is saved).

The detector itself always judges the raw demodulated audio; nothing here changes detection.

StreamingDenoiser  spectral gating with a minimum-statistics noise estimate, processed in
                   chunks of any size with exact overlap-add continuity between chunks.
SmoothGain         slow automatic level control: the gain glides across each chunk instead
                   of jumping between chunks, with a soft limiter for peaks.
fade()             raised-cosine fade in / out.
SyncDetector       synchronous AM / synchronous single-sideband detection (carrier recovery).
VoiceBand          hard band-pass around the voice band (elliptic, steep edges).
"""
import numpy as np
from scipy.signal import butter, ellip, firwin, lfilter, sosfilt, sosfilt_zi

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
    if cfg_audio.get("voice_band", True):
        y = VoiceBand(cfg_audio.get("voice_lo_hz", 300), cfg_audio.get("voice_hi_hz", 3000)).process(y)
    if cfg_audio.get("denoise", True):
        y = StreamingDenoiser(cfg_audio.get("denoise_strength", 1.0), cfg_audio.get("denoise_floor_db", -18)).process(y)
    y = SmoothGain().process(y)
    return fade(y, FS, cfg_audio.get("fade_s", 0.15), cfg_audio.get("fade_s", 0.15))


class SyncDetector:
    """Synchronous detection of AM on 16 kHz complex baseband, chunk by chunk with state.

    A second-order phase-locked loop follows the carrier (pull-in +-150 Hz: the track's frequency
    is only known to an analysis bin) on a 400 Hz decimated copy of the carrier band. Its phase
    error is weighted by the carrier's strength relative to its recent average, so during a fade
    the loop coasts on its last frequency and phase instead of chasing noise. The signal is then
    demodulated against the recovered carrier: linear at low SNR and free of the distortion
    selective fading causes in an envelope detector.
    sideband=False: synchronous AM (both sidebands). True: synchronous single sideband, keeping
    whichever side carries less interference (switching only on a clear difference)."""

    DEC = 40                                                     # 16 kHz -> 400 Hz for the loop

    def __init__(self, bw_hz=3000.0, sideband=False, loop_hz=4.0):
        self.sideband = sideband
        self.lp = butter(2, 150, fs=FS, output="sos")
        self.zl = None
        self.delay = int(round(np.sqrt(2) / (2 * np.pi * 150) * FS))   # the low-pass's group delay (samples)
        T, zeta = self.DEC / FS, 0.707
        wn = 2 * loop_hz / (zeta + 1 / (4 * zeta)) * 2 * np.pi
        self.alpha, self.beta = 2 * zeta * wn * T, (wn * T) ** 2
        self.theta = self.nu = None                              # loop phase, phase step per loop sample
        self.amp = None                                          # recent carrier strength
        self.n = 0                                               # input samples seen
        self.prev = None                                         # last (sample index, phase) for interpolation
        half = (min(bw_hz, 3800) - 150) / 2                      # one-sided pass: 150 Hz .. bw
        k = np.arange(129) - 64
        base = firwin(129, half, fs=FS)
        self.taps = {s: (base * np.exp(s * 2j * np.pi * (150 + half) * k / FS)).astype(np.complex64) for s in (1, -1)}
        self.hist = np.zeros(128, np.complex64)                  # FIR history across chunks
        self.side, self.eu, self.el = 1, None, None

    def _loop(self, d):
        out = np.empty(len(d))
        if self.theta is None:
            r = np.sum(d[1:] * np.conj(d[:-1])) if len(d) > 1 else 0
            self.nu, self.theta = float(np.angle(r)) if len(d) > 1 else 0.0, float(np.angle(d[0]))
            self.amp = float(np.mean(np.abs(d))) + 1e-12
        th, nu, a, b, amp = self.theta, self.nu, self.alpha, self.beta, self.amp
        lim = 2 * np.pi * 150 * self.DEC / FS
        for i, x in enumerate(d):
            out[i] = th
            m = abs(x)
            amp += 0.01 * (m - amp)                              # ~0.25 s memory of the carrier level
            e = (x * complex(np.cos(th), -np.sin(th))).imag / (amp + 1e-12)
            e = max(-1.0, min(1.0, e))                           # weak carrier -> small error -> coast
            th += nu + a * e
            nu = max(-lim, min(lim, nu + b * e))
        self.theta, self.nu, self.amp = th, nu, amp
        return out

    def process(self, y):
        y = np.asarray(y, np.complex64)
        L = len(y)
        if not L:
            return np.zeros(0, np.float32)
        if self.zl is None:
            self.zl = (sosfilt_zi(self.lp) * y[0]).astype(np.complex128)
        c, self.zl = sosfilt(self.lp, y, zi=self.zl)
        first = (-self.n) % self.DEC                             # loop samples on a fixed grid across chunks
        idx = np.arange(first, L, self.DEC)
        th = self._loop(c[idx]) if len(idx) else np.zeros(0)
        # phase at every input sample: the loop phases sit at (index - low-pass delay)
        pos = self.n + idx - self.delay
        if self.prev is not None:
            pos, th = np.r_[self.prev[0], pos], np.r_[self.prev[1], th]
        if len(pos):
            self.prev = (pos[-1], th[-1])
            step = self.nu / self.DEC
            grid = self.n + np.arange(L)
            ph = np.interp(grid, pos, th, left=th[0], right=th[-1])
            beyond = grid > pos[-1]                              # extrapolate past the last loop sample
            ph[beyond] = th[-1] + (grid[beyond] - pos[-1]) * step
        else:
            ph = np.zeros(L)
        self.n += L
        zc = y * np.exp(-1j * ph)                                # carrier on the real axis
        if not self.sideband:
            return zc.real.astype(np.float32)
        full = np.concatenate([self.hist, zc.astype(np.complex64)])
        self.hist = full[-128:]
        u = np.convolve(full, self.taps[1], mode="valid")
        l = np.convolve(full, self.taps[-1], mode="valid")
        eu, el = float(np.mean(np.abs(u) ** 2)), float(np.mean(np.abs(l) ** 2))
        self.eu = eu if self.eu is None else 0.8 * self.eu + 0.2 * eu
        self.el = el if self.el is None else 0.8 * self.el + 0.2 * el
        # interference adds energy to one side only: keep the quieter side (3 dB hysteresis)
        if self.side == 1 and self.eu > 2 * self.el:
            self.side = -1
        elif self.side == -1 and self.el > 2 * self.eu:
            self.side = 1
        return (2 * (u if self.side == 1 else l).real).astype(np.float32)


class VoiceBand:
    """Hard band-pass for playback: flat between lo and hi, at least 50 dB down outside."""

    def __init__(self, lo_hz=300.0, hi_hz=3000.0):
        lo, hi = max(50.0, float(lo_hz)), min(FS / 2 - 200, float(hi_hz))
        self.sos = ellip(6, 0.5, 50, [lo, max(lo + 100, hi)], btype="bandpass", fs=FS, output="sos")
        self.zi = None

    def process(self, x):
        x = np.asarray(x, np.float32)
        if self.zi is None:
            self.zi = np.zeros((self.sos.shape[0], 2))
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


def channel_hz(au):
    """Pre-detection channel half-width: just above the top of the voice band."""
    return float(au.get("voice_hi_hz", 3000)) + 300.0


def detector_for(au, mode):
    """The detector a track uses: None = envelope (AM) or product detector (SSB tracks)."""
    m = au.get("demod", "auto")
    if mode != 0 or m == "envelope":
        return None
    return SyncDetector(channel_hz(au), sideband=(m == "sync_sb"))

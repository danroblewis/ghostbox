"""The power spectrogram of wideband IQ (the live engine's waterfall)."""
import numpy as np
from scipy import fft as sp_fft


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

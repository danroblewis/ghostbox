"""Loading and saving IQ captures.

.cu8 = interleaved unsigned 8-bit I/Q (RTL-SDR, rtl_sdr)
.cs8 = interleaved signed 8-bit I/Q   (HackRF, hackrf_transfer)
"""
import re
from pathlib import Path

import numpy as np

CU8_OFFSET = 127.4
CU8_SCALE = 128.0


def parse_name(path):
    """Pull center freq / sample rate out of names like fm_96.6MHz_2.4Msps_g8.7.cu8."""
    name = Path(path).name
    fc = re.search(r"([\d.]+)MHz", name)
    fs = re.search(r"([\d.]+)Msps", name)
    return (float(fc.group(1)) * 1e6 if fc else None,
            float(fs.group(1)) * 1e6 if fs else 2.4e6)


def load_iq(path, start_s=0.0, dur_s=None, fs=None):
    """Return complex64 baseband samples with the DC offset removed (.cu8 or .cs8 by extension)."""
    if fs is None:
        fs = parse_name(path)[1]
    signed = Path(path).suffix == ".cs8"
    raw = np.memmap(path, dtype=np.int8 if signed else np.uint8, mode="r")
    a = int(start_s * fs) * 2
    b = len(raw) if dur_s is None else min(len(raw), a + int(dur_s * fs) * 2)
    u = np.asarray(raw[a:b], dtype=np.float32)
    off = 0.0 if signed else CU8_OFFSET
    x = ((u[0::2] - off) + 1j * (u[1::2] - off)).astype(np.complex64) / CU8_SCALE
    x -= x.mean()
    return x


load_cu8 = load_iq  # backwards-compatible name


def save_cu8(path, x):
    """Requantize complex samples (same scale as load_cu8) back to .cu8."""
    out = np.empty(2 * len(x), dtype=np.uint8)
    out[0::2] = np.clip(np.round(x.real * CU8_SCALE + CU8_OFFSET), 0, 255)
    out[1::2] = np.clip(np.round(x.imag * CU8_SCALE + CU8_OFFSET), 0, 255)
    out.tofile(path)

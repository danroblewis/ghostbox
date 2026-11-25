"""Inject synthetic AM voices with known (time, frequency) paths into a real capture.

This gives ground truth for the detector: a static AM voice, a hopping one, a
sliding one, and one that amplitude-modulates an existing FM broadcast station.

    python -m ghostbox.inject captures/fm_96.6MHz_2.4Msps_g8.7.cu8 captures/speech.wav
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, firwin, hilbert, oaconvolve, resample_poly, sosfiltfilt

from .iq import load_cu8, parse_name, save_cu8


AUDIO_SSB = 16000


def load_voice(wav_path, fs):
    rate, v = wavfile.read(wav_path)
    v = v.astype(np.float32)
    if v.ndim > 1:
        v = v.mean(axis=1)
    v = sosfiltfilt(butter(6, [300, 3400], btype="bandpass", fs=rate, output="sos"), v)  # voice-radio bandwidth
    v /= np.abs(v).max()
    v = np.tanh(4 * v) / np.tanh(4)  # broadcast-style compression: raises average modulation
    return resample_poly(v, int(fs), rate).astype(np.float32), int(fs)


def voice_segment(v, start, n):
    idx = (start + np.arange(n)) % len(v)
    return v[idx]


def am_signal(voice, freq_hz, fs, amp, depth=0.9):
    """AM with carrier; freq_hz is an array (per-sample) so hops/slides are phase-continuous."""
    phase = 2 * np.pi * np.cumsum(freq_hz) / fs
    return (amp * (1 + depth * voice) * np.exp(1j * phase)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("voice_wav")
    ap.add_argument("--level-db", type=float, default=-30,
                    help="AM carrier amplitude relative to the capture's total RMS")
    ap.add_argument("--station-offset-khz", type=float, default=-100,
                    help="offset of the FM station to amplitude-modulate (96.5 MHz in the 96.6 capture)")
    ap.add_argument("--station-depth", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    fc, fs = parse_name(args.capture)
    x = load_cu8(args.capture)
    n_total = len(x)
    rms = float(np.sqrt(np.mean(np.abs(x[: int(fs)]) ** 2)))
    amp = rms * 10 ** (args.level_db / 20)
    v, _ = load_voice(args.voice_wav, fs)
    rng = np.random.default_rng(args.seed)
    events = []

    def span(t0, t1):
        return int(t0 * fs), min(n_total, int(t1 * fs))

    # A: static AM voice
    a, b = span(2, 7)
    f = np.full(b - a, 500e3)
    x[a:b] += am_signal(voice_segment(v, 0, b - a), f, fs, amp)
    events.append({"kind": "static", "t0": 2, "t1": 7, "path": [[2, 500e3], [7, 500e3]]})

    # B: hopping AM voice, 250 ms dwell at random offsets
    a, b = span(9, 16)
    dwell = int(0.25 * fs)
    hops = rng.uniform(-1.0e6, 1.0e6, size=(b - a) // dwell + 1)
    f = np.repeat(hops, dwell)[: b - a]
    x[a:b] += am_signal(voice_segment(v, int(3 * fs), b - a), f, fs, amp)
    path = []
    for i, h in enumerate(hops[: (b - a) // dwell + 1]):
        path += [[9 + 0.25 * i, float(h)], [min(16, 9 + 0.25 * (i + 1)), float(h)]]
    events.append({"kind": "hop", "t0": 9, "t1": 16, "path": path})

    # C: sliding AM voice, -900 -> -500 kHz over 5 s
    a, b = span(18, 23)
    f = np.linspace(-900e3, -500e3, b - a)
    x[a:b] += am_signal(voice_segment(v, int(6 * fs), b - a), f, fs, amp)
    events.append({"kind": "slide", "t0": 18, "t1": 23, "path": [[18, -900e3], [23, -500e3]]})

    # D: voice amplitude-modulating an existing FM station (no new carrier)
    a, b = span(25, 29.5)
    off = args.station_offset_khz * 1e3
    t = np.arange(a, b) / fs
    shift = np.exp(-2j * np.pi * off * t).astype(np.complex64)
    taps = firwin(511, 110e3, fs=fs).astype(np.float32)
    station = oaconvolve(x[a:b] * shift, taps, mode="same") * np.conj(shift)
    x[a:b] += args.station_depth * voice_segment(v, int(1 * fs), b - a) * station
    events.append({"kind": "station_am", "t0": 25, "t1": 29.5, "path": [[25, off], [29.5, off]]})

    # E: single-sideband (upper) voice, no carrier
    a, b = span(25, 29.5)
    v = voice_segment(load_voice(args.voice_wav, AUDIO_SSB)[0], int(4 * AUDIO_SSB), int(4.5 * AUDIO_SSB))
    analytic = hilbert(v).astype(np.complex64)                    # audio with only positive frequencies
    up = resample_poly(analytic, int(fs // 1000), AUDIO_SSB // 1000)[: b - a]
    t = np.arange(a, a + len(up)) / fs
    x[a:a + len(up)] += (amp * 1.5 * up * np.exp(2j * np.pi * 250e3 * t)).astype(np.complex64)
    events.append({"kind": "ssb_usb", "t0": 25, "t1": 29.5, "path": [[25, 250e3], [29.5, 250e3]]})

    out = Path(args.out) if args.out else Path(args.capture).with_name(Path(args.capture).stem + "_inj.cu8")
    save_cu8(out, x)
    meta = {"source": str(args.capture), "fc": fc, "fs": fs, "level_db": args.level_db,
            "station_depth": args.station_depth, "events": events}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

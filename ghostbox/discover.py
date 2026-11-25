"""Blind AM-voice discovery: find voice-carrying AM transmitters in a capture with
no knowledge of where they are, then (optionally) check the picks against a
known channel grid and render AM-demodulated WAVs for listening.

    python -m ghostbox.discover captures/am_2.2MHz_4Msps_hackrf_a1l16g20.cs8 \
        --fmin 520e3 --fmax 1720e3 --grid 10e3 --wav-dir captures

The detector sees only the IQ file. --grid is used *after* detection, purely to
score the picks (AM broadcast carriers in the Americas sit on multiples of 10 kHz).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import firwin, oaconvolve, resample_poly

from .iq import load_iq, parse_name
from .maps import maps_from_power, power_frames


def chunked_power(path, fs, bin_hz=125, frame_rate=25, chunk_s=10.0, dc_guard_hz=2000):
    """Power spectrogram of a whole capture, loading chunk_s seconds of IQ at a time."""
    nfft = int(round(fs / bin_hz))
    n_total = Path(path).stat().st_size // 2
    dur = n_total / fs
    ts, Ps = [], []
    start = 0.0
    while start < dur - 1.0 / frame_rate:
        x = load_iq(path, start, chunk_s, fs)
        t, f, P = power_frames(x, fs, nfft, frame_rate)
        ts.append(t + start)
        Ps.append(P)
        start += len(t) / frame_rate
    P = np.concatenate(Ps)
    P[:, np.abs(f) <= dc_guard_hz] = np.median(P, axis=1, keepdims=True)
    return np.concatenate(ts), f, P


def pick_peaks(score, f, n, min_sep_hz):
    order = np.argsort(score)[::-1]
    picks = []
    for k in order:
        if all(abs(f[k] - f[j]) >= min_sep_hz for j in picks):
            picks.append(k)
        if len(picks) == n:
            break
    return picks


def am_demod_wav(path, fs, offset_hz, out, start=0.0, dur=20.0, audio_bw=4500):
    x = load_iq(path, start, dur, fs)
    t = np.arange(len(x)) / fs
    y = x * np.exp(-2j * np.pi * offset_hz * t).astype(np.complex64)
    decim = int(fs // 16000)
    y = oaconvolve(y, firwin(801, audio_bw, fs=fs).astype(np.float32), mode="same")[::decim]
    rate = fs / decim
    env = np.abs(y)
    a = env - np.convolve(env, np.ones(int(rate / 10)) / int(rate / 10), "same")  # strip carrier
    a = np.clip(a / (np.percentile(np.abs(a), 99.5) + 1e-12), -1, 1) * 0.8
    a = resample_poly(a, 16000, int(rate)) if int(rate) != 16000 else a
    wavfile.write(out, 16000, (a * 32767).astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--fmin", type=float, help="absolute Hz; restrict search (e.g. a band edge)")
    ap.add_argument("--fmax", type=float)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-sep", type=float, default=7e3)
    ap.add_argument("--grid", type=float, help="score-only: known channel spacing in Hz")
    ap.add_argument("--grid-tol", type=float, default=250)
    ap.add_argument("--wav-dir", help="write AM-demodulated WAVs for picks and controls here")
    ap.add_argument("--n-wav", type=int, default=6)
    ap.add_argument("--prominence-hz", type=float, default=30e3,
                    help="half-width of the frequency neighbourhood used for prominence")
    ap.add_argument("--no-ring", action="store_true",
                    help="disable the FM-rejection ring test (use where there is no FM, e.g. AM broadcast band)")
    args = ap.parse_args()

    fc, fs = parse_name(args.capture)
    t, f, P = chunked_power(args.capture, fs)
    keep = np.ones(len(f), bool)
    if args.fmin is not None:
        keep &= fc + f >= args.fmin
    if args.fmax is not None:
        keep &= fc + f <= args.fmax
    f, P = f[keep], P[:, keep]
    M = maps_from_power(t, f, P, **({"ring_hz": None} if args.no_ring else {}))

    # stations are always on, so rank by raw evidence (not novelty), averaged over the capture.
    # Then take prominence along frequency: a real AM transmitter is centred on its carrier, so
    # its evidence peaks there; broadband interference whose loudness pulses everywhere at once
    # looks "symmetric" about every point and forms a plateau, which this cancels.
    ev_raw = M.raw.mean(axis=0)
    df = float(f[1] - f[0])
    ev = ev_raw - median_filter(ev_raw, size=int(2 * args.prominence_hz / df) | 1, mode="nearest")
    picks = pick_peaks(ev, f, args.top, args.min_sep)
    power_db = 10 * np.log10(P.mean(axis=0))
    floor = np.median(power_db)

    print(f"{'rank':>4} {'freq (kHz)':>11} {'evidence':>8} {'active':>6} {'carrier dB':>10}"
          + ("  grid offset" if args.grid else ""))
    hits = 0
    for r, k in enumerate(picks, 1):
        fa = fc + f[k]
        line = (f"{r:>4} {fa / 1e3:>11.2f} {ev[k]:>8.3f} {np.mean(M.raw[:, k] > 0.5):>6.0%}"
                f" {power_db[max(0, k - 2):k + 3].max() - floor:>10.1f}")
        if args.grid:
            off = fa - np.round(fa / args.grid) * args.grid
            hit = abs(off) <= args.grid_tol
            hits += hit
            line += f"  {off:+7.0f} Hz {'ON GRID' if hit else ''}"
        print(line)
    if args.grid:
        chance = 2 * args.grid_tol / args.grid
        print(f"\n{hits}/{len(picks)} picks within +-{args.grid_tol:.0f} Hz of the {args.grid / 1e3:.0f} kHz grid"
              f" (chance level {chance:.0%})")

    if args.wav_dir:
        out = Path(args.wav_dir)
        # controls: strong carriers the detector did NOT pick
        bins_db = power_db - floor
        strong = [k for k in np.argsort(bins_db)[::-1] if bins_db[k] > 15
                  and all(abs(f[k] - f[j]) > 10e3 for j in picks)]
        controls = []
        for k in strong:
            if all(abs(f[k] - f[j]) > 10e3 for j in controls):
                controls.append(k)
            if len(controls) == 2:
                break
        stem = Path(args.capture).stem.split("_")[0]
        jobs = [(k, f"{stem}_pick{r}_{(fc + f[k]) / 1e3:.1f}kHz", f"Blind pick #{r}: evidence {ev[k]:.3f}")
                for r, k in enumerate(picks[: args.n_wav], 1)]
        jobs += [(k, f"{stem}_control_{(fc + f[k]) / 1e3:.1f}kHz",
                  f"Control: strong carrier ({bins_db[k]:.0f} dB) the detector did NOT pick, evidence {ev[k]:.3f}")
                 for k in controls]
        for k, name, note in jobs:
            am_demod_wav(args.capture, fs, f[k], out / f"{name}.wav")
            (out / f"{name}.txt").write_text(note + f". AM-demodulated, first 20 s.")
            print(f"wrote {out / name}.wav")


if __name__ == "__main__":
    main()

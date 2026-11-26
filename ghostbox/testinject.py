"""Test the detector on real signals: inject narrowband IQ recordings (an AM voice, Morse, data...) into
a real wideband capture at chosen frequencies and levels, run the detector over the result with the
default checks, and report what it found near each one.

    python -m ghostbox.testinject OUT_DIR --base iq_ring/A.cs8 iq_ring/B.cs8 --fc 9152060 --fs 2e6 \\
        --add ham_75m.wav:-300000:35 --add morse.wav:250000:30 [--detail]

--add FILE:OFFSET_HZ:LEVEL_DB  FILE is a stereo WAV of IQ (I left, Q right; any rate, int16 or float),
    e.g. an SDR# / SDRuno / KiwiSDR IQ recording. It's placed OFFSET_HZ from the base capture's centre,
    with its strongest line (an AM carrier) LEVEL_DB above the capture's noise in one 125 Hz bin.
    AM voices start being caught around +30 dB. Short files loop over the capture.
--base  raw 8-bit IQ (.cs8, as the IQ ring stores it) at --fc / --fs, concatenated in order.
"""
import argparse
import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .live.autotune import analyse_iq, analysis_config, rescore
from .live.config import DEFAULT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--base", nargs="+", required=True, type=Path)
    ap.add_argument("--fc", type=float, required=True, help="base capture centre (Hz)")
    ap.add_argument("--fs", type=float, required=True, help="base capture sample rate (Hz)")
    ap.add_argument("--add", action="append", default=[], metavar="FILE:OFFSET_HZ:LEVEL_DB")
    ap.add_argument("--detail", action="store_true", help="list every candidate near each injection")
    a = ap.parse_args()
    fc, fs = a.fc, a.fs
    a.out_dir.mkdir(parents=True, exist_ok=True)

    base = np.concatenate([np.fromfile(p, np.int8).astype(np.float32) for p in a.base]).view(np.complex64) / 128
    n = len(base)
    nfft = int(fs / 125)                                  # the analysis bin
    def spectrum(z):
        seg = z[: nfft * 8].reshape(8, nfft)
        return np.mean(np.abs(np.fft.fft(seg * np.hanning(nfft), axis=1)) ** 2, axis=0)
    Pn = np.median(spectrum(base))                        # noise per bin
    x = base.copy()
    inj = []
    for spec in a.add:
        path, off, level = spec.rsplit(":", 2)
        off, level = float(off), float(level)
        rate, w = wavfile.read(path)
        w = w.astype(np.float32)
        z = (w[:, 0] + 1j * w[:, 1]).astype(np.complex64)
        fr = Fraction(int(fs), int(rate)).limit_denominator(2000)
        z = np.resize(resample_poly(z, fr.numerator, fr.denominator).astype(np.complex64), n)
        Pm = spectrum(z)
        kc = int(Pm.argmax())
        f_car = (kc if kc < nfft // 2 else kc - nfft) * fs / nfft      # the recording's strongest line
        z *= np.sqrt(Pn * 10 ** (level / 10) / (Pm[kc] + 1e-30))
        x += z * np.exp(2j * np.pi * off * np.arange(n) / fs).astype(np.complex64)
        inj.append({"file": Path(path).name, "f": fc + off + f_car, "level": level})
    # the capture is whole 8-bit steps and an injection is often smaller than one: dither before rounding
    # (as a real receiver's own noise does before its ADC), or rounding just restores the capture
    rng = np.random.default_rng(0)
    iq = np.stack([x.real, x.imag], 1) * 128
    iq += rng.random(iq.shape, dtype=np.float32) - rng.random(iq.shape, dtype=np.float32)
    cap = a.out_dir / "injected.cs8"
    np.clip(np.round(iq), -127, 127).astype(np.int8).tofile(cap)

    dur = n / fs
    cfg = copy.deepcopy(DEFAULT)
    cfg["sdr"].update(center_hz=fc, sample_rate=fs, fmin_hz=None, fmax_hz=None)
    tcfg, score_all = analysis_config(cfg)
    tcfg["checks"] = copy.deepcopy(DEFAULT["checks"])     # the live defaults, not the calibration set
    tcfg["checks"]["profile"].update(weight=0.0, enabled=False)
    boxes = analyse_iq(str(a.out_dir), tcfg, score_all, [cap], fs, fc, dur, 0)
    live = tcfg["checks"]
    no_tone = copy.deepcopy(live)
    no_tone["tone"].update(weight=0.0, enabled=False)
    near = lambda j: [b for b in boxes if b["fmin"] - 3e3 <= j["f"] <= b["fmax"] + 3e3]
    g = lambda d, k: d.get(k, float("nan")) if d.get(k) is not None else float("nan")

    print(f"{dur:.0f} s at {fc / 1e6:.4f} MHz: {len(boxes)} candidates")
    for j in inj:
        nb = near(j)
        best = max(nb, key=lambda b: rescore(b, live), default=None)
        line = f"  {j['file']} @ {j['f'] / 1e6:.4f} MHz, {j['level']:+.0f} dB: {len(nb)} candidates"
        if best:
            s = best["scores"]
            line += (f"; best confidence {rescore(best, live):.2f} (without the tone check {rescore(best, no_tone):.2f}), "
                     f"ML voice {g(s, 'vad'):.2f}, formants {g(s, 'formants'):.2f}, tone {g(s, 'tone'):.2f}, comb {g(s, 'comb'):.2f}, "
                     f"twinning {g(s, 'twinning'):.2f}")
        print(line)
        if a.detail:
            for b in sorted(nb, key=lambda b: b["t0"]):
                info = b.get("info", {})
                print(f"      {(b['fmin'] - j['f']) / 1e3:+.1f}..{(b['fmax'] - j['f']) / 1e3:+.1f} kHz, {b['t1'] - b['t0']:.1f} s, "
                      f"{b.get('mode')}, confidence {rescore(b, live):.2f}, tonality {g(info, 'tonality'):.2f}, "
                      f"carrier steps/s {g(info, 'carrier_steps_hz'):.1f}, ML voice {g(b['scores'], 'vad'):.2f}")
    others = [b for b in boxes if not any(b in near(j) for j in inj)]
    top = sorted(((rescore(b, live), b) for b in others), key=lambda t: -t[0])[:5]
    print("  strongest other candidates (MHz, confidence):", [(round((b["fmin"] + b["fmax"]) / 2e6, 4), round(c, 2)) for c, b in top])


if __name__ == "__main__":
    main()

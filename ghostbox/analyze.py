"""Compute whole-band AM-voice maps for a capture, plot them, and (if a ground-truth
.json from ghostbox.inject sits next to the capture) score detection.

    python -m ghostbox.analyze captures/fm_96.6MHz_2.4Msps_g8.7_inj.cu8 -o out.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .iq import load_cu8, parse_name
from .maps import compute_maps


def truth_freq(event, t):
    """Ground-truth frequency of an event at times t (NaN outside the event)."""
    pts = np.array(event["path"], dtype=float)
    out = np.full_like(t, np.nan, dtype=float)
    inside = (t >= event["t0"]) & (t < event["t1"])
    if event["kind"] == "hop":
        for (ta, fa), (tb, _) in zip(pts[0::2], pts[1::2]):
            m = (t >= ta) & (t < tb)
            out[m] = fa
    else:
        out[inside] = np.interp(t[inside], pts[:, 0], pts[:, 1])
    out[~inside] = np.nan
    return out


def evaluate(maps, events, tol_hz=6e3, edge_s=0.6):
    """Per event: how often the single global argmax of the score lands on the truth.

    Also reports truth-vs-background score contrast. edge_s trims event starts/ends
    (and hop boundaries) where the 1 s statistics window is only partly filled.
    """
    t, f, S = maps.t, maps.f, maps.score
    rows = []
    for ev in events:
        ft = truth_freq(ev, t)
        ok = ~np.isnan(ft) & (t >= ev["t0"] + edge_s) & (t < ev["t1"] - edge_s)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        kt = np.abs(f[None, :] - ft[idx, None]).argmin(axis=1)
        near = np.abs(f[None, :] - ft[idx, None]) <= tol_hz
        top = S[idx].argmax(axis=1)
        hit = np.abs(f[top] - ft[idx]) <= tol_hz
        on = np.array([S[i, max(0, k - 3): k + 4].max() for i, k in zip(idx, kt)])
        bg = np.where(near, -np.inf, S[idx]).max(axis=1)
        rows.append({"kind": ev["kind"], "frames": len(idx), "top1_hit": float(hit.mean()),
                     "truth_score": float(np.median(on)), "best_other": float(np.median(bg))})
    return rows


def plot(maps, events, out, title, fc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step, pool = 1, 16  # max-pool 16 bins so narrow lines survive downscaling

    def shrink(a):
        a = a[::step]
        k = a.shape[1] // pool * pool
        return a[:, :k].reshape(a.shape[0], -1, pool).max(axis=2)

    t = maps.t[::step]
    ext = [maps.f[0] / 1e3, maps.f[-1] / 1e3, t[-1], t[0]]
    panels = [
        ("power (dB)", 10 * np.log10(shrink(maps.P) + 1e-12), "viridis", None),
        ("sideband syllabic modulation (log10, vs noise)", np.log10(shrink(maps.mod)), "magma", (0, 1.5)),
        ("sideband twinning corr (AM~+1)", shrink(maps.sym), "coolwarm", (-1, 1)),
        ("ring co-modulation (FM-like~+1)", shrink(maps.ring), "coolwarm", (-1, 1)),
        ("AM-voice evidence (raw)", shrink(maps.raw), "inferno", None),
        ("novelty score (raw - per-bin baseline)", shrink(maps.score), "inferno", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(32, 9), sharey=True)
    for ax, (name, img, cmap, lim) in zip(axes, panels):
        if lim is None:
            lo, hi = np.percentile(img, [2, 99.8])
        else:
            lo, hi = lim
        im = ax.imshow(img, aspect="auto", extent=ext, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel(f"offset from {fc / 1e6:.1f} MHz (kHz)")
        fig.colorbar(im, ax=ax, fraction=0.04)
    axes[0].set_ylabel("time (s)")
    for ev in events:
        ft = truth_freq(ev, maps.t)
        axes[-1].plot(ft / 1e3 + 40, maps.t, color="cyan", lw=0.8, alpha=0.7)
    fig.suptitle(title + ("   (cyan = injected truth, drawn +40 kHz to the right)" if events else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    fc, fs = parse_name(args.capture)
    x = load_cu8(args.capture, args.start, args.dur, fs)
    maps = compute_maps(x, fs)
    maps.t += args.start

    truth = Path(args.capture).with_suffix(".json")
    events = json.loads(truth.read_text())["events"] if truth.exists() else []
    for r in evaluate(maps, events):
        print(f"{r['kind']:>11}: top-1 hit {r['top1_hit']:5.1%}   score on truth {r['truth_score']:.2f}"
              f"   best elsewhere {r['best_other']:.2f}   ({r['frames']} frames)")

    out = args.out or str(Path(args.capture).with_suffix(".png"))
    plot(maps, events, out, Path(args.capture).name, fc)


if __name__ == "__main__":
    main()

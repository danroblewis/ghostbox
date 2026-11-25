"""Hidden-path search: find the hopping/sliding track through a (time x frequency)
evidence map that best explains it as one voice.

Viterbi over states = frequency bins plus one "off" state (no ghost). Per frame:

  on(k)  <- S(k) - cost + max( best on-state within +-slide_bins of k   (stay / slide)
                              best on-state anywhere - hop_penalty     (hop)
                              off - start_penalty )                    (appear)
  off    <- max( off, best on-state anywhere )                         (vanish)

`cost` is what a frame must earn to be worth staying on, so noise cannot build up
a path; evidence that lines up over time (a voice that stays, slides or hops)
can. Slide uses a sliding-window max along frequency and hop uses the previous
frame's global max, so each step is O(bins) instead of O(bins^2).
"""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Track:
    t: np.ndarray        # frame times
    f: np.ndarray        # path frequency per frame (NaN when off)
    segments: list       # [(t_start, t_end, kind_of_entry)] contiguous on-runs


def pool_freq(S, f, pool):
    k = S.shape[1] // pool * pool
    return S[:, :k].reshape(len(S), -1, pool).max(axis=2), f[:k].reshape(-1, pool).mean(axis=1)


def viterbi(S, f, t, cost=0.4, hop_penalty=1.5, start_penalty=1.5, slide_bins=8):
    T, K = S.shape
    NEG = -1e18
    on = np.full(K, NEG)
    off = 0.0
    # back[t, k]: previous bin for on-state (or -1 = came from off); off_back[t]: prev bin or -1
    back = np.empty((T, K), dtype=np.int32)
    off_back = np.empty(T, dtype=np.int32)
    idx = np.arange(K)
    for i in range(T):
        padded = np.concatenate([np.full(slide_bins, NEG), on, np.full(slide_bins, NEG)])
        win = sliding_window_view(padded, 2 * slide_bins + 1)       # [K, 2D+1]
        j = win.argmax(axis=1)
        local, local_from = win[idx, j], idx + j - slide_bins
        g = int(on.argmax())
        hop = on[g] - hop_penalty
        start = off - start_penalty
        best = local.copy()
        frm = local_from.astype(np.int32)
        m = hop > best
        best[m], frm[m] = hop, g
        m = start > best
        best[m], frm[m] = start, -1
        new_on = best + S[i] - cost
        if on[g] > off:
            new_off, off_back[i] = on[g], g
        else:
            new_off, off_back[i] = off, -1
        back[i] = frm
        on, off = new_on, new_off

    # backtrack from the better final state
    f_path = np.full(T, np.nan)
    k = int(on.argmax()) if on.max() > off else -1
    for i in range(T - 1, -1, -1):
        if k >= 0:
            f_path[i] = f[k]
            k = int(back[i, k])
        else:
            k = int(off_back[i])
    segments = []
    run = None
    for i in range(T):
        if not np.isnan(f_path[i]) and run is None:
            run = i
        if (np.isnan(f_path[i]) or i == T - 1) and run is not None:
            end = i if np.isnan(f_path[i]) else i + 1
            segments.append((float(t[run]), float(t[end - 1])))
            run = None
    return Track(t, f_path, segments)

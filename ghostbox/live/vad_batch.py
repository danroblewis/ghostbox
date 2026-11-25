"""Batched Silero VAD: score many audios together, one 512-sample block per model call.

vad_probs_batch(audios) gives the same per-audio result as running stage2.vad_probs on each
audio separately (normalise by max abs, reset the model, one call per 32 ms block), but the
Silero JIT model is fed a [batch, 512] tensor so every audio advances one block per call.
Shorter audios are zero-padded; their outputs past their own end are dropped.
"""
import threading

import numpy as np

FS = 16000
BLOCK = 512
MAX_BATCH = 12               # see vad_probs_batch: larger batches hit a slow path in the model

_local = threading.local()      # one model per calling thread: the model carries recurrent state
_status = {"error": None}


def _model():
    model = getattr(_local, "vad", None)
    if model is None and _status["error"] is None:
        try:
            import torch
            from silero_vad import load_silero_vad
            torch.set_num_threads(1)
            model = _local.vad = load_silero_vad()
        except Exception as e:  # pragma: no cover
            _status["error"] = str(e)
    return model


def vad_probs_batch(audios):
    """audios: list of 1-D float arrays at 16 kHz. Returns a list (same order) of 1-D float32
    arrays: Silero speech probability per complete 512-sample block (len(a) // 512 values;
    empty if the audio is shorter than one block or Silero is unavailable)."""
    nblk = [len(a) // BLOCK for a in audios]
    out = [np.zeros(0, np.float32) for _ in audios]
    idx = [i for i, k in enumerate(nblk) if k > 0]
    if not idx:
        return out
    model = _model()
    if model is None:
        return out
    # The JIT model gets ~7x slower per item at batch >= 16 on CPU (measured with torch 2.11,
    # 1 thread: ~30 us/item at batch 8..15 vs ~200 us/item at 16+), so run groups of at most
    # MAX_BATCH, balanced in size, with similar lengths together to waste little on padding.
    idx.sort(key=lambda i: nblk[i])
    ngroups = -(-len(idx) // MAX_BATCH)
    for g in range(ngroups):
        group = idx[g * len(idx) // ngroups:(g + 1) * len(idx) // ngroups]
        for i, p in zip(group, _run(model, [audios[i] for i in group], [nblk[i] for i in group])):
            out[i] = p
    return out


def _run(model, audios, nblk):
    import torch
    kmax = max(nblk)
    x = np.zeros((len(audios), kmax * BLOCK), np.float32)
    for row, (a, k) in enumerate(zip(audios, nblk)):
        a = np.asarray(a[:k * BLOCK])
        x[row, :k * BLOCK] = np.clip(a / (np.abs(a).max() + 1e-9), -1, 1)
    probs = np.empty((len(audios), kmax), np.float32)
    with torch.no_grad():
        model.reset_states()
        for b in range(kmax):
            xb = torch.from_numpy(np.ascontiguousarray(x[:, b * BLOCK:(b + 1) * BLOCK]))
            probs[:, b] = model(xb, FS).numpy()[:, 0]
    return [probs[row, :k].copy() for row, k in enumerate(nblk)]

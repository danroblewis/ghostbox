"""Local speech recognition for finished detections and hop chains (engines: see asr.py).

One model is loaded once, in-process, and kept resident (reloaded only if the model name or
thread settings change). A small pool of worker threads shares it; each worker transcribes a
*batch* of clips at a time with faster-whisper's batched pipeline. Whisper always encodes a full
30 s window however short the clip, so batching many clips — each still in its own window, so no
context leaks between them — is what makes it keep up (~8x one-at-a-time on CPU).

The queue is prioritised by detection confidence: the most confident clips are transcribed first,
the least confident are dropped when the queue is full, and clips that waited too long are skipped.

Whisper hallucinates on noise (it will happily "hear" phrases in static), so every result carries
Whisper's own no_speech_prob for the UI to show.
"""
import heapq
import itertools
import json
import threading
import time
import traceback

import re

import numpy as np

from . import asr

FS = 16000



# Whisper was trained on subtitled video; on noise it falls back to the commonest subtitle lines.
STOCK = {"you", "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
         "thank you so much for watching", "please subscribe", "subscribe", "the end", "bye", "bye bye", "so",
         "okay", "oh", "uh", "um", "hmm", "music", "applause", "silence", "i'm sorry", "i love you",
         "subtitles by the amara.org community", "thanks for watching and see you next time", "see you next time"}


def _phrase_re(phrase):
    """A user phrase as a pattern that ignores case, punctuation, apostrophes and spaces between
    its letters ("who's" = "whos", "Dragon Ball" = "dragonball"), matching whole words only."""
    chars = [c for c in phrase.lower() if c.isalnum()]
    if not chars:
        return None
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\W_]*".join(re.escape(c) for c in chars) + r"(?![A-Za-z0-9])",
                      re.IGNORECASE)


def collapse_loops(text, min_repeats=3, max_n=10):
    """Whisper's repetition loop ("to be the one to be the one to be the one ..."): a run of 3+
    back-to-back copies of the same 1..10-word phrase becomes one copy (and a trailing partial
    copy is dropped)."""
    words = text.split()
    key = [re.sub(r"[^\w']", "", w.lower()) for w in words]
    out, i = [], 0
    while i < len(words):
        best = None
        for n in range(1, max_n + 1):
            unit = key[i:i + n]
            if len(unit) < n:
                break
            k = 1
            while key[i + k * n:i + (k + 1) * n] == unit:
                k += 1
            if k >= min_repeats and (best is None or k * n > best[0] * best[1]):
                best = (k, n)
        if best:
            k, n = best
            out += words[i:i + n]
            j = i + k * n
            m = 0                                          # a trailing partial copy
            while j + m < len(words) and m < n and key[j + m] == key[i + m]:
                m += 1
            i = j + (m if m < n else 0)
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def strip_stock(text, phrases=()):
    """Remove the user's phrases wherever they occur, then stock sentences ("The end.", "Thank
    you.") that make up a whole sentence; keep the rest."""
    text = collapse_loops(text)
    for ph in phrases:
        rx = _phrase_re(ph)
        if rx is not None:
            text = rx.sub(" ", text)
    text = re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ,;:-")
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    keep = [p for p in parts if p and not is_stock_phrase(p)]
    return " ".join(keep).strip()


def is_stock_phrase(text):
    t = re.sub(r"[^a-z' .]+", " ", text.lower()).replace(".", " ")
    t = " ".join(t.split())
    if not t:                                              # no Latin words: filler only if it has no letters
        return not any(ch.isalpha() for ch in text)       # at all ("...", "100,000"); other scripts are kept
    words = t.split()
    if len(set(words)) == 1 and len(words) > 1:            # "you you you", "the the the"
        return True
    return t in STOCK or "amara.org" in text.lower()

class Transcriber:
    def __init__(self, config, emit, on_result=None):
        self.config = config
        self.emit = emit
        self.on_result = on_result          # (keys, field, result): update in-memory detections
        self._heap, self._seq = [], itertools.count()
        self._cv = threading.Condition()
        self._model_lock = threading.Lock()
        self.model = self.pipe = None
        self.model_name, self._model_key = None, None
        self.error = ""
        self.busy = 0
        self.stats = {"done": 0, "dropped": 0, "stale": 0, "rate": 0.0}
        self._recent = []                   # (time, clips) for a throughput estimate
        self._workers = []
        self._ensure_workers()

    # ---------------- queue ----------------
    @property
    def q(self):                            # kept for status code that asks q.qsize()
        return self

    def qsize(self):
        return len(self._heap)

    def submit(self, item, priority=0.0):
        """item: {kind: track|chain, id, chain, audio (float32 16 kHz), keys, meta_paths, wav}"""
        if len(item["audio"]) < FS * 0.3:
            return
        cfg = self.config.get()["whisper"]
        item["t_submit"] = time.time()
        with self._cv:
            heapq.heappush(self._heap, (-float(priority), next(self._seq), item))
            limit = int(cfg.get("max_queue", 64))
            if len(self._heap) > limit:      # drop the least confident clip
                worst = max(range(len(self._heap)), key=lambda i: (self._heap[i][0], -self._heap[i][1]))
                self._heap[worst] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                self.stats["dropped"] += 1
            self._cv.notify()
        self._ensure_workers()

    def _take_batch(self, size, linger=0.25):
        with self._cv:
            while not self._heap:
                self._cv.wait()
            deadline = time.time() + linger  # give a burst of detections a moment to join the batch
            while len(self._heap) < size and time.time() < deadline:
                self._cv.wait(timeout=max(0.0, deadline - time.time()))
            batch = [heapq.heappop(self._heap)[2] for _ in range(min(size, len(self._heap)))]
        return batch

    # ---------------- model ----------------
    def _ensure_workers(self):
        n = int(self.config.get()["whisper"].get("workers", 2))
        while len(self._workers) < n:
            th = threading.Thread(target=self._run, args=(len(self._workers),), daemon=True)
            self._workers.append(th)
            th.start()

    def _load(self, cfg):
        key = (cfg["model"], int(cfg.get("cpu_threads", 8)), int(cfg.get("workers", 2)))
        with self._model_lock:
            if self._model_key != key:
                self.model = None                           # free the old one before loading the next
                self.model = asr.load(cfg["model"], cpu_threads=key[1], workers=key[2])
                self.model_name, self._model_key = cfg["model"], key
            return self.model

    # ---------------- workers ----------------
    def _run(self, idx):
        while True:
            cfg = self.config.get()["whisper"]
            if idx >= int(cfg.get("workers", 2)):          # pool shrunk: idle this worker
                time.sleep(2)
                continue
            batch = self._take_batch(int(cfg.get("batch_size", 16)))
            if not cfg["enabled"]:
                continue
            fresh = [it for it in batch if time.time() - it["t_submit"] <= cfg.get("max_age_s", 120)]
            self.stats["stale"] += len(batch) - len(fresh)
            if not fresh:
                continue
            self.busy += 1
            try:
                results = self._transcribe(fresh, cfg)
                for it, res in zip(fresh, results):
                    self._deliver(it, res)
                self.stats["done"] += len(fresh)
                now = time.time()
                self._recent = [(t, n) for t, n in self._recent if now - t < 60] + [(now, len(fresh))]
                span = max(1.0, now - self._recent[0][0]) if len(self._recent) > 1 else 60.0
                self.stats["rate"] = round(sum(n for _, n in self._recent) / span, 2)
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                self.busy -= 1

    def _transcribe(self, items, cfg):
        eng = self._load(cfg)
        lang = None if cfg["language"] in ("", "auto") else cfg["language"]
        out = []
        for it, (raw, doubt) in zip(items, eng.transcribe([it["audio"] for it in items], lang)):
            text = strip_stock(raw, cfg.get("filter_phrases", ()))   # training-data filler is dropped
            out.append({"text": text, "no_speech_prob": 1.0 if not text else float(doubt),
                        **({"filtered": raw} if raw != text else {}),
                        "language": eng.language, "model": cfg["model"]})
        return out

    def _deliver(self, item, res):
        self.emit({"type": "transcript", "kind": item["kind"], "id": item["id"],
                   "chain": item["chain"], "wav": item.get("wav"), "members": item.get("members"),
                   "keys": item.get("keys", []), **res})
        key = "transcript" if item["kind"] == "track" else "chain_transcript"
        if self.on_result:
            self.on_result(item.get("keys", []), key, res)
        for p in item.get("meta_paths", []):
            try:
                m = json.loads(p.read_text())
                m[key] = res
                p.write_text(json.dumps(m, indent=1, default=float))
            except Exception:
                pass

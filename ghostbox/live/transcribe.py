"""Local Whisper transcription (faster-whisper) for finished detections and hop chains.

Runs in its own thread so the detection engine never waits on it. Results are
emitted to clients and written into the recordings' JSON metadata.

Whisper hallucinates on noise (it will happily "hear" phrases in static), so every
result carries Whisper's own no_speech_prob and avg_logprob for the UI to show.
"""
import json
import queue
import threading
import traceback

import numpy as np


class Transcriber:
    def __init__(self, config, emit):
        self.config = config
        self.emit = emit
        self.q = queue.Queue(maxsize=50)
        self.model, self.model_name = None, None
        self.error = ""
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, item):
        """item: {kind: track|chain, id, chain, audio (float32 16 kHz), meta_paths: [...], wav}"""
        if len(item["audio"]) < 16000 * 0.3:
            return
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass

    def _load(self, name):
        if self.model_name != name:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)
            self.model_name = name

    def _run(self):
        while True:
            item = self.q.get()
            cfg = self.config.get()["whisper"]
            if not cfg["enabled"]:
                continue
            try:
                self._load(cfg["model"])
                a = item["audio"].astype(np.float32)
                a = a / (np.abs(a).max() + 1e-9) * 0.9
                lang = None if cfg["language"] in ("", "auto") else cfg["language"]
                segs, info = self.model.transcribe(a, language=lang, beam_size=1,
                                                   condition_on_previous_text=False, vad_filter=False)
                segs = list(segs)
                text = " ".join(s.text.strip() for s in segs).strip()
                res = {
                    "text": text,
                    "no_speech_prob": float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0,
                    "avg_logprob": float(np.mean([s.avg_logprob for s in segs])) if segs else None,
                    "language": info.language, "language_prob": float(info.language_probability),
                    "model": cfg["model"],
                }
                self.emit({"type": "transcript", "kind": item["kind"], "id": item["id"],
                           "chain": item["chain"], "wav": item.get("wav"), "members": item.get("members"),
                           **res})
                key = "transcript" if item["kind"] == "track" else "chain_transcript"
                for p in item.get("meta_paths", []):
                    try:
                        m = json.loads(p.read_text())
                        m[key] = res
                        p.write_text(json.dumps(m, indent=1, default=float))
                    except Exception:
                        pass
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()

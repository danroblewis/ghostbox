"""Local speech-recognition engines behind one interface.

Each engine takes a list of 16 kHz float32 clips and returns, per clip, (text, no_speech) where
no_speech in [0, 1] is the engine's own doubt (1 = probably not speech), used to dim results.

- whisper:  faster-whisper, batched; the best accuracy, but it invents fluent text from noise
- moonshine: Useful Sensors' Moonshine (transformers); English; sized to the clip, fast on short ones
- wav2vec2: CTC character model (transformers); English; no language model, so it can't invent
            whole sentences: on noise it gives short fragments or nothing
- vosk:     Kaldi small English model; very light; fixed vocabulary with per-word confidence

Models download on first use and stay loaded.
"""
import json
import threading

import numpy as np

FS = 16000

# model id -> (engine, what to load, label)
MODELS = {
    "tiny": ("whisper", "tiny", "Whisper tiny"),
    "base": ("whisper", "base", "Whisper base"),
    "small": ("whisper", "small", "Whisper small"),
    "medium": ("whisper", "medium", "Whisper medium"),
    "large-v3-turbo": ("whisper", "large-v3-turbo", "Whisper large-v3 turbo"),
    "distil-large-v3": ("whisper", "distil-large-v3", "Whisper distil-large-v3 (English)"),
    "moonshine-tiny": ("moonshine", "UsefulSensors/moonshine-tiny", "Moonshine tiny (English)"),
    "moonshine-base": ("moonshine", "UsefulSensors/moonshine-base", "Moonshine base (English)"),
    "wav2vec2": ("wav2vec2", "facebook/wav2vec2-base-960h", "wav2vec2 base (English, no language model)"),
    "vosk-small-en": ("vosk", "vosk-model-small-en-us-0.15", "Vosk small (English)"),
}


def _norm(a):
    a = a.astype(np.float32)
    return a / (np.abs(a).max() + 1e-9) * 0.9


class WhisperEngine:
    GAP = np.zeros(int(0.5 * FS), np.float32)

    def __init__(self, name, cpu_threads=8, workers=2):
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        self.model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=cpu_threads,
                                  num_workers=max(1, int(workers)))
        self.pipe = BatchedInferencePipeline(model=self.model)
        self.language = None

    def transcribe(self, clips, language=None):
        parts, spans, t = [], [], 0.0
        for a in clips:                                     # each clip in its own 30 s window
            a = _norm(a)
            spans.append((t, t + len(a) / FS))
            parts += [a, self.GAP]
            t += len(a) / FS + len(self.GAP) / FS
        segs, info = self.pipe.transcribe(np.concatenate(parts), batch_size=max(1, len(clips)), beam_size=1,
                                          language=language, without_timestamps=True,
                                          clip_timestamps=[{"start": a, "end": b} for a, b in spans])
        texts, nsp = [[] for _ in clips], [[] for _ in clips]
        for s in segs:
            i = min(range(len(spans)), key=lambda k: abs(spans[k][0] - s.start))
            if s.text.strip():
                texts[i].append(s.text.strip())
            if getattr(s, "no_speech_prob", None) is not None:
                nsp[i].append(s.no_speech_prob)
        self.language = info.language
        return [(" ".join(texts[i]).strip(), float(np.mean(nsp[i])) if nsp[i] else (0.5 if texts[i] else 1.0))
                for i in range(len(clips))]


class MoonshineEngine:
    def __init__(self, name, cpu_threads=8, **_):
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        torch.set_num_threads(cpu_threads)
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = MoonshineForConditionalGeneration.from_pretrained(name).eval()
        self.model.generation_config.max_length = None      # we set max_new_tokens per clip
        self.language = "en"

    def transcribe(self, clips, language=None):
        out = []
        for a in clips:                                     # sized to each clip: no 30 s padding
            inp = self.proc(_norm(a), sampling_rate=FS, return_tensors="pt")
            # ~6.5 tokens per second of audio is plenty; capping it stops runaway repetition on noise
            max_new = int(np.ceil(len(a) / FS * 6.5)) + 2
            with self.torch.no_grad():
                g = self.model.generate(**inp, max_new_tokens=max_new, output_scores=True,
                                        return_dict_in_generate=True)
            text = self.proc.batch_decode(g.sequences, skip_special_tokens=True)[0].strip()
            # doubt = 1 - mean probability of the chosen tokens
            if g.scores:
                pr = [float(self.torch.softmax(s[0], -1).max()) for s in g.scores]
                doubt = 1.0 - float(np.mean(pr))
            else:
                doubt = 1.0
            out.append((text, doubt if text else 1.0))
        return out


class Wav2Vec2Engine:
    def __init__(self, name, cpu_threads=8, **_):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        torch.set_num_threads(cpu_threads)
        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).eval()
        self.language = "en"

    def transcribe(self, clips, language=None):
        inp = self.proc([_norm(a) for a in clips], sampling_rate=FS, return_tensors="pt", padding=True)
        with self.torch.no_grad():
            logits = self.model(inp.input_values, attention_mask=inp.get("attention_mask")).logits
        prob = self.torch.softmax(logits, -1)
        ids = prob.argmax(-1)
        texts = self.proc.batch_decode(ids)
        blank = self.proc.tokenizer.pad_token_id
        out = []
        for i, a in enumerate(clips):
            n = min(ids.shape[1], int(np.ceil(len(a) / FS * 50)))   # 50 frames per second
            keep = ids[i, :n] != blank
            conf = float(prob[i, :n].max(-1).values[keep].mean()) if keep.any() else 0.0
            text = texts[i].strip().capitalize()
            out.append((text, 1.0 - conf if text else 1.0))
        return out


class VoskEngine:
    def __init__(self, name, **_):
        import vosk
        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(model_name=name)
        self.language = "en"

    def transcribe(self, clips, language=None):
        out = []
        for a in clips:
            rec = self.vosk.KaldiRecognizer(self.model, FS)
            rec.SetWords(True)
            pcm = (np.clip(_norm(a), -1, 1) * 32767).astype("<i2").tobytes()
            rec.AcceptWaveform(pcm)
            r = json.loads(rec.FinalResult())
            words = r.get("result", [])
            text = r.get("text", "").strip().capitalize()
            conf = float(np.mean([w["conf"] for w in words])) if words else 0.0
            out.append((text, 1.0 - conf if text else 1.0))
        return out


ENGINES = {"whisper": WhisperEngine, "moonshine": MoonshineEngine, "wav2vec2": Wav2Vec2Engine, "vosk": VoskEngine}
_lock = threading.Lock()


def load(model_id, cpu_threads=8, workers=2):
    engine, name, _ = MODELS.get(model_id, MODELS["base"])
    with _lock:
        return ENGINES[engine](name, cpu_threads=cpu_threads, workers=workers)

"""Web server for the live ghost box.

    python3 -m ghostbox.live.server --port 8765

Open the printed URL once (it carries the access key and sets a cookie). The page
controls the radio, so every route requires the key.
"""
import argparse
import asyncio
import json
import re
import secrets
import time
from pathlib import Path

from aiohttp import WSMsgType, web
import numpy as np

from .config import CHECKS, PRESETS, SENSITIVITY, VIEWS, Config
from .autotune import AutoTune, pick_segments
from .engine import Engine, iq_cut_wav, render_segments
from .transcribe import strip_stock
from .asr import MODELS as ASR_MODELS
from .survey import Survey

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_token(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    tok = secrets.token_urlsafe(18)
    path.write_text(tok)
    return tok


class Hub:
    """Fans engine events out to clients. Waterfall rows go only to clients watching that view."""

    def __init__(self, loop):
        self.loop = loop
        self.clients = {}            # queue -> waterfall view mode

    def emit(self, msg):  # called from the engine thread
        mode = msg.get("mode") if msg.get("type") == "wf" else None
        data = json.dumps(msg, default=float)
        self.loop.call_soon_threadsafe(self._fanout, data, mode)

    def _fanout(self, data, mode=None):
        for q, m in list(self.clients.items()):
            if mode is not None and m != mode:
                continue
            if q.qsize() > 600:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(data)


def make_app(args):
    token = load_token(ROOT / "live_data" / "token")
    config = Config(ROOT / "live_data" / "config.json")
    app = web.Application(middlewares=[])
    state = {}

    @web.middleware
    async def auth(request, handler):
        k = request.query.get("k")
        if k == token:
            resp = web.HTTPFound("/") if request.path == "/" else None
            if resp:
                resp.set_cookie("gbk", token, max_age=30 * 86400, httponly=True, samesite="Lax", secure=False)
                raise resp
        elif request.cookies.get("gbk") != token:
            return web.Response(status=401, text="ghostbox: open the URL with ?k=<key> printed by the server")
        return await handler(request)

    app.middlewares.append(auth)

    async def on_startup(app):
        hub = Hub(asyncio.get_running_loop())
        state["hub"] = hub
        state["engine"] = Engine(config, hub.emit, ROOT)
        state["tune"] = AutoTune(ROOT, config, hub.emit, apply, session=lambda: state["engine"].session())
        state["survey"] = Survey(ROOT, config, apply, lambda: state["engine"], hub.emit)
        state["auto_task"] = asyncio.create_task(auto_tune_loop())

    def rx_signature(c):
        """Receiver settings: any change (frequency, span, gain, amp, range, device) re-arms auto-tune."""
        return json.dumps({k: v for k, v in c["sdr"].items() if k != "file"}, sort_keys=True)

    async def auto_tune_loop():
        """With automatic recalibration on: calibrate after every receiver change once `minutes` of IQ
        has been recorded since it, and recalibrate whenever nothing has been caught for `minutes`
        (including after a run that found too little: it simply tries again later)."""
        while True:
            await asyncio.sleep(10)
            try:
                c = config.get()
                at, tune, eng = c["autotune"], state["tune"], state["engine"]
                sv = state["survey"]
                if sv.running:
                    continue
                if not at["auto"] or c["sdr"]["source"] == "file" or tune.state.get("state") == "running":
                    continue
                if not c["iq_ring"]["enabled"]:                # auto-tune needs the IQ recording
                    apply({"iq_ring": {"enabled": True}})
                    continue
                eng.show_conf = tune.show_conf if tune.show_conf is not None else c["detect"]["threshold"]
                changed = tune.auto_band != rx_signature(c)
                due = time.time() - max(eng.last_hit, tune.last_end, sv.ended_at) >= at["minutes"] * 60
                if not (changed or due):
                    continue
                _, have = await asyncio.get_running_loop().run_in_executor(
                    None, pick_segments, ROOT, c, at["minutes"], eng.session())
                if have >= at["minutes"] * 60 * 0.95:
                    tune.start(at["minutes"], at["per_min"], auto=True)
            except Exception:
                pass

    async def on_cleanup(app):
        if state.get("auto_task"):
            state["auto_task"].cancel()
        state["engine"].stop()

    async def no_cache(request, response):
        # the page changes often while we iterate; don't let browsers run stale JS/CSS
        if request.path.startswith("/static/") or request.path == "/":
            response.headers["Cache-Control"] = "no-cache"

    app.on_response_prepare.append(no_cache)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def index(request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=20, max_msg_size=8 << 20)
        await sock.prepare(request)
        q = asyncio.Queue()
        hub = state["hub"]
        eng = state["engine"]
        await sock.send_str(json.dumps({
            "type": "hello", "config": config.get(), "checks": CHECKS, "views": VIEWS,
            "asr_models": [[k, v[2]] for k, v in ASR_MODELS.items()],
            "root": str(ROOT.resolve()),
            "presets": {k: v["label"] for k, v in PRESETS.items()},
            "sensitivity": SENSITIVITY, "epoch": eng.t_origin,
            "status": eng.status, "detections": eng.snapshot(),
            "reset": {"fc": getattr(eng, "fc", 0), "fs": getattr(eng, "fs", 0),
                      "fmin": getattr(eng, "f_lo", 0), "fmax": getattr(eng, "f_hi", 0),
                      "frame_rate": getattr(eng, "fr", 25)},
        }, default=float))
        hub.clients[q] = "power"

        async def writer():
            while True:
                data = await q.get()
                await sock.send_str(data)

        wt = asyncio.create_task(writer())
        try:
            async for msg in sock:
                if msg.type == WSMsgType.ERROR:
                    break
                if msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except ValueError:
                        continue
                    if m.get("type") == "view" and m.get("mode") in {v["id"] for v in VIEWS}:
                        hub.clients[q] = m["mode"]
        finally:
            hub.clients.pop(q, None)
            wt.cancel()
        return sock

    def apply(patch):
        cfg, sdr_changed = config.update(patch)
        if sdr_changed:
            state["engine"].request_restart()
        state["hub"].emit({"type": "config", "config": cfg})
        return cfg

    async def history(request):
        view = request.query.get("view", "power")
        t0, t1 = float(request.query.get("t0", 0)), float(request.query.get("t1", 1e12))
        chunks, boxes = await asyncio.get_running_loop().run_in_executor(
            None, state["engine"].history, view, t0, t1)
        return web.json_response({"chunks": chunks, "boxes": boxes}, dumps=lambda o: json.dumps(o, default=float))

    async def get_config(request):
        return web.json_response(config.get())

    async def post_config(request):
        return web.json_response(apply(await request.json()))

    PRESET_DIR = ROOT / "live_data" / "presets"
    PRESET_KEYS = ("sdr", "analysis", "detect", "checks", "audio")   # a preset = receiver + detector + playback

    def saved_presets():
        out = []
        for p in sorted(PRESET_DIR.glob("*.json"), reverse=True) if PRESET_DIR.exists() else []:
            try:
                m = json.loads(p.read_text())
                out.append({"file": p.name, "label": m.get("label", p.stem), "saved": m.get("saved")})
            except Exception:
                pass
        return out

    async def get_presets(request):
        return web.json_response({"builtin": {k: v["label"] for k, v in PRESETS.items()}, "saved": saved_presets()})

    async def save_preset(request):
        """Save the current receiver + detector + playback settings. File name: date, device, range, name."""
        name = re.sub(r"[^A-Za-z0-9 _.-]+", "", str((await request.json()).get("name", "")).strip())[:60] or "preset"
        c = config.get()
        sdr = c["sdr"]
        lo = sdr["fmin_hz"] if sdr.get("fmin_hz") is not None else sdr["center_hz"] - sdr["sample_rate"] / 2
        hi = sdr["fmax_hz"] if sdr.get("fmax_hz") is not None else sdr["center_hz"] + sdr["sample_rate"] / 2
        stamp = time.strftime("%Y-%m-%d")
        slug = re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-").lower() or "preset"
        fname = f"{stamp}_{sdr['source']}_{max(0, lo) / 1e6:.3f}-{hi / 1e6:.3f}MHz_{slug}.json"
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
        body = {"label": f"{name} ({sdr['source']}, {max(0, lo) / 1e6:.3f}-{hi / 1e6:.3f} MHz, {stamp})",
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: c[k] for k in PRESET_KEYS}}
        (PRESET_DIR / fname).write_text(json.dumps(body, indent=1))
        return web.json_response({"file": fname, "presets": saved_presets()})

    async def delete_preset(request):
        f = request.match_info["file"]
        if not SAFE.match(f) or not f.endswith(".json"):
            raise web.HTTPNotFound()
        (PRESET_DIR / f).unlink(missing_ok=True)
        return web.json_response({"presets": saved_presets()})

    async def get_survey(request):
        """Current/last survey (?file= for a saved one); ?list=1 for the saved survey names."""
        sv = state["survey"]
        if request.query.get("list"):
            return web.json_response({"saved": sv.saved(), "state": sv.state})
        f = request.query.get("file")
        if f:
            if not SAFE.match(f):
                raise web.HTTPNotFound()
            res = await asyncio.get_running_loop().run_in_executor(None, sv.load, f)
        else:
            res = await asyncio.get_running_loop().run_in_executor(None, sv.meta)
        return web.json_response({"state": sv.state, "result": res}, dumps=lambda o: json.dumps(o, default=float))

    async def survey_image(request):
        """One view of a survey as raw uint8 [rows, columns] (meta says the shape)."""
        f, view = request.query.get("file") or None, request.query.get("view", "power")
        if (f and not SAFE.match(f)) or not SAFE.match(view):
            raise web.HTTPNotFound()
        img = await asyncio.get_running_loop().run_in_executor(None, state["survey"].image, view, f)
        if img is None:
            raise web.HTTPNotFound(text="view not recorded in this survey")
        return web.Response(body=np.ascontiguousarray(img, np.uint8).tobytes(), content_type="application/octet-stream",
                            headers={"X-Rows": str(img.shape[0]), "X-Cols": str(img.shape[1])})

    async def get_devices(request):
        from .sources import list_devices
        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, list_devices))

    async def sdr_test(request):
        """Is the receiver delivering samples, and at what level? The running device is checked
        without opening it again (it's busy); any other device is opened for half a second."""
        from .sources import level_report, test_device
        b = await request.json()
        c = config.get()
        sdr = {**c["sdr"], **{k: v for k, v in b.items() if k in ("source", "device")}}
        eng = state["engine"]
        live = sdr["source"] == c["sdr"]["source"] and str(sdr.get("device") or "") == str(c["sdr"].get("device") or "")
        if live and c["sdr"]["source"] != "file":
            age = time.time() - getattr(eng, "in_seen", 0)
            if age > 3:
                err = (eng.status or {}).get("error") if isinstance(getattr(eng, "status", None), dict) else None
                return web.json_response({"ok": False, "live": True, "message": "The receiver isn't delivering samples"
                                          + (f": {err}" if err else " (is it plugged in?)")})
            rep = level_report(np.asarray(eng.in_last))
            return web.json_response({"ok": True, "live": True, **rep,
                                      "message": f"Receiving {sdr['sample_rate'] / 1e6:g} MS/s now"})
        res = await asyncio.get_running_loop().run_in_executor(None, test_device, sdr)
        return web.json_response(res)

    async def get_detection(request):
        meta = await asyncio.get_running_loop().run_in_executor(None, state["engine"].detail, request.query.get("key", ""))
        if meta is None:
            raise web.HTTPNotFound()
        meta = {k: v for k, v in meta.items() if k not in ("checks", "detect")}
        return web.json_response(meta, dumps=lambda o: json.dumps(o, default=float))

    def survey_segments(f, track_id):
        """(segments, timeline base, step, track) for a survey sample's IQ in the ring."""
        st, t = state["survey"].track(f, track_id)
        if t is None or "iq_t" not in t:
            raise web.HTTPNotFound(text="no IQ recorded for this sample")
        ring = ROOT / config.get()["iq_ring"]["dir"]
        segs = []
        for name in st.get("iq_files", []):
            p = ring / name
            if p.exists() and p.with_suffix(".json").exists():
                segs.append((json.loads(p.with_suffix(".json").read_text()), p))
        if not segs:
            raise web.HTTPNotFound(text="this sample's IQ is no longer in the IQ ring")
        return segs, segs[0][0]["t0"], st, t

    async def clip_source(q):
        """(segments, t0, t1, path points, mode) of a detection (?key=) or survey sample (?survey=&track=)."""
        au, eng, loop = config.get()["audio"], state["engine"], asyncio.get_running_loop()
        pre, post = au.get("preroll_s", 0.5), au.get("postroll_s", 0.5)
        if q.get("survey"):
            if not SAFE.match(q["survey"]):
                raise web.HTTPNotFound()
            segs, base, st, t = survey_segments(q["survey"], int(q.get("track", -1)))
            t0, t1 = base + t["iq_t"][0] - pre, base + t["iq_t"][1] + post
            return segs, t0, t1, [[base + r, f] for r, f in t.get("path") or []] or [[t0, t["f"]]], t.get("mode", 0)
        meta = await loop.run_in_executor(None, eng.detail, q.get("key", ""))
        if meta is None:
            raise web.HTTPNotFound()
        t0, t1 = meta["stream_t0"] - pre, meta["stream_t1"] + post
        segs = await loop.run_in_executor(None, eng.ring_segments, meta["epoch"], t0, t1)
        if not segs:
            raise web.HTTPNotFound(text="this detection's IQ is no longer in the IQ ring")
        return segs, t0, t1, [[meta["stream_t0"] + r, f] for r, f in meta["path"]] or [[t0, meta["freq_hz"]["mean"]]], meta.get("mode", 0)

    async def tune_audio(request):
        """Search demodulation + filtering for this one detection (see audiotune)."""
        from .audiotune import as_audio_settings, tune
        from .engine import load_iq
        q = dict(await request.json())
        segs, t0, t1, pts, mode = await clip_source(q)
        loop = asyncio.get_running_loop()

        def job():
            loaded = load_iq(segs, t0, t1)
            if loaded is None:
                return None
            return tune(loaded, pts, config.get()["audio"], mode)
        res = await loop.run_in_executor(None, job)
        if res is None:
            raise web.HTTPNotFound(text="no IQ for this detection")
        res["apply"] = as_audio_settings(res["best"])
        return web.json_response(res, dumps=lambda o: json.dumps(o, default=float))

    async def render(request):
        """A detection / survey sample re-rendered from the IQ ring with other audio settings, for A/B:
        ?demod=envelope|sync|sync_sb|auto &denoise=0|1 &band=0|1 (anything left out: as configured)."""
        q, eng, loop = request.query, state["engine"], asyncio.get_running_loop()
        au = dict(config.get()["audio"])
        if q.get("demod") in ("auto", "envelope", "sync", "sync_sb", "suppressed"):
            au["demod"] = q["demod"]
        for k, key in (("denoise", "denoise"), ("band", "voice_band")):
            if q.get(k) in ("0", "1"):
                au[key] = q[k] == "1"
        for k, key in (("lo", "voice_lo_hz"), ("hi", "voice_hi_hz"), ("strength", "denoise_strength"), ("floor", "denoise_floor_db")):
            if q.get(k):
                au[key] = float(q[k])
        offset = float(q.get("offset") or 0.0)
        pre, post = au.get("preroll_s", 0.5), au.get("postroll_s", 0.5)
        if q.get("survey"):
            if not SAFE.match(q["survey"]):
                raise web.HTTPNotFound()
            segs, base, st, t = survey_segments(q["survey"], int(q.get("track", -1)))
            t0, t1 = base + t["iq_t"][0] - pre, base + t["iq_t"][1] + post
            pts = [[base + r, f] for r, f in t.get("path") or []] or [[t0, t["f"]]]
            mode = t.get("mode", 0)
        else:
            meta = await loop.run_in_executor(None, eng.detail, q.get("key", ""))
            if meta is None:
                raise web.HTTPNotFound()
            t0, t1 = meta["stream_t0"] - pre, meta["stream_t1"] + post
            segs = await loop.run_in_executor(None, eng.ring_segments, meta["epoch"], t0, t1)
            if not segs:
                raise web.HTTPNotFound(text="this detection's IQ is no longer in the IQ ring")
            pts = [[meta["stream_t0"] + r, f] for r, f in meta["path"]] or [[t0, meta["freq_hz"]["mean"]]]
            mode = meta.get("mode", 0)
        if q.get("mode") in ("-1", "0", "1"):
            mode = int(q["mode"])
        pcm = await loop.run_in_executor(None, render_segments, segs, t0, t1, pts, au, mode, offset)
        if pcm is None:
            raise web.HTTPNotFound(text="no IQ in that time range")
        import io
        from scipy.io import wavfile
        buf = io.BytesIO()
        wavfile.write(buf, 16000, pcm)
        return web.Response(body=buf.getvalue(), content_type="audio/wav")

    async def get_iq(request):
        """IQ around a detection (?key=) or a survey sample (?survey=&track=), as a stereo I/Q WAV."""
        q, eng = request.query, state["engine"]
        pad = min(5.0, max(0.0, float(q.get("pad", 1.0))))
        rate = int(q.get("rate", 48000))
        loop = asyncio.get_running_loop()
        if q.get("survey"):
            f = q["survey"]
            if not SAFE.match(f):
                raise web.HTTPNotFound()
            segs, base, st, t = survey_segments(f, int(q.get("track", -1)))
            t0, t1, fcen, utc = base + t["iq_t"][0] - pad, base + t["iq_t"][1] + pad, t["f"], None
        else:
            meta = await loop.run_in_executor(None, eng.detail, q.get("key", ""))
            if meta is None:
                raise web.HTTPNotFound()
            t0, t1 = meta["stream_t0"] - pad, meta["stream_t1"] + pad
            fcen, utc = meta["freq_hz"]["mean"], meta.get("utc_start")
            segs = await loop.run_in_executor(None, eng.ring_segments, meta["epoch"], t0, t1)
            if not segs:
                raise web.HTTPNotFound(text="this detection's IQ is no longer in the IQ ring")
        data, _ = await loop.run_in_executor(None, iq_cut_wav, segs, t0, t1, fcen, rate)
        if data is None:
            raise web.HTTPNotFound(text="no IQ in that time range")
        stamp = (utc or "")[:19].replace("-", "").replace(":", "").replace("T", "-")
        name = f"iq_{fcen / 1e6:.4f}MHz_{stamp or 'survey'}_{rate // 1000}k.wav"
        return web.Response(body=data, content_type="audio/wav",
                            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    async def survey_preview(request):
        img = state["survey"].preview_image(int(request.query.get("i", -1)))
        if img is None:
            raise web.HTTPNotFound()
        return web.Response(body=np.ascontiguousarray(img).tobytes(), content_type="application/octet-stream",
                            headers={"X-Rows": str(img.shape[0]), "X-Cols": str(img.shape[1])})

    async def survey_clip(request):
        f, c = request.query.get("file", ""), request.query.get("clip", "")
        if not (SAFE.match(f) and SAFE.match(c)):
            raise web.HTTPNotFound()
        p = state["survey"].clip_path(f, c)
        if p is None:
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Content-Type": "audio/wav"})

    async def start_survey(request):
        b = await request.json()
        c = config.get()
        if c["sdr"]["source"] == "file":
            return web.json_response({"error": "A survey needs a live receiver."}, status=400)
        if state["tune"].state.get("state") == "running":
            return web.json_response({"error": "Calibration is running; try again when it finishes."}, status=409)
        lo, hi = float(b["lo_hz"]), float(b["hi_hz"])
        if not (hi > lo >= 0):
            return web.json_response({"error": "Give a range with From below To."}, status=400)
        ok = state["survey"].start(lo, hi, float(b.get("dwell_s", 30)))
        return web.json_response({"ok": ok, **state["survey"].state})

    async def stop_survey(request):
        state["survey"].stop()
        return web.json_response({"ok": True})

    async def post_preset(request):
        """Apply a built-in preset ({name}) or a saved one ({file})."""
        b = await request.json()
        if b.get("file"):
            f = b["file"]
            if not SAFE.match(f) or not (PRESET_DIR / f).is_file():
                return web.json_response({"error": "unknown preset"}, status=400)
            m = json.loads((PRESET_DIR / f).read_text())
            return web.json_response(apply({k: m[k] for k in PRESET_KEYS if k in m}))
        name = b.get("name")
        if name not in PRESETS:
            return web.json_response({"error": "unknown preset"}, status=400)
        patch = {k: v for k, v in PRESETS[name].items() if k != "label"}
        return web.json_response(apply(patch))

    async def post_reset(request):
        cfg, _ = config.reset()
        state["engine"].request_restart()
        state["hub"].emit({"type": "config", "config": cfg})
        return web.json_response(cfg)

    async def post_restart(request):
        state["engine"].request_restart()
        return web.json_response({"ok": True})

    def rec_dir():
        return ROOT / config.get()["record"]["dir"]

    meta_cache = {}   # path -> (mtime, slim meta); recordings are re-read only when they change

    def slim_meta(p):
        mt = p.stat().st_mtime
        hit = meta_cache.get(p)
        if hit and hit[0] == mt:
            return hit[1]
        m = json.loads(p.read_text())
        m["confidence"].pop("history", None)
        for k in ("checks", "detect", "path", "sdr"):
            m.pop(k, None)
        m.setdefault("saved", True)
        m.setdefault("key", "wav:" + m["files"]["wav"])
        m.setdefault("clip", "/rec/" + m["files"]["wav"])
        meta_cache[p] = (mt, m)
        return m

    def list_recordings(offset, limit, q, min_conf=0.0):
        """Detections newest first: in-memory ones (not saved) merged with saved ones on disk."""
        d = rec_dir()
        paths = sorted(d.glob("*.json"), reverse=True) if d.exists() else []
        live = set(paths)
        for k in [k for k in meta_cache if k not in live]:
            meta_cache.pop(k, None)
        metas = []
        for p in paths:
            try:
                metas.append(slim_meta(p))
            except Exception:
                continue
        for m in state["engine"].mem_list():
            m = {k: v for k, v in m.items() if k not in ("checks", "detect", "path", "sdr")}
            m["confidence"] = {k: v for k, v in m["confidence"].items() if k != "history"}
            metas.append(m)
        metas.sort(key=lambda m: m.get("utc_start", ""), reverse=True)
        phrases = config.get()["whisper"].get("filter_phrases", [])
        for m in metas:                                    # re-filter from the original text, so edits to
            for f in ("transcript", "chain_transcript"):   # the phrase list apply to saved ones too
                t = m.get(f)
                src = t and (t.get("filtered") or t.get("text"))
                if src:
                    clean = strip_stock(src, phrases)
                    if clean != t.get("text") or (clean == src and "filtered" in t):
                        m[f] = {k: v for k, v in t.items() if k != "filtered"}
                        m[f]["text"] = clean
                        if clean != src:
                            m[f]["filtered"] = src
                        if not clean:
                            m[f]["no_speech_prob"] = 1.0
        q = q.strip().lower()
        out, matched = [], 0
        for m in metas:
            if m["confidence"]["max"] < min_conf:
                continue
            if q:
                hay = " ".join([(m.get("transcript") or {}).get("text", ""),
                                (m.get("chain_transcript") or {}).get("text", ""),
                                f"{m['freq_hz']['mean'] / 1e6:.4f}", f"{m['freq_hz']['mean'] / 1e3:.1f}"]).lower()
                if q not in hay:
                    continue
            if offset <= matched < offset + limit:
                out.append(m)
            matched += 1
        return {"total": matched, "items": out}

    async def recordings(request):
        """Saved detections, newest first, paginated, optional search over transcripts/frequency."""
        offset = max(0, int(request.query.get("offset", 0)))
        limit = max(1, min(500, int(request.query.get("limit", 100))))
        q = request.query.get("q", "")
        min_conf = float(request.query.get("min_conf", 0))
        res = await asyncio.get_running_loop().run_in_executor(None, list_recordings, offset, limit, q, min_conf)
        return web.json_response(res, dumps=lambda o: json.dumps(o, default=float))

    async def clip(request):
        key = request.query.get("key", "")
        data = await asyncio.get_running_loop().run_in_executor(None, state["engine"].clip, key)
        if data is None:
            raise web.HTTPNotFound(text="audio no longer available (dropped from memory and not in the IQ ring)")
        return web.Response(body=data, content_type="audio/wav")

    async def save(request):
        key = (await request.json()).get("key", "")
        meta = await asyncio.get_running_loop().run_in_executor(None, state["engine"].save, key)
        if meta is None:
            return web.json_response({"error": "not available"}, status=404)
        return web.json_response(meta, dumps=lambda o: json.dumps(o, default=float))

    async def tune_start(request):
        b = await request.json()
        if state["survey"].running:
            return web.json_response({"error": "A survey is running; calibrate when it finishes."}, status=409)
        at = config.get()["autotune"]
        ok = state["tune"].start(b.get("minutes", at["minutes"]), b.get("per_min", at["per_min"]))
        return web.json_response({"ok": ok, **state["tune"].state})

    async def tune_get(request):
        return web.json_response(state["tune"].state, dumps=lambda o: json.dumps(o, default=float))

    async def tune_ready(request):
        """How much continuous raw IQ exists for the current band (the same rule auto-tune uses)."""
        minutes = float(request.query.get("minutes", 2))
        c = config.get()
        need = minutes * 60
        if c["sdr"]["source"] == "file":
            return web.json_response({"ready": False, "reason": "file", "have_s": 0, "need_s": need})
        _, have = await asyncio.get_running_loop().run_in_executor(
            None, pick_segments, ROOT, c, minutes, state["engine"].session())
        reason = None if have >= need * 0.95 else ("off" if not c["iq_ring"]["enabled"] else "recording")
        return web.json_response({"ready": reason is None, "reason": reason, "have_s": round(have, 1), "need_s": need,
                                  "segment_s": c["iq_ring"]["segment_s"], "auto": c["autotune"]["auto"],
                                  "tuned": state["tune"].auto_band == rx_signature(c),
                                  "recal_in_s": max(0, round(c["autotune"]["minutes"] * 60 - (time.time() - max(
                                      state["engine"].last_hit, state["tune"].last_end))))})

    async def tune_revert(request):
        return web.json_response({"ok": state["tune"].revert()})

    async def rec_file(request):
        name = request.match_info["name"]
        if not SAFE.match(name) or Path(name).suffix not in (".wav", ".json", ".npy"):
            raise web.HTTPNotFound()
        p = rec_dir() / name
        if not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def storage(request):
        import shutil

        def du(path):
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
        c = config.get()
        disk = shutil.disk_usage(ROOT)
        return web.json_response({
            "iq_ring": du(ROOT / c["iq_ring"]["dir"]), "recordings": du(ROOT / c["record"]["dir"]),
            "captures": du(ROOT / "captures"), "free": disk.free, "total": disk.total,
            "paused": state["engine"].storage.get("paused", False)})

    async def capture_files(request):
        out = []
        for sub in ("captures", config.get()["iq_ring"]["dir"]):
            d = ROOT / sub
            if d.exists():
                out += [f"{sub}/{p.name}" for p in sorted(d.glob("*"), reverse=sub != "captures")
                        if p.suffix in (".cu8", ".cs8")]
        return web.json_response(out)

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws)
    app.router.add_get("/api/history", history)
    app.router.add_get("/api/detections", recordings)
    app.router.add_get("/api/clip", clip)
    app.router.add_post("/api/save", save)
    app.router.add_post("/api/autotune", tune_start)
    app.router.add_get("/api/autotune", tune_get)
    app.router.add_post("/api/autotune/revert", tune_revert)
    app.router.add_get("/api/autotune/ready", tune_ready)
    app.router.add_get("/api/config", get_config)
    app.router.add_post("/api/config", post_config)
    app.router.add_post("/api/preset", post_preset)
    app.router.add_get("/api/survey", get_survey)
    app.router.add_post("/api/survey", start_survey)
    app.router.add_post("/api/survey/stop", stop_survey)
    app.router.add_get("/api/survey/image", survey_image)
    app.router.add_get("/api/survey/preview", survey_preview)
    app.router.add_get("/api/survey/clip", survey_clip)
    app.router.add_get("/api/detection", get_detection)
    app.router.add_get("/api/devices", get_devices)
    app.router.add_post("/api/sdr/test", sdr_test)
    app.router.add_get("/api/iq", get_iq)
    app.router.add_get("/api/render", render)
    app.router.add_post("/api/tune_audio", tune_audio)
    app.router.add_get("/api/presets", get_presets)
    app.router.add_post("/api/presets", save_preset)
    app.router.add_delete("/api/presets/{file}", delete_preset)
    app.router.add_post("/api/reset", post_reset)
    app.router.add_post("/api/restart", post_restart)
    app.router.add_get("/api/recordings", recordings)
    app.router.add_get("/api/captures", capture_files)
    app.router.add_get("/api/storage", storage)
    app.router.add_get("/rec/{name}", rec_file)
    app.router.add_static("/static", STATIC)
    return app, token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app, token = make_app(args)
    print(f"ghostbox live: http://{args.host}:{args.port}/?k={token}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()

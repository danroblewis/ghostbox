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

from .config import CHECKS, PRESETS, SENSITIVITY, VIEWS, Config
from .autotune import AutoTune, pick_segments
from .engine import Engine
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
            res = await asyncio.get_running_loop().run_in_executor(None, sv.public)
        return web.json_response({"state": sv.state, "result": res}, dumps=lambda o: json.dumps(o, default=float))

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

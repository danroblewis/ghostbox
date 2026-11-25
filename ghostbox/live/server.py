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
from pathlib import Path

from aiohttp import WSMsgType, web

from .config import CHECKS, PRESETS, Config
from .engine import Engine

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
    def __init__(self, loop):
        self.loop = loop
        self.clients = set()

    def emit(self, msg):  # called from the engine thread
        data = json.dumps(msg, default=float)
        self.loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data):
        for q in list(self.clients):
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

    async def on_cleanup(app):
        state["engine"].stop()

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
            "type": "hello", "config": config.get(), "checks": CHECKS,
            "presets": {k: v["label"] for k, v in PRESETS.items()},
            "status": eng.status, "detections": eng.snapshot(),
            "reset": {"fc": getattr(eng, "fc", 0), "fs": getattr(eng, "fs", 0),
                      "fmin": getattr(eng, "f_lo", 0), "fmax": getattr(eng, "f_hi", 0),
                      "frame_rate": getattr(eng, "fr", 25)},
        }, default=float))
        for m in list(getattr(eng, "wf_hist", [])):          # backfill the waterfall
            await sock.send_str(json.dumps(m))
        hub.clients.add(q)

        async def writer():
            while True:
                data = await q.get()
                await sock.send_str(data)

        wt = asyncio.create_task(writer())
        try:
            async for msg in sock:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            hub.clients.discard(q)
            wt.cancel()
        return sock

    def apply(patch):
        cfg, sdr_changed = config.update(patch)
        if sdr_changed:
            state["engine"].request_restart()
        state["hub"].emit({"type": "config", "config": cfg})
        return cfg

    async def get_config(request):
        return web.json_response(config.get())

    async def post_config(request):
        return web.json_response(apply(await request.json()))

    async def post_preset(request):
        name = (await request.json()).get("name")
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

    async def recordings(request):
        d = rec_dir()
        metas = []
        if d.exists():
            for p in sorted(d.glob("*.json"), reverse=True)[:300]:
                try:
                    m = json.loads(p.read_text())
                    m["confidence"].pop("history", None)
                    m.pop("checks", None)
                    m.pop("detect", None)
                    metas.append(m)
                except Exception:
                    pass
        return web.json_response(metas)

    async def rec_file(request):
        name = request.match_info["name"]
        if not SAFE.match(name) or Path(name).suffix not in (".wav", ".json", ".npy"):
            raise web.HTTPNotFound()
        p = rec_dir() / name
        if not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def library(request):
        d = ROOT / "captures"
        out = []
        for p in sorted(d.glob("*")):
            if p.suffix in (".wav", ".png") and p.is_file():
                side = p.with_suffix(".txt")
                out.append({"name": p.name, "kind": p.suffix[1:],
                            "note": side.read_text().strip() if side.is_file() else ""})
        return web.json_response(out)

    async def lib_file(request):
        name = request.match_info["name"]
        if not SAFE.match(name) or Path(name).suffix not in (".wav", ".png"):
            raise web.HTTPNotFound()
        p = ROOT / "captures" / name
        if not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def capture_files(request):
        d = ROOT / "captures"
        return web.json_response([f"captures/{p.name}" for p in sorted(d.glob("*")) if p.suffix in (".cu8", ".cs8")])

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws)
    app.router.add_get("/api/config", get_config)
    app.router.add_post("/api/config", post_config)
    app.router.add_post("/api/preset", post_preset)
    app.router.add_post("/api/reset", post_reset)
    app.router.add_post("/api/restart", post_restart)
    app.router.add_get("/api/recordings", recordings)
    app.router.add_get("/api/library", library)
    app.router.add_get("/api/captures", capture_files)
    app.router.add_get("/rec/{name}", rec_file)
    app.router.add_get("/lib/{name}", lib_file)
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

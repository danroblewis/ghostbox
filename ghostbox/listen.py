"""Tiny read-only web page for listening to generated WAVs and viewing plots.

Serves ONLY *.wav and *.png files sitting directly in one directory (default:
captures/), plus an index page. No directory listing, no subdirectories, no other
file types, GET/HEAD only, so it is safe-ish to expose through a tunnel.

    python3 -m ghostbox.listen --dir captures --port 8765
"""
import argparse
import html
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALLOWED = {".wav": "audio/wav", ".png": "image/png"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

NOTES = {
    "air_118.751MHz_am.wav": "Airband 118.751 MHz, AM-demodulated. Detector flagged as voice-like ~45% of the time.",
    "air_120.316MHz_am.wav": "Airband 120.316 MHz, AM-demodulated. Detector flagged as voice-like ~61% of the time.",
    "air_120.000MHz_am.wav": "Control: strongest airband carrier (120.000 MHz), NOT flagged as voice.",
    "speech.wav": "Synthetic test voice (macOS say) used for the injected ground truth.",
    "inj.png": "FM 96.6 MHz capture with injected AM voices (cyan = truth).",
    "air.png": "Airband 119.5 MHz capture, detector maps.",
}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ghostbox Listening Room</title>
<style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--accent:#7c3aed}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--accent:#a78bfa}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}}
.name{{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--accent)}}
.note{{color:var(--muted);font-size:13px;margin:2px 0 8px}}
audio{{width:100%}} img{{width:100%;height:auto;border-radius:6px}}
a{{color:var(--accent)}}
</style></head><body><main>
<h1>Ghostbox listening room</h1>
<div class="note">Read-only. Audio and plots generated from SDR captures.</div>
<h2>Audio</h2>{audio}
<h2>Plots</h2>{plots}
</main></body></html>"""


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ghostbox"
        sys_version = ""

        def _files(self, ext):
            return sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ext)

        def _index(self):
            def card(p, body):
                note = NOTES.get(p.name, "")
                return (f'<div class="card"><div class="name">{html.escape(p.name)}</div>'
                        f'<div class="note">{html.escape(note)}</div>{body}</div>')
            audio = "".join(card(p, f'<audio controls preload="none" src="{html.escape(p.name)}"></audio>')
                            for p in self._files(".wav")) or '<div class="note">none</div>'
            plots = "".join(card(p, f'<a href="{html.escape(p.name)}"><img loading="lazy" src="{html.escape(p.name)}"></a>')
                            for p in self._files(".png")) or '<div class="note">none</div>'
            body = PAGE.format(audio=audio, plots=plots).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return body

        def _serve(self, head):
            path = self.path.split("?", 1)[0].lstrip("/")
            if path in ("", "index.html"):
                body = self._index()
                if not head:
                    self.wfile.write(body)
                return
            if not SAFE_NAME.match(path) or Path(path).suffix not in ALLOWED:
                return self.send_error(404)
            fp = root / path
            if not fp.is_file() or fp.resolve().parent != root:
                return self.send_error(404)
            size = fp.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ALLOWED[fp.suffix])
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            if head:
                return
            with open(fp, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            self._serve(head=False)

        def do_HEAD(self):
            self._serve(head=True)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="captures")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    root = Path(args.dir).resolve()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"serving *.wav / *.png from {root} on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

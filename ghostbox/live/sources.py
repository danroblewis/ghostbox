"""IQ sources for the live engine: HackRF / RTL-SDR via their CLI tools streaming to
stdout, or a capture file replayed at real-time pace (loops forever).

All sources yield complex64 blocks scaled like ghostbox.iq (8-bit full scale ~ 1.0).
"""
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

CU8_OFFSET = 127.4


def _to_complex(raw, signed):
    """Interleaved 8-bit I/Q -> complex64 (scaled /128) via a float32 view: ~5x faster than slicing."""
    u = raw.astype(np.float32)
    if not signed:
        u -= CU8_OFFSET
    x = u.view(np.complex64)
    x *= 1 / 128.0
    return x


def _to_cs8(raw, signed):
    """The same samples as signed 8-bit (what the IQ ring stores), without re-quantizing."""
    return raw if signed else (raw.astype(np.int16) - 127).clip(-127, 127).astype(np.int8)


class Source:
    fs: float
    fc: float
    status = "stopped"
    error = ""

    def read(self, n, timeout=2.0):
        raise NotImplementedError

    def stop(self):
        pass


class ProcSource(Source):
    """Runs an SDR CLI that writes interleaved 8-bit IQ to stdout."""

    def __init__(self, cmd, fs, fc, signed):
        self.cmd, self.fs, self.fc, self.signed = cmd, fs, fc, signed
        self.q = queue.Queue(maxsize=400)
        self.buf = np.zeros(0, np.uint8 if not signed else np.int8)
        self.dropped = 0
        self._stop = threading.Event()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.status = "running"
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._errs, daemon=True).start()

    def _pump(self):
        dtype = np.int8 if self.signed else np.uint8
        while not self._stop.is_set():
            chunk = self.proc.stdout.read(1 << 18)
            if not chunk:
                break
            try:
                self.q.put_nowait(np.frombuffer(chunk, dtype=dtype))
            except queue.Full:
                self.dropped += len(chunk) // 2
        if not self._stop.is_set():
            self.status = "exited"

    def _errs(self):
        for line in iter(self.proc.stderr.readline, b""):
            s = line.decode(errors="replace").strip()
            low = s.lower()
            if any(w in low for w in ("error", "fail", "no hackrf", "no supported", "not found", "busy")):
                self.error = s

    def read(self, n, timeout=2.0):
        """Return exactly n complex samples, or None if the device stopped/timed out."""
        need = 2 * n
        parts = [self.buf]
        have = len(self.buf)
        deadline = time.time() + timeout
        while have < need:
            try:
                c = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                return None
            parts.append(c)
            have += len(c)
        t0 = time.time()
        allb = np.concatenate(parts)
        self.buf = allb[need:]
        self.last_raw = allb[:need]
        x = _to_complex(allb[:need], self.signed)
        self.convert_s = time.time() - t0
        return x

    def stop(self):
        self._stop.set()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.status = "stopped"


class FileSource(Source):
    """Replays one or more .cu8/.cs8 captures (played back to back).

    paced=True: real-time pace (for live replay); False: as fast as the reader asks (auto-tune).
    loop=True: start over at the end; False: read() returns None once the data runs out."""

    def __init__(self, paths, fs, fc, paced=True, loop=True):
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        self.paths, self.fs, self.fc, self.paced, self.loop = [Path(p) for p in paths], fs, fc, paced, loop
        self.signed = self.paths[0].suffix == ".cs8"
        self.raws = [np.memmap(p, dtype=np.int8 if self.signed else np.uint8, mode="r") for p in self.paths]
        self.total = sum(len(r) for r in self.raws) // 2
        self.idx, self.pos = 0, 0
        self.t0 = None
        self.sent = 0
        self.dropped = 0
        self.status = "running"

    def read(self, n, timeout=2.0):
        if self.t0 is None:
            self.t0 = time.time()
        if self.paced:                                   # don't run ahead of wall clock
            ahead = (self.sent + n) / self.fs - (time.time() - self.t0)
            if ahead > 0:
                time.sleep(ahead)
        need = 2 * n
        out = []
        while need > 0:
            raw = self.raws[self.idx]
            take = min(need, len(raw) - self.pos)
            out.append(np.asarray(raw[self.pos:self.pos + take]))
            self.pos += take
            need -= take
            if self.pos >= len(raw):
                self.idx, self.pos = self.idx + 1, 0
                if self.idx >= len(self.raws):
                    if not self.loop:
                        self.status = "finished"
                        return None
                    self.idx = 0
        self.sent += n
        self.last_raw = np.concatenate(out)
        return _to_complex(self.last_raw, self.signed)


def make_source(sdr):
    kind = sdr["source"]
    fs, fc = float(sdr["sample_rate"]), float(sdr["center_hz"])
    if kind == "hackrf":
        from .config import hackrf_gains
        lna, vga = hackrf_gains(sdr.get("gain_db", 36))
        cmd = ["hackrf_transfer", "-r", "-", "-f", str(int(fc)), "-s", str(int(fs)),
               "-a", "1" if sdr.get("amp") else "0", "-l", str(lna), "-g", str(vga)]
        if sdr.get("device"):
            cmd += ["-d", str(sdr["device"])]
        return ProcSource(cmd, fs, fc, signed=True)
    if kind == "rtlsdr":
        cmd = ["rtl_sdr", "-f", str(int(fc)), "-s", str(int(fs)), "-g", str(sdr.get("gain_db", 0)),
               "-p", str(int(sdr.get("ppm", 0)))]
        if sdr.get("device") not in (None, ""):
            cmd += ["-d", str(sdr["device"])]
        cmd += ["-"]
        return ProcSource(cmd, fs, fc, signed=False)
    if kind == "file":
        return FileSource(sdr["file"]["path"], fs, fc)
    raise ValueError(f"unknown source {kind}")


def list_devices():
    """Connected receivers: [{kind, id (serial or index for -d), label}]. A device in use by the live
    receiver is still listed (HackRF) or may not be (some RTL-SDR builds can't enumerate it)."""
    import re
    out = []
    try:
        r = subprocess.run(["hackrf_info"], capture_output=True, text=True, timeout=8)
        txt = r.stdout + r.stderr
        for m in re.finditer(r"Index:\s*(\d+)\s*\n\s*Serial number:\s*([0-9a-fA-F]+)", txt):
            serial = m.group(2).lstrip("0") or "0"
            out.append({"kind": "hackrf", "id": m.group(2), "label": f"HackRF #{m.group(1)} (serial …{serial[-8:]})"})
    except Exception:
        pass
    try:
        r = subprocess.run(["rtl_test", "-t"], capture_output=True, text=True, timeout=6)
        txt = r.stdout + r.stderr
        for m in re.finditer(r"^\s*(\d+):\s*(.+?),\s*(.+?),\s*SN:\s*(\S+)", txt, re.M):
            out.append({"kind": "rtlsdr", "id": m.group(1), "label": f"#{m.group(1)} {m.group(2)} {m.group(3)} (SN {m.group(4)})"})
    except Exception:
        pass
    return out


def test_device(sdr, seconds=0.5):
    """Open a receiver briefly and read `seconds` of samples. Returns a plain-language result."""
    fs = float(sdr["sample_rate"])
    n = int(fs * seconds)
    if sdr["source"] == "hackrf":
        from .config import hackrf_gains
        lna, vga = hackrf_gains(sdr.get("gain_db", 36))
        cmd = ["hackrf_transfer", "-r", "-", "-n", str(n), "-f", str(int(sdr["center_hz"])), "-s", str(int(fs)),
               "-a", "1" if sdr.get("amp") else "0", "-l", str(lna), "-g", str(vga)]
        signed = True
    elif sdr["source"] == "rtlsdr":
        cmd = ["rtl_sdr", "-n", str(n), "-f", str(int(sdr["center_hz"])), "-s", str(int(fs)), "-g", str(sdr.get("gain_db", 0))]
        signed = False
    else:
        return {"ok": False, "message": "A replay file isn't a device."}
    if sdr.get("device") not in (None, ""):
        cmd += ["-d", str(sdr["device"])]
    cmd += ["-"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
    except FileNotFoundError:
        return {"ok": False, "message": f"{cmd[0]} isn't installed."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "The device didn't deliver samples within 10 s."}
    raw = np.frombuffer(r.stdout, np.uint8 if not signed else np.int8)
    if len(raw) < 1000:
        err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
        why = next((l for l in err if "fail" in l.lower() or "error" in l.lower() or "no " in l.lower()), err[-1] if err else "no samples")
        return {"ok": False, "message": f"No samples: {why}"}
    return {"ok": True, **level_report(_to_complex(raw, signed), time.time() - t0)}


def level_report(x, took=None):
    """Signal level (dBFS) and clipping of complex samples in [-1, 1]."""
    x = x[:: max(1, len(x) // 200000)]
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    clip = float(np.mean((np.abs(x.real) > 0.98) | (np.abs(x.imag) > 0.98)))
    db = 20 * np.log10(rms)
    verdict = ("clipping: lower the gain" if clip > 0.002 else "very quiet: raise the gain or check the antenna" if db < -45
               else "good level")
    return {"level_dbfs": round(db, 1), "clipping_pct": round(100 * clip, 2), "verdict": verdict,
            **({"took_s": round(took, 2)} if took is not None else {})}

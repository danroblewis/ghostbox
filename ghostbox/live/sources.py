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
    """Replays a .cu8/.cs8 capture at real-time pace, looping."""

    def __init__(self, path, fs, fc):
        self.path, self.fs, self.fc = Path(path), fs, fc
        self.signed = self.path.suffix == ".cs8"
        self.raw = np.memmap(self.path, dtype=np.int8 if self.signed else np.uint8, mode="r")
        self.pos = 0
        self.t0 = None
        self.sent = 0
        self.dropped = 0
        self.status = "running"

    def read(self, n, timeout=2.0):
        if self.t0 is None:
            self.t0 = time.time()
        # pace: don't run ahead of wall clock
        ahead = (self.sent + n) / self.fs - (time.time() - self.t0)
        if ahead > 0:
            time.sleep(ahead)
        need = 2 * n
        out = []
        while need > 0:
            take = min(need, len(self.raw) - self.pos)
            out.append(np.asarray(self.raw[self.pos:self.pos + take]))
            self.pos += take
            need -= take
            if self.pos >= len(self.raw):
                self.pos = 0  # loop
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
        return ProcSource(cmd, fs, fc, signed=True)
    if kind == "rtlsdr":
        cmd = ["rtl_sdr", "-f", str(int(fc)), "-s", str(int(fs)), "-g", str(sdr.get("gain_db", 0)),
               "-p", str(int(sdr.get("ppm", 0))), "-"]
        return ProcSource(cmd, fs, fc, signed=False)
    if kind == "file":
        return FileSource(sdr["file"]["path"], fs, fc)
    raise ValueError(f"unknown source {kind}")

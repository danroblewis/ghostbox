"""`ghostbox` command: start the live web app.

    ghostbox                                  # last saved settings, http://127.0.0.1:8765
    ghostbox --device rtlsdr --freq 119.5M --rate 2.4M --gain 30
    ghostbox --list-devices

Receiver options given here are written to the saved config (live_data/config.json in the data
directory) before the server starts, exactly as if they had been changed in the web page, so they
also apply to the next start.
"""
import argparse
import asyncio
import os
import re
import sys
import webbrowser
from pathlib import Path

_SUFFIX = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def parse_hz(text):
    """'100M', '96.6 MHz', '520k', '2.4e6', '1200000' -> Hz (float)."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*([kmg]?)(?:hz)?(?:sps|s/s)?\s*", str(text), re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"not a frequency: {text!r} (examples: 100M, 96.6MHz, 520k, 2400000)")
    return float(m.group(1)) * _SUFFIX[m.group(2).lower()]


def default_data_dir():
    """Where data goes when --data-dir isn't given: the source checkout this runs from (what
    `python3 -m ghostbox.live.server` has always used), otherwise the current directory."""
    repo = Path(__file__).resolve().parents[1]
    if (repo / "pyproject.toml").exists() and (repo / "ghostbox" / "live").is_dir():
        return repo
    return Path.cwd()


def build_parser():
    ap = argparse.ArgumentParser(
        prog="ghostbox",
        description="SDR-native spirit box: the raw spectrum of a whole band of IQ as a live waterfall "
                    "you can listen into, in a web UI.",
        epilog="Receiver options (--device, --serial/--device-index, --file, --freq, --rate, --gain, --amp) "
               "are saved to the config in the data directory, as if changed in the web page.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default 127.0.0.1; 0.0.0.0 = every interface)")
    ap.add_argument("--port", type=int, default=8765, help="web port (default 8765)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="folder for config, recordings, IQ ring, surveys and presets "
                         "(default: the source checkout when run from one, else the current directory)")
    ap.add_argument("--open", action="store_true", help="open the web page in the default browser")
    ap.add_argument("--list-devices", action="store_true",
                    help="list connected HackRF / RTL-SDR receivers and exit")
    ap.add_argument("--version", action="store_true", help="print the version and exit")

    rx = ap.add_argument_group("receiver (saved to the config)")
    rx.add_argument("--device", choices=("hackrf", "rtlsdr", "file"), help="receiver type")
    who = rx.add_mutually_exclusive_group()
    who.add_argument("--serial", help="which HackRF / RTL-SDR, by serial number (see --list-devices)")
    who.add_argument("--device-index", type=int, help="which RTL-SDR, by index (see --list-devices)")
    rx.add_argument("--first-device", action="store_true",
                    help="forget a saved --serial/--device-index and use the first receiver found")
    rx.add_argument("--file", type=Path,
                    help="IQ capture to replay with --device file (.cu8 unsigned or .cs8 signed 8-bit; "
                         "a name like x_96.6MHz_2.4Msps.cu8 sets the centre frequency and rate)")
    rx.add_argument("--freq", type=parse_hz, help="centre frequency, e.g. 2.2M, 120MHz, 96600000")
    rx.add_argument("--rate", type=parse_hz,
                    help="sample rate = span heard at once, e.g. 4M (HackRF up to 20M, RTL-SDR up to ~2.4M)")
    rx.add_argument("--gain", type=float, help="gain in dB (HackRF: split into LNA + VGA; RTL-SDR: 0 = auto)")
    amp = rx.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", default=None, help="HackRF RF amplifier on (+14 dB)")
    amp.add_argument("--no-amp", dest="amp", action="store_false", help="HackRF RF amplifier off")
    return ap


def config_patch(args, cwd):
    """The `sdr` config changes the command line asks for (empty dict = none)."""
    sdr = {}
    if args.device:
        sdr["source"] = args.device
    if args.serial is not None:
        sdr["device"] = args.serial
    elif args.device_index is not None:
        sdr["device"] = str(args.device_index)
    elif args.first_device:
        sdr["device"] = ""
    if args.file is not None:
        p = args.file if args.file.is_absolute() else (cwd / args.file)
        sdr["file"] = {"path": str(p.resolve())}
    if args.freq is not None:
        sdr["center_hz"] = args.freq
    if args.rate is not None:
        sdr["sample_rate"] = args.rate
    if args.freq is not None or args.rate is not None or args.file is not None:
        sdr["fmin_hz"] = sdr["fmax_hz"] = None        # the old analysis range may not fit: use the whole capture
    if args.gain is not None:
        sdr["gain_db"] = args.gain
    if args.amp is not None:
        sdr["amp"] = args.amp
    return {"sdr": sdr} if sdr else {}


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.version:
        from importlib.metadata import PackageNotFoundError, version
        try:
            print(version("ghostbox"))
        except PackageNotFoundError:
            print("ghostbox (not installed)")
        return 0

    if args.list_devices:
        from .live.sources import list_devices
        devs = list_devices()
        if not devs:
            print("No receivers found (needs hackrf_info and/or rtl_test on PATH). "
                  "A receiver already in use by another program may not be listed.")
            return 1
        for d in devs:
            flag = "--serial" if d["kind"] == "hackrf" else "--device-index"
            print(f"{d['kind']:7} {d['label']}\n        ghostbox --device {d['kind']} {flag} {d['id']}")
        return 0

    cwd = Path.cwd()
    data_dir = (args.data_dir or default_data_dir()).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    patch = config_patch(args, cwd)
    if args.device == "file" and "file" not in patch.get("sdr", {}):
        print("note: --device file replays the capture saved in the config; choose one with --file", file=sys.stderr)
    if patch.get("sdr", {}).get("file") and not Path(patch["sdr"]["file"]["path"]).exists():
        print(f"error: no such capture: {patch['sdr']['file']['path']}", file=sys.stderr)
        return 2

    # Relative paths in the config (e.g. a replay file under captures/) resolve against the data
    # directory, as they always have against the folder the server was started from.
    os.chdir(data_dir)

    from aiohttp import web
    from .live import server
    from .live.config import Config

    server.ROOT = data_dir                              # make_app reads this for every data path
    if patch:
        Config(data_dir / "live_data" / "config.json").update(patch)

    app, token = server.make_app(args)
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url = f"http://{host}:{args.port}/?k={token}"
    print(f"ghostbox live: {url}", flush=True)
    print(f"data: {data_dir}", flush=True)
    if args.open:
        async def open_browser(app):                  # once the server is about to accept connections
            asyncio.get_running_loop().call_later(0.5, webbrowser.open, url)
        app.on_startup.append(open_browser)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

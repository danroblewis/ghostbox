# ghostbox

An SDR-native spirit box, being rebuilt from the ground up. A classic spirit box sweeps one tuner
across the broadcast band and leaves it to the listener to hear words in the fragments. ghostbox
takes the whole slice of spectrum a software-defined radio delivers at once (up to 20 MHz on a
HackRF, about 2.4 MHz on an RTL-SDR), shows it as a live waterfall, and turns the whole of it into
sound.

**Where it stands.** Every filter and detector has been removed: the next ones are being added one
at a time, each understood before the next. A ghost is taken to be a range of the spectrum whose
energy rises and falls like a voice, with nothing steady under it; ghostbox never looks for, locks
onto or uses a carrier. Right now the pipeline has no steps: what you hear is the whole band,
summed.

## Features

- **Live power waterfall** of the whole span, with history, zoom and colour maps.
- **The pipeline**: radio steps change the IQ of the whole band, then the power of everything left
  is added up 16,000 times a second into sound (SUM), then audio steps change the sound. It plays
  a few seconds behind live, and every step can be added, switched and adjusted while listening.
  Steps so far: *Flatten receiver* (radio), *Audio EQ* and *Low-pass* (audio).
- **IQ ring**: an optional rolling raw-IQ recording, so anything can be replayed or exported later.
- **Surveys** that step the receiver across a wide range and map its power.
- **Saved recordings** from before the rebuild stay browsable (transcript, playback, IQ export).
- **Transcription** engines stay installed (Whisper, Moonshine, wav2vec2, Vosk; all local) for when
  something produces clips again.
- Presets, several themes, a phone layout, and replay of IQ capture files without any hardware.

## Hardware

| Receiver | Tools needed on `PATH` | Span |
|---|---|---|
| HackRF One | `hackrf_transfer`, `hackrf_info` | up to 20 MHz |
| RTL-SDR (RTL2832U sticks, e.g. NooElec, RTL-SDR Blog) | `rtl_sdr`, `rtl_test` | up to ~2.4 MHz |
| none | - | replay `.cu8` / `.cs8` IQ files |

ghostbox drives the receivers through these command-line tools, so install them first:

```sh
# macOS (Homebrew)
brew install hackrf librtlsdr

# Debian / Ubuntu
sudo apt install hackrf rtl-sdr
```

Check that the radio is seen with `hackrf_info` or `rtl_test -t`. On Linux, RTL-SDR sticks may
need the DVB kernel driver blacklisted (`dvb_usb_rtl28xxu`) and the udev rules shipped with the
`rtl-sdr` package. For HF (below ~24 MHz) an RTL-SDR needs direct sampling or an upconverter.

## Install

Requires [uv](https://docs.astral.sh/uv/) (it fetches a suitable Python itself) and the receiver
tools above. One command installs the `ghostbox` command with everything the default settings use,
including Whisper speech recognition:

```sh
uv tool install git+https://github.com/danroblewis/ghostbox.git
```

Whisper's speech model (about 140 MB for the default) downloads the first time something is
transcribed. Then:

```sh
ghostbox --list-devices                       # what's connected
ghostbox --device hackrf --open               # start, and open the page in your browser
```

Data (settings, recordings, the IQ recording, surveys) goes in the folder you run it from, or the
one given with `--data-dir`. To update later: `uv tool upgrade ghostbox`; to remove: `uv tool uninstall ghostbox`.

Optional extra speech engines:

| extra | adds | for |
|---|---|---|
| `transformers` | transformers | Moonshine and wav2vec2 transcription |
| `vosk` | vosk | Vosk transcription |
| `all` | all of the above | |

e.g. `uv tool install "ghostbox[all] @ git+https://github.com/danroblewis/ghostbox.git"`.

### From a clone (for development)

```sh
git clone https://github.com/danroblewis/ghostbox.git
cd ghostbox
uv sync            # add --all-extras for the optional engines
uv run ghostbox --open
```

## Quick start

```sh
uv run ghostbox --list-devices                 # what's connected
uv run ghostbox --device hackrf --freq 2.2M --rate 4M --gain 36 --open
uv run ghostbox --device rtlsdr --freq 119.5M --rate 2.4M --gain 30
uv run ghostbox --device file --file some_capture_96.6MHz_2.4Msps.cu8   # no hardware
```

The server prints a URL like `http://127.0.0.1:8765/?k=...`. Open it once: the `k` value is an
access key (stored in `live_data/token` in the data directory) that sets a cookie in that browser,
because the page controls your radio. Without the key every page and API call is refused. To use
it from another machine, start with `--host 0.0.0.0` and open the URL with that machine's address;
there is no TLS, so only do that on a network you trust.

The original way to start it still works from a source checkout:
`python3 -m ghostbox.live.server --port 8765` (data goes to the checkout folder).


## On a phone (field use)

Open the same page on a phone and it switches to a field layout: the waterfall (swipe up to go back in
time, pinch to zoom), what you're hearing, the audio waveform and spectrogram, and
the saved recordings. Pause is in the top bar; tuning, display options, volume, Mark, the map and
settings are in the menu. **Mark** bookmarks the moment you're hearing (saved to `live_data/marks.jsonl`
and drawn on the waterfall). Settings › Appearance can force the phone or desktop layout.

To keep radio quiet around other people's instruments (EMF meters react to phones), connect the phone
by cable rather than Wi-Fi: phone in airplane mode, a USB-C Ethernet adapter to the computer's Ethernet
port, and open the computer's address with `--host 0.0.0.0`. Turn the computer's own Wi-Fi and
Bluetooth off too, and keep the antenna a few metres from both on its coax.

## Command-line options

| option | |
|---|---|
| `--port N` | web port (default 8765) |
| `--host ADDR` | listen address (default 127.0.0.1) |
| `--open` | open the page in the default browser |
| `--data-dir DIR` | where config and data live (see below) |
| `--list-devices` | list connected HackRF / RTL-SDR receivers with the option to select each, then exit |
| `--device hackrf\|rtlsdr\|file` | receiver type |
| `--serial SN` / `--device-index N` | which receiver when several are connected (HackRF serial; RTL-SDR index or serial) |
| `--first-device` | forget a saved serial/index, use the first receiver found |
| `--file PATH` | IQ capture to replay with `--device file` (`.cu8` unsigned or `.cs8` signed 8-bit interleaved I/Q; a name containing e.g. `96.6MHz_2.4Msps` sets centre and rate) |
| `--freq F` | centre frequency: `2.2M`, `120MHz`, `96600000` |
| `--rate R` | sample rate = span heard at once: `4M`, `2.4M` |
| `--gain DB` | gain in dB (HackRF: split between LNA and VGA; RTL-SDR: 0 = automatic) |
| `--amp` / `--no-amp` | HackRF RF amplifier (+14 dB) |
| `--version` | print the version |

Receiver options are **saved** into the config, exactly as if you had changed them in the page,
so they also apply to the next start. Giving `--freq`, `--rate` or `--file` also resets the analysis range to
the whole capture. Everything else (the pipeline, transcription, storage) is set in the web page.

Gain: start low. Strong broadcast signals overload the front end and create false signals;
the receiver test in the app reports clipping. On the FM band an RTL-SDR clips above roughly 10 dB.

## Where data goes

Everything lives under the data directory: `--data-dir`, or by default the source checkout when
run from one (as `python3 -m ghostbox.live.server` always did), otherwise the current directory.

| path | contents |
|---|---|
| `live_data/config.json` | all settings |
| `live_data/token` | the access key |
| `live_data/presets/` | settings you saved |
| `live_data/surveys/` | surveys and their maps |
| `recordings/` | recordings saved before the rebuild: WAV audio, JSON details (capped, oldest deleted) |
| `iq_ring/` | rolling raw IQ recording when enabled (capped, oldest deleted) |
| `captures/` | IQ files you put there are offered for replay |

Recordings and the IQ ring can contain anything your receiver picked up; they are excluded by the
`.gitignore`. Copy the folders elsewhere to keep an investigation.

## Getting started in the app

The **?** button in the page shows the same steps.

1. **Connect your receiver.** Open *⚙ › Receiver & storage*, choose the receiver type and, if you
   have several, the device. Click *Test*: it should report a good level. If it says *clipping*,
   lower the gain; if it says *very quiet*, raise it or check the antenna.
2. **Choose how much to see.** The *span* selector sets how wide a slice the waterfall shows.
3. **Listen.** What you hear is the whole band, summed (see *How it works*). Click the page once
   to enable sound. *Pause* (or space) stops playback; click the waterfall or drag the playback
   line to play from a little further back.
4. **Survey a wide range.** Click *Map*, then *New*, enter a range and a dwell per step, and start.
   It steps across the range and maps its power.

## How it works

**Waterfall.** The IQ stream is turned into a power spectrogram (bins of roughly 125-250 Hz,
25 frames/s), pooled to the screen's width and sent to the page as it arrives, with history kept on
the server so the page can scroll back.

**The pipeline** (`pipeline.py`). Raw IQ goes through the radio steps in order (each takes IQ and
returns IQ of the same length), then SUM makes sound, then the audio steps (sound in, sound out)
and a slow level control. SUM: the power of all the IQ,
moment by moment, as 16,000 samples a second. That is
the band's total power moment by moment (by Parseval's theorem, the same as adding up every column
of a spectrum of that sliver). Its steady part (below 20 Hz, inaudible) is taken out.

- *Flatten receiver* (radio): a receiver doesn't pass every frequency equally, so signals that move
  in frequency (FM) change in measured power as they cross its slopes, and SUM hears them ("slope
  detection"). This step learns the receiver's response from the noise floor (the 20th percentile
  of each frequency's average power over a window wider than a station, default 1 MHz) and boosts
  or cuts the IQ to flatten it, in overlapping FFT blocks. At full strength FM stations fade from
  the sum.
- *Audio EQ* (audio): a graphic equalizer, one peaking filter per octave from 63 Hz to 6.3 kHz.
- *Low-pass* (audio): removes everything above a cutoff, steeply (Butterworth, order 8 by default).

SUM never lets fast power changes fold into the sound: two stations 40 kHz apart make the band's
power pulse at 40 kHz, and a plain 1/16,000 s average would turn that into a tone at 8 kHz. So SUM
averages the power in exact 1/512,000 s slices, then low-passes in two stages (512 k -> 64 k ->
16 k samples a second): changes up to 7 kHz pass unchanged, changes above 8 kHz are removed.

The waterfall shows the IQ after the radio steps; *Raw IQ* on the chain bar shows it untouched. Before the sum everything is
in the radio domain (radio frequency on the axis); after it, in the audio domain (the panel under
the waterfall shows the sound's own spectrum). The page plays each chunk at its stream time plus a
delay, so the sound stays locked to its place on the waterfall.

## Repository layout

- `ghostbox/live/` - the live app: `server.py` (web server and API), `engine.py` (receiver,
  waterfall, IQ ring), `pipeline.py` (the IQ steps and the sum into sound), `audio.py` (level and fades), `autotune.py` (calibration shell:
  finding and replaying recorded IQ), `survey.py`, `asr.py` / `transcribe.py` (speech recognition),
  `sources.py` (receivers), `config.py` (settings and defaults), `static/` (web page).
- `ghostbox/cli.py` - the `ghostbox` command.
- `ghostbox/maps.py`, `ghostbox/iq.py` - the power spectrogram, IQ file loading.

## License

MIT; see [LICENSE](LICENSE).

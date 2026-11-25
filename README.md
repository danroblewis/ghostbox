# spiritbox-sdr (ghostbox)

An SDR-native spirit box. A classic spirit box sweeps one tuner across the broadcast band and
leaves it to the listener to hear words in the fragments. ghostbox works the other way round: it
takes the whole slice of spectrum a software-defined radio delivers at once (up to 20 MHz on a
HackRF, about 2.4 MHz on an RTL-SDR), looks at every frequency in it simultaneously, and picks out
the signals that carry something shaped like a human voice, including voices that hop between
channels or slide in frequency. You listen to those, a few seconds behind live, in a web page.

It is a signal-processing tool, not an oracle. Everything it reports is a radio signal that passed
a set of voice-likeness checks: broadcast and amateur stations, aviation, utility transmissions,
interference, and occasionally noise that happens to look speech-like. What you make of a
detection is up to you; the app gives you the evidence (spectrogram, check scores, audio, IQ) to
judge it.

## Features

- **Whole-band detection** of AM (double-sideband, with or without carrier) and SSB voice directly
  from IQ: every frequency bin is scored on every frame, no sweeping.
- **Live web UI**: waterfall (power, combined voice score, novelty, or any single check), detection
  list with confidence, delayed live playback locked to the waterfall, per-detection waveform and
  spectrum.
- **Hops and slides**: detections are tracked across frequency; a voice that jumps to another
  channel or drifts is followed and joined into one chain.
- **Calibration** to the band you are on, automatically or on demand, using the recorded IQ.
- **Surveys** that step the receiver across a wide range (e.g. 0.5-30 MHz), calibrate on each
  step, transcribe what they find and draw a map you can zoom into and replay from.
- **Demodulators**: synchronous AM, suppressed-carrier (carrier recovery), synchronous sideband,
  envelope, USB/LSB, plus voice band-pass, noise reduction and level control for playback.
- **"Tune to this detection"**: searches demodulation and filtering settings for one detection.
- **Transcription** (optional): Whisper (faster-whisper), Moonshine, wav2vec2 or Vosk, all local.
- **Recording**: detections saved as WAV + JSON (+ narrowband IQ); an optional rolling raw-IQ
  recording ("IQ ring") so anything can be re-analysed or exported later.
- Presets, several themes, and replay of IQ capture files without any hardware.

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

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/danroblewis/spiritbox-sdr.git
cd spiritbox-sdr
uv sync --extra recommended
```

Extras (all optional; the detector and web app run on numpy, scipy and aiohttp alone):

| extra | adds | used for |
|---|---|---|
| `vad` | torch, silero-vad | the ML voice detector check, calibration, "Tune to this detection" |
| `whisper` | faster-whisper | transcription with Whisper (default engine; surveys use Whisper tiny) |
| `transformers` | torch, transformers | Moonshine and wav2vec2 transcription |
| `vosk` | vosk | Vosk transcription |
| `plots` | matplotlib | the offline analysis script `ghostbox.analyze` |
| `recommended` | `vad` + `whisper` | what the default settings use |
| `all` | everything above | |

`uv sync --all-extras` installs everything. Speech models download on first use. Without the
`vad` extra the ML voice check scores nothing and calibration is much weaker; without a
transcription engine, transcripts are simply missing.

Alternatively, without cloning: `uv tool install "ghostbox[recommended] @ git+https://github.com/danroblewis/spiritbox-sdr"`;
then choose where data goes with `--data-dir` (see below).

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
the whole capture. Everything else (detector, checks, playback, transcription, storage) is set in
the web page.

Gain: start low. Strong broadcast signals overload the front end and create false "stations";
the receiver test in the app reports clipping. On the FM band an RTL-SDR clips above roughly 10 dB.

## Where data goes

Everything lives under the data directory: `--data-dir`, or by default the source checkout when
run from one (as `python3 -m ghostbox.live.server` always did), otherwise the current directory.

| path | contents |
|---|---|
| `live_data/config.json` | all settings |
| `live_data/token` | the access key |
| `live_data/presets/` | settings you saved |
| `live_data/surveys/` | surveys, their maps and clips |
| `live_data/profiles/` | learned per-band background profiles |
| `recordings/` | saved detections: WAV audio, JSON details, narrowband IQ (capped, oldest deleted) |
| `iq_ring/` | rolling raw IQ recording when enabled (capped, oldest deleted) |
| `captures/` | IQ files you put there are offered for replay |

Recordings and the IQ ring can contain anything your receiver picked up; they are excluded by the
`.gitignore`. Copy the folders elsewhere to keep an investigation.

## Getting started in the app

The **?** button in the page shows the same steps.

1. **Connect your receiver.** Open *Settings > Receiver*, choose the receiver type and, if you
   have several, the device. Click *Test*: it should report a good level. If it says *clipping*,
   lower the gain; if it says *very quiet*, raise it or check the antenna.
2. **Choose how much to listen to.** The *span* selector sets how wide a slice you hear at once.
   Wider covers more but needs more CPU; 4 MHz is a good start. Keep the CPU indicator below about
   70%; if the app says it can't keep up, pick a narrower span.
3. **Let it calibrate itself.** In the *Detector* tab, turn on *Recalibrate automatically*. After
   a couple of minutes on a band it studies what it heard, works out which checks separate voices
   best there, and sets its thresholds. It repeats this after a retune and when nothing has been
   caught for a while.
4. **Listen to what it finds.** The *Detections* tab lists voice-like signals, newest first, with
   a confidence. Audio plays a few seconds behind live; click the page once to enable sound.
   *Pause* (or space) stops live playback. Click a detection for its transcript, playback with a
   spectrogram, other demodulations and downloads; *Tune to this detection* finds the clearest
   settings for it.
5. **Survey a wide range.** Click *Map*, then *New*, enter a range (e.g. 0.5-30 MHz) and a dwell
   per step (1 min), and start. It steps across the range, calibrates and transcribes each step,
   and fills in the map as it goes (with a 4 MHz span, 0.5-30 MHz is 10 steps, about 12 minutes).
   Zoom with the scroll wheel, click a box for details, and use *Tune here* to listen live.
6. **Keep what you found.** Detections at least 70% confident (adjustable) are saved to
   `recordings/` automatically; save others from the list before closing. The raw IQ ring is a
   rolling buffer, so download the *IQ WAV* of anything important from its detail window.

## How it works

**Whole-band IQ checks (stage 1).** The IQ stream is turned into a power spectrogram (bins of
roughly 125-250 Hz, 25 frames/s) plus a fine 25 Hz spectrum for pitch. Every bin is scored on
every frame by a set of checks, each giving 0-1:

- AM structure: upper and lower sidebands that rise and fall together (*twinning*, also along
  slanted lines for drifting voices), a mirror-image sideband spectrum, a carrier above its
  sidebands. SSB: voice-band rhythm on one side of a (suppressed) carrier only. AM and SSB are
  alternative families; whichever fits better counts.
- Speech rhythm: sideband loudness pulsing at the syllable rate (2-10 Hz, with a 3-6 Hz peak and
  little energy at 8-12 Hz), loudness range, voiced/unvoiced alternation.
- Pitch: evenly spaced harmonic lines around a carrier at a human voice pitch, measured from the
  raw IQ for all frequencies at once, and whether that pitch glides like speech.
- Spectral shape: energy strongest near the carrier and falling toward 3.4 kHz.
- Rejections: wideband FM (loudness also pulsing 4.5-8 kHz out), broadband electrical pulses.
- Novelty and prominence: evidence is compared with each bin's running baseline and its
  neighbours, and optionally with a learned per-band profile ("unusual for this band").

The scores are combined as a weighted geometric mean. Candidate bins are linked over time into
tracks that may slide in frequency or hop to another channel.

**Per-detection checks (stage 2).** Each track is demodulated from the raw IQ along exactly its
path and checked again on the audio and narrowband IQ: pitch comb and motion, voicing, rhythm,
AM phase coherence (carrier and sidebands in phase, unlike FM or noise), cross-channel
co-modulation (another frequency with the same syllable rhythm: one voice spread across or hopping
between channels), hop continuity (pitch carried across a hop) and, if installed, the Silero
neural voice-activity detector. The result is the detection's confidence.

**Calibration.** There are no labelled "ghost" signals to train on, but the band's background can
be measured. Calibration replays the last few minutes of recorded IQ through the detector with
permissive thresholds, keeps every candidate with all its check scores, then searches check-weight
combinations offline, choosing the one whose top detections the ML voice detector rates most
speech-like at the target rate (the ML check's own weight is held fixed so it can't vote for
itself). It sets the weights and thresholds accordingly; on a quiet band that means more sensitive
settings than the defaults.

**Surveys.** The receiver records a dwell of IQ at each step across a range while a worker
processes finished steps: replay through the detector, calibrate on that step, recompute
confidences, transcribe what remains with Whisper tiny, and add the step to a map of power, voice
score and every check. The receiver returns to where it was afterwards.

**Demodulation and playback.** Playback uses synchronous AM when a carrier is present, a
suppressed-carrier loop when not, and a product detector for SSB, followed by a steep voice
band-pass, spectral-gating noise reduction and slow level control. The detector itself always
judges the unprocessed demodulated audio; playback processing never changes what is detected.

**Transcription.** Detections above a confidence threshold (and whole hop chains, stitched) are
transcribed locally. Whisper is the most accurate but can invent fluent text from noise; wav2vec2
has no language model and returns fragments rather than sentences on noise; Vosk is the lightest.
Common filler phrases that Whisper produces from noise are filtered out.

## Repository layout

- `ghostbox/live/` - the live app: `server.py` (web server and API), `engine.py` (pipeline),
  `stage1.py` / `stage2.py` (checks), `audio.py` (demodulators, playback processing),
  `autotune.py` (calibration), `audiotune.py` (tune to a detection), `survey.py`, `asr.py` /
  `transcribe.py` (speech recognition), `sources.py` (receivers), `config.py` (settings and
  defaults), `static/` (web page).
- `ghostbox/cli.py` - the `ghostbox` command.
- `ghostbox/*.py` - the offline prototype tools (IQ loading, analysis, voice injection for tests).

## License

MIT; see [LICENSE](LICENSE).

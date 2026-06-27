# Nova — offline voice assistant

**N**atural-language **O**riented **V**oice **A**ssistant — "Claude, but it listens." Nova runs
**fully offline** on your PC: it hears you, thinks locally, talks back, and *does things*
(open apps, search, control media, and more). No cloud, no API keys required.

Pipeline: **Whisper** (speech-to-text) → optional VAE text cleaner → **Ollama** brain
(local LLM with function-calling + memory) → **Piper** (text-to-speech). Optional wake word
and barge-in on top.

---

## Project status

**Working now — fully offline, no cloud, no API keys:**
- ✅ Speech-to-text — faster-whisper on GPU (Vosk fallback)
- ✅ Brain — Ollama (`qwen3:8b`) with function-calling tools + persistent, self-summarizing memory
- ✅ Text-to-speech — Piper (ElevenLabs optional)
- ✅ Realtime loop — wake word (openWakeWord) + barge-in
- ✅ Skills framework — drop a file in `skills/` to add a capability (11 skills live)
- ✅ Extras — voice model-switching, system "vibe check", Attenborough mode, sass counter
- ✅ Hardened app-launcher (no shell injection); logs auto-purge after 7 days; integration-tested 27/27

**In progress:**
- 🔄 Custom **"hey Nova"** wake word — training locally (16.5 GB data + 5,000 clips done; model training running)

**Planned (see [ROADMAP.md](ROADMAP.md)):**
- ⬜ Phase 1 — learns from your corrections (RAG)
- ⬜ Phase 3 — connected services (email, Discord, calendar)
- ⬜ Phase 4 — computer-use (on-screen agency)
- ⬜ Phase 5 — self-improvement (DPO/LoRA + self-extending skills)
- ⬜ Phase 6 — proactivity
- ⬜ More Windows control (typing, clipboard, windows); optional `gemma4:12b` brain

---

## Prerequisites — install these first

Model weights and datasets are **not in this repo** (they're in `.gitignore` — too big for
git). Download them with the steps below.

### 1. Python deps
Python 3.10+ then:
```bash
pip install -r requirements.txt
```
- Optional alternative backends (Vosk STT, ElevenLabs TTS, wake word): `pip install -r requirements-optional.txt`
- Phase-1 VAE text cleaner / training extras: `pip install -r requirements-vae.txt`

### 2. Ollama (the local LLM brain)
Install from **https://ollama.com/download**, then pull the default model:
```bash
ollama pull qwen3:8b
```
Ollama serves an OpenAI-compatible API at `http://localhost:11434`. To use a different model,
set `LOCAL_LLM_MODEL` in `.env` (e.g. `gemma4:12b` — see https://ollama.com/library).

### 3. Piper voice (text-to-speech)
Download a voice from **https://huggingface.co/rhasspy/piper-voices** and place the two files
in `models/piper/`. The default Nova expects is **en_US-amy-medium**:
- [en_US-amy-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx)
- [en_US-amy-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json)

Override the path with `PIPER_MODEL_PATH` in `.env` to use a different voice.

### 4. Whisper model (speech-to-text)
**Auto-downloads** on first run (`faster-whisper` pulls `small.en` from HuggingFace) — no manual
step. For **GPU** acceleration on Windows you also need the CUDA 12 runtime (already in
`requirements.txt`):
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
No NVIDIA GPU? Set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` in `.env`.

### 5. Wake word (optional)
`hey_jarvis` (and `alexa`, `hey_mycroft`, `hey_rhasspy`) **auto-download** via
[openWakeWord](https://github.com/dscripka/openWakeWord) on first use. A custom **"hey Nova"**
must be trained — see [scripts/wakeword/README.md](scripts/wakeword/README.md). Enable with
`USE_WAKE_WORD=1` in `.env`.

### 6. Configure
Copy the template and edit as needed:
```bash
cp .env.example .env
```
The offline defaults work out of the box. Common tweaks: `MIC_INDEX` (run `python test_mic.py`
to list devices), `LOCAL_LLM_MODEL`, `USE_WAKE_WORD`, `USE_BARGE_IN`.

---

## Run
```bash
python nova.py
```
Speak, and Nova listens → thinks locally → replies aloud. Say **"exit"**, **"quit"**, or
**"stop"** to leave. With `USE_WAKE_WORD=1`, say the wake phrase first.

---

## What it can do
- **Actions:** open websites & apps, web search, tell the time, media/volume keys, power control.
- **Meta:** switch model by voice ("switch to gemma"), list its own skills ("what can you do").
- **Fun:** vibe check (system stats), Attenborough mode (narrates your screen), sass counter.
- **Realtime:** wake word + barge-in (talk over Nova to interrupt it).

**Adding a capability = dropping one file in [`skills/`](skills/)** — each skill self-registers
via the `@skill` decorator.

---

## Project layout
| Path | What |
|---|---|
| `nova.py` / `realtime.py` | entry point + the listen→think→speak loop |
| `stt/` | speech-to-text (Whisper, Vosk) |
| `brain/` | the LLM brain: function-calling loop + conversation memory |
| `skills/` | capabilities (web, apps, system, media, fun…) — auto-discovered |
| `tts/` | text-to-speech (Piper, ElevenLabs) |
| `wake/` | wake-word detection |
| `representation/` | optional Phase-1 VAE text cleaner |
| `utils/` | logs (auto-purged after 7 days), system stats, CLI banner |

Runtime state (`data/*.json`, `logs/`) and downloaded weights (`models/`, `training/`) are
gitignored and stay on your machine only.

---

## Troubleshooting & hard-won fixes

A full log of the real issues we hit bringing Nova up on **Windows 11 + Python 3.14**
(bleeding-edge), and exactly how each was solved. Most come from very new package versions
that dropped old APIs, or from Linux-only assumptions in upstream tooling.

### Environment / PATH
- **`python` opens the Microsoft Store instead of running** — the Store *App execution alias*
  shadows the real interpreter. Fix: Settings → Apps → Advanced app settings → App execution
  aliases → turn off `python.exe`/`python3.exe`, or call Python by its full path.
- **`python`/`ollama` "not recognized" in a terminal** — the shell captured `PATH` before the
  install. Fix: open a fresh terminal (or refresh `PATH` from the registry).

### GPU speech-to-text (faster-whisper)
- **`RuntimeError: Library cublas64_12.dll is not found`** — ctranslate2 needs the CUDA 12
  runtime. Fix: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`; `stt/whisper_stt.py` then
  adds their `bin` folders to the DLL search path via `os.add_dll_directory()` **before**
  importing `faster_whisper`. No NVIDIA GPU? Set `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`.

### Text-to-speech (Piper)
- **`PiperVoice` has no attribute `synthesize_stream_raw`** — the API changed in piper-tts ≥1.3.
  Fix: use `voice.synthesize(text)` and iterate the yielded `AudioChunk` objects
  (`.audio_int16_array`, `.sample_rate`). See `tts/piper_tts.py`.
- **Voice download is truncated / `INVALID_PROTOBUF: Protobuf parsing failed`** —
  `Invoke-WebRequest` cut the Git-LFS file short. Fix: download via `huggingface_hub`
  (handles LFS + integrity), not a raw HTTP GET.

### Brain (Ollama / qwen3)
- **Nova reads its reasoning aloud** — qwen3 emits `<think>…</think>`. Fix: strip those blocks
  before speaking (`_strip_think` in `brain/gpt_llm.py`).
- **Every reply errors with "model not found"** — `LOCAL_LLM_MODEL` points at a model you
  haven't pulled. Fix: `ollama pull <model>` or change `.env`.
- **Ollama server dies during a network blip; pulls crawl** — restart the Ollama app. A
  half-finished `ollama pull` leaves a `sha256-…-partial` blob in `~/.ollama/models/blobs`
  (safe to delete to reclaim space; complete blobs have no `-partial` suffix).

### Large downloads on a flaky/slow link
- **HF feature file stalls / `httpx.ReadTimeout`** — HuggingFace's Xet backend and the default
  10s read-timeout choke on big files over a poor connection. Fixes that worked:
  `HF_HUB_DISABLE_XET=1`, `HF_HUB_DOWNLOAD_TIMEOUT=120`, and ultimately a **byte-range
  resumable downloader** (`Range: bytes=N-`, retry-resume up to hundreds of times) launched as
  a **detached** `Start-Process` — harness/agent background tasks have a lifetime cap and die
  mid-download, but a detached process survives.
- **`Start-Process` fails with `can't open file 'Z:\VS'`** — the space in `Z:\VS CODE` split the
  argument. Fix: pass a **relative** script path + `-WorkingDirectory "Z:\VS CODE\Nova"`.

### Custom "hey Nova" wake-word training — why it needs WSL/Linux
openWakeWord's training pipeline assumes Linux. On Windows + Python 3.14 we hit a *cascade* of
incompatibilities. Each was individually patched in `scripts/wakeword/train_wakeword.py`, but
they kept coming — in order:

1. **MSVC compiler required** — `audiomentations` pulls `numpy-minmax`/`numpy-rms`, which have no
   Python 3.14 wheel and build from C source. Fix: pin **`audiomentations<0.34`** (drops them).
2. **`acoustics` import fails** — does `from scipy.special import sph_harm` /
   `from scipy.interpolate import interp2d`; both **removed in scipy ≥1.17 / ≥1.14**. Fix:
   restore the names on `scipy` before openWakeWord imports `acoustics`.
3. **numpy 2.0 removed aliases** (`np.float`, `np.int`, …) used by old deps. Fix: restore them.
4. **`generate_samples` signature mismatch** — the current piper-sample-generator moved the
   function into a package and now **requires a `model` arg** the (older) trainer never passes.
   Fix: a tiny `generate_samples.py` adapter shim that injects the `.pt` model path.
5. **`ImportError: TorchCodec is required` (via `torchaudio.load`)** — torchaudio ≥2.9 routes
   audio loading through TorchCodec (not installed; needs FFmpeg). Fix: monkeypatch
   `torchaudio.load` to fall back to **soundfile** (already installed).
6. **`module 'torchaudio' has no attribute 'info'`** — same removal. Fix: soundfile `info` shim.
7. **`ValueError: Clip does not have the correct sample rate!`** — piper clips are 22.05 kHz and
   ESC-50 noise is 44.1 kHz, but openWakeWord is **strict 16 kHz**. Fix: resample all clips +
   noise to 16 kHz (scipy `resample_poly`).
8. **`torchmetrics` ↔ `speechbrain` k2_fsa clash** — importing `openwakeword.data` (→ speechbrain)
   before `torchmetrics` triggers a lazy import of `k2_fsa`, an optional dep needing `k2`
   (uninstalled). Fix: `import torchmetrics` first.
9. **`PermissionError: [WinError 32]` in `trim_mmap`** — openWakeWord `os.remove`s a feature file
   that is **still memory-mapped** (`utils.py` keeps the `open_memmap` handle open across the
   trim). Deleting an mmap'd file is legal on Linux, **forbidden on Windows**, and the lock is
   held inside the library where we can't reach it. **Not fixable from outside.**
10. **Next would be** `DataLoader(num_workers>0)` pickling a generator-backed `IterableDataset`,
    which fails on Windows (spawn) but works on Linux (fork).

**Conclusion:** issues 9–10 are fundamental Linux assumptions (delete-while-mmapped, fork-based
multiprocessing), not one-line patches. The right fix is to **train in WSL2** (Ubuntu on
Windows). The data we already downloaded is reused from `/mnt/z/VS CODE/Nova/training/` — no
re-download. Everything *else* in Nova runs natively on Windows; only the custom wake word
needs Linux, and `hey_jarvis` works in the meantime.

### Power outage mid-training
The training ran as a detached process and died with the outage, but **all downloaded data and
generated clips survived on disk** — so recovery was just relaunching; nothing re-downloaded or
re-generated. (The expensive steps write to disk and are inherently resumable.)

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

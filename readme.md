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

- ✅ Custom **"hey Nova"** wake word — trained in WSL (see Troubleshooting); enable with `USE_WAKE_WORD=1` + `WAKE_WORD_MODEL=models/wakeword/hey_nova.onnx`

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
- **`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`** — ctranslate2
  needs the CUDA 12 runtime. Fix: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`;
  `stt/whisper_stt.py` adds their `bin` folders to the DLL search path via
  `os.add_dll_directory()` **before** importing `faster_whisper`. **Crucial gotcha:** keep the
  handles `os.add_dll_directory()` returns **alive** (store them in a module global) — if they're
  garbage-collected, the directory is *removed* from the search path, and ctranslate2 loads
  cuBLAS **lazily at `encode()`** (the first real transcription), long after import, so it fails
  *then*, not at startup. (Also add the dirs to `PATH` so dependent DLLs resolve.) **Testing
  gotcha:** transcribing silence skips `encode()` (the VAD filters it), so a "whisper works" test
  on silence is a false pass — validate with `vad_filter=False` on real/energetic audio. No
  NVIDIA GPU? Set `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`.

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
- **Replies take ~20 s** — it's *not* your CPU or network (everything's local). Two causes:
  (1) qwen3 *thinks* (chain-of-thought) before every answer, even after we hide it — pure
  latency. Fix: disable it with `/no_think` in the system prompt (done in `gpt_llm.py`).
  (2) The **first** reply after startup loads the 5 GB model into VRAM (~10 s, one-time);
  later turns are warm (~2 s plain, ~4 s with a tool call). **Gaming at the same time** is the
  other big factor — the game and the LLM fight over the same VRAM/GPU, and if VRAM is full
  Ollama spills layers to the CPU (much slower). For snappy replies, don't game simultaneously,
  or `switch to` a smaller model while gaming.
- **Nova uses emoji** even though the prompt says not to — models leak them anyway. Fix: a hard
  emoji/pictograph strip on the final text (`_clean_reply` + `_EMOJI_RE` in `gpt_llm.py`), plus
  a firmer prompt rule.
- **Wrong / made-up time or date** (e.g. "11:32 AM on Sunday, June 14" when it's really late
  June) — qwen3 **didn't call the `get_current_time` tool**, it *hallucinated* a time. Worse,
  that wrong answer got saved to `data/memory.json`, so it parroted the same lie on every later
  turn (**memory poisoning**). Fixes: (1) a `CRITICAL:` prompt rule forcing a tool call for any
  time/date question and `temperature=0.3` for obedience; (2) clear the poisoned history
  (`del data/memory.json` — it rebuilds empty). The clock and tool were always correct; the
  model just wasn't using them.
- **Every reply errors with "model not found"** — `LOCAL_LLM_MODEL` points at a model you
  haven't pulled. Fix: `ollama pull <model>` or change `.env`.

### Voice & interaction (robotic voice, barge-in, exit words)
- **Voice sounds robotic** — that's the Piper voice model, not a bug. Swap `amy-medium` for a
  more natural one — **`en_US-libritts_r-medium`** (LibriTTS-R, the most human-like) or
  `en_US-hfc_female-medium`. Download the `.onnx` + `.onnx.json` from
  [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/piper/` (use
  `huggingface_hub`, not a plain fetch — a raw download truncates the model → `INVALID_PROTOBUF`)
  and point `PIPER_MODEL_PATH` at it in `.env`.
- **Barge-in won't stop her when you talk over her** — the detector is a mic-energy threshold, so
  it's setup-dependent. Run once with `BARGE_DEBUG=1` to print your peak mic level each reply,
  then set `BARGE_RMS_THRESHOLD` in `.env` just under it. On speakers the mic also hears Nova's
  own voice (echo) — **headphones make barge-in reliable**. A monitor-mic that fails to open now
  prints a warning instead of silently disabling barge-in.
- **"quit" doesn't exit but "exit" does** — "quit" is a short plosive word Whisper often
  mis-hears as "with"/"it". Use **"exit", "stop", "goodbye", or "bye"** (all matched by
  `is_exit_command`, whole-utterance so "stop the music" won't quit). We deliberately DON'T map
  "with"/"it" to quit — that would cause random accidental exits.
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

### Running the wake-word training in WSL2 (the fix that worked)
Moving to WSL2 (Ubuntu) cleared the two fatal Windows blockers — but a few new issues came up
there too:

- **WSL not installed** — `wsl --install` in an **admin** PowerShell, then **reboot**. (The
  Start-menu Ubuntu launcher may "open and close instantly"; running `wsl ...` from PowerShell
  works regardless.)
- **Privileged setup without a password** — `wsl -u root <cmd>` runs as root passwordless, so
  `apt` installs need no login.
- **Garbled output / quoting hell** — set `$env:WSL_UTF8=1` for clean text, and pass multi-line
  bash via a file: `wsl bash -c "tr -d '\r' < /mnt/c/.../script.sh | bash"` (the `tr` strips
  Windows CRLF). Inline heredocs through PowerShell→WSL→bash get mangled.
- **Ubuntu 26.04 also ships Python 3.14** — so the scipy/numpy/torchaudio API-removal issues
  *recur*, but the **same** `train_wakeword.py` shims fix them (they're cross-platform). The two
  *fatal* Windows blockers (`trim_mmap` `os.remove`, fork multiprocessing) simply don't exist on
  Linux, so training **completes**.
- **`openwakeword==0.6.0` won't install on Linux** — it hard-requires `tflite-runtime`, which has
  no Python 3.14 wheel, so pip silently falls back to **0.4.0, which has no `train` module at
  all**. Fix: `pip install --no-deps openwakeword==0.6.0` (training uses ONNX, not tflite, and the
  deps are already installed).
- **Big file over the drive mount** — numpy `mmap_mode='r'` on the 16.5 GB feature file works
  fine from `/mnt/z` (tested), so **no need to copy 16.5 GB into Linux**. Keep the *output*
  (computed features, the model) on the WSL **native** filesystem (`~/...`); only read big inputs
  from `/mnt/z`.
- **piper not installed in WSL** — the trainer still `import`s `generate_samples`, so the shim
  degrades to a stub (we reuse the already-generated 16 kHz clips, so generation is skipped).
- **Keeping the long job alive** — launch with `setsid … &` so it survives the `wsl.exe` call
  returning.

One-time setup, then run (reuses all the Windows-side data/clips):
```bash
# in WSL (Ubuntu)
sudo apt install -y python3-pip python3-venv build-essential ffmpeg libsndfile1
python3 -m venv ~/nova-train
~/nova-train/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
~/nova-train/bin/pip install "audiomentations<0.34" torch_audiomentations speechbrain acoustics \
    torchinfo torchmetrics pronouncing mutagen soundfile
~/nova-train/bin/pip install --no-deps openwakeword==0.6.0
~/nova-train/bin/pip install onnxscript   # torch's ONNX exporter needs it
~/nova-train/bin/python -c "import openwakeword.utils as u; u.download_models()"

cd "/mnt/z/VS CODE/Nova"
NOVA_TRAIN_CONFIG=scripts/wakeword/hey_nova_wsl.yaml ~/nova-train/bin/python scripts/wakeword/train_wakeword.py
```
The result `hey_nova.onnx` goes to `models/wakeword/` on the Windows side; set `USE_WAKE_WORD=1`
and `WAKE_WORD_MODEL=models/wakeword/hey_nova.onnx` in `.env`.

### WSL training — runtime gotchas (memory, multiprocessing, export)
Even on Linux, getting the run to actually *finish* took a few more fixes — all handled in
`scripts/wakeword/train_wakeword.py` and the WSL config:

- **OOM-killed at ~75% of training.** WSL2 only gets ~half your Windows RAM (7.6 GiB here), and
  two things blew past it — both fixed:
  - openWakeWord builds a **sliding-window array** from `validation_set_features.npy`
    (`[X[i:i+16] for i in range(0, N, 1)]` over ~480k frames = a multi-GB array). Fix: **truncate
    the FP-validation set** (e.g. to 50k frames) — it only estimates a false-positive rate and
    doesn't affect the trained model.
  - mmap'ing the 16.5 GB feature file straight from `/mnt/z` (the Windows drive mount) creates
    **non-reclaimable** pages that accumulate. Fix: **copy big inputs to native ext4** (`~/...`),
    where mmap pages are evictable file-cache, and point the config there. (Raising the WSL RAM
    cap via `.wslconfig` helps too, but the array can exceed even that — shrink it.)
- **`PicklingError: Can't pickle <class '__main__.IterDataset'>` at the training step.** Python
  3.14 changed the default multiprocessing start method to `forkserver` on Linux, so
  `DataLoader(num_workers>0)` tries to pickle openWakeWord's generator-backed dataset. Fix:
  `multiprocessing.set_start_method("fork", force=True)`.
- **`OSError: ... written` / corrupt `.npy`.** `np.save` over a file that's still mmap-open
  corrupts it (partial write). Fix: write the truncated set to a **new** filename, reading from
  the intact source.
- **`ModuleNotFoundError: No module named 'onnxscript'` at the final export.** torch's ONNX
  exporter needs it. Fix: `pip install onnxscript` (now in the setup above).

Operational note: long *blocking* WSL commands launched from PowerShell get SIGTERM'd after ~30 s
in this setup, so run the training **detached** (`setsid … &`, writing to a log) and poll with
short, non-blocking checks.

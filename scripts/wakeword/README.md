# Training a custom "hey Nova" wake word

The wake word is a trained neural net, not a string — there is no pretrained "hey nova",
so we have to make one. Nova's code is already ready for it: once you have a
`hey_nova.onnx`, drop it in `models/wakeword/` and set in `.env`:

```ini
USE_WAKE_WORD=1
WAKE_WORD_MODEL=models/wakeword/hey_nova.onnx
```

Pick one of three routes.

---

## Route A — Train locally (fully offline, free)  ← needs a C++ compiler
**Blocker found:** the training stack (`audiomentations` → `numpy-minmax`/`numpy-rms`)
has no prebuilt wheel for Python 3.14 and must compile, which needs the
**Microsoft C++ Build Tools** (not installed).

1. Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   (select "Desktop development with C++"). One-time, ~2–6 GB.
2. Install the training deps:
   ```
   pip install torchinfo torchmetrics pronouncing audiomentations torch_audiomentations speechbrain mutagen acoustics
   ```
   (torch + torchaudio + openwakeword are already installed.)
3. The sample generator is already cloned at `training/piper-sample-generator/`.
4. Get the example config from the openWakeWord repo
   (`examples/custom_model.yml`), set `target_phrase: ["hey nova"]`,
   `model_name: hey_nova`, point `piper_sample_generator_path` at the clone above,
   and fill `rir_paths` / `background_paths` with a noise + impulse-response dataset
   (the official notebook lists the datasets to download — a few GB).
5. Run the pipeline (CPU here, ~1 hr):
   ```
   python -m openwakeword.train --training_config hey_nova.yaml --generate_clips --augment_clips --train_model
   ```
6. Copy the resulting `hey_nova.onnx` into `models/wakeword/`.

## Route B — Train in Google Colab (free GPU, ~30–60 min)  ← CHOSEN
Nothing heavy touches your PC: datasets + compute live on Colab; only the final
~1–2 MB `hey_nova.onnx` comes back.

1. Open openWakeWord's **automatic_model_training.ipynb** (linked from the openWakeWord
   GitHub repo) in Google Colab.
2. Runtime → Change runtime type → **GPU**.
3. In the config cell set the target phrase to **`hey nova`** (and model name `hey_nova`).
   Leave the dataset/augmentation cells as-is — they auto-download on Colab.
4. Run all cells. At the end it exports `hey_nova.onnx`.
5. Download that file → put it in `models/wakeword/` here.
6. In `.env`: `USE_WAKE_WORD=1` and `WAKE_WORD_MODEL=models/wakeword/hey_nova.onnx`.

Detection then runs 100% offline. Tip: generate a few thousand samples for a robust model;
if it false-triggers or misses, raise/lower `WAKE_WORD_THRESHOLD` in `.env`.

## Route C — Picovoice Porcupine (~5 min)  ← fastest, but online creation + account
1. Free account at https://console.picovoice.ai/ → type "Hey Nova" → download the `.ppn`.
2. `pip install pvporcupine` and add a small `WakeWordDetector` adapter (same interface as
   `wake/wake_word.py`). Detection runs offline; only model *creation* uses their console.

---

Recommendation: **Route B** if you want it working today; **Route A** if you want it 100%
self-hosted and don't mind installing the compiler.

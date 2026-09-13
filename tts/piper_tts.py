# tts/piper_tts.py
# Offline text-to-speech with Piper (runs on CPU, no internet).
# Needs a voice model: a .onnx file plus its matching .onnx.json, pointed to by
# PIPER_MODEL_PATH. Download voices from:
#   https://huggingface.co/rhasspy/piper-voices

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from config import PIPER_MODEL_PATH

_voice = None
_voice_lock = threading.Lock()  # warm_up() and the first reply may race to load it


def _get_voice():
    global _voice
    with _voice_lock:
        if _voice is None:
            from piper.voice import PiperVoice  # imported lazily so config can load without piper
            _voice = PiperVoice.load(PIPER_MODEL_PATH)
    return _voice


def warm_up() -> None:
    """Load the voice in the background at startup. The first load takes ~3 s,
    which would otherwise land on the first thing Nova says."""
    def load():
        try:
            _get_voice()
        except Exception:
            pass  # a real problem will surface (and be reported) on the first speak

    threading.Thread(target=load, name="piper-warmup", daemon=True).start()


def synthesize(text: str):
    """Synthesize `text` -> (int16 mono numpy array, sample_rate)."""
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.int16), 22050

    voice = _get_voice()
    arrays = []
    rate = 22050
    for chunk in voice.synthesize(text):  # piper-tts >=1.3 yields AudioChunk objects
        rate = chunk.sample_rate
        arrays.append(chunk.audio_int16_array)

    if not arrays:
        return np.zeros(0, dtype=np.int16), rate
    return np.concatenate(arrays), rate


def play_async(samples: np.ndarray, rate: int) -> None:
    """Start playback without blocking (used by the barge-in loop)."""
    sd.play(samples, samplerate=rate)


def stop_playback() -> None:
    """Cut playback off immediately (barge-in)."""
    sd.stop()


def speak(text: str) -> None:
    """Synthesize and play `text`, blocking until finished."""
    try:
        samples, rate = synthesize(text)
        if samples.size == 0:
            return
        sd.play(samples, samplerate=rate)
        sd.wait()
    except Exception as e:
        print(f"[Piper TTS error] {e}")
        print(f"Nova: {text}")  # fall back to text so you still get the reply

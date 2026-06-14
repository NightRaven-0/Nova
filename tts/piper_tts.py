# tts/piper_tts.py
# Offline text-to-speech with Piper (runs on CPU, no internet).
# Needs a voice model: a .onnx file plus its matching .onnx.json, pointed to by
# PIPER_MODEL_PATH. Download voices from:
#   https://huggingface.co/rhasspy/piper-voices

from __future__ import annotations

import numpy as np
import sounddevice as sd

from config import PIPER_MODEL_PATH

_voice = None


def _get_voice():
    global _voice
    if _voice is None:
        from piper.voice import PiperVoice  # imported lazily so config can load without piper
        _voice = PiperVoice.load(PIPER_MODEL_PATH)
    return _voice


def speak(text: str) -> None:
    """Synthesize `text` offline and play it through the default output device."""
    text = (text or "").strip()
    if not text:
        return
    try:
        voice = _get_voice()
        rate = voice.config.sample_rate

        raw = bytearray()
        for chunk in voice.synthesize_stream_raw(text):
            raw.extend(chunk)

        samples = np.frombuffer(bytes(raw), dtype=np.int16)
        sd.play(samples, samplerate=rate)
        sd.wait()
    except Exception as e:
        print(f"[Piper TTS error] {e}")
        print(f"Nova: {text}")  # fall back to text so you still get the reply

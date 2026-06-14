# tts/voice.py
# Picks the TTS backend at call time (lazy import).

from config import TTS_BACKEND


def speak(text: str) -> None:
    if TTS_BACKEND == "none":
        print(f"Nova: {text}")
        return
    if TTS_BACKEND == "elevenlabs":
        from tts.elevenlabs_tts import speak as _impl
    else:  # default: piper
        from tts.piper_tts import speak as _impl
    _impl(text)

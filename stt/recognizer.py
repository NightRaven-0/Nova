# stt/recognizer.py
# Picks the STT backend at call time (lazy import so only one model loads).

from config import STT_BACKEND


def listen_and_transcribe() -> str:
    if STT_BACKEND == "vosk":
        from stt.vosk_stt import listen_and_transcribe as _impl
    else:  # default: whisper
        from stt.whisper_stt import listen_and_transcribe as _impl
    return _impl()

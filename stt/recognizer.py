# stt/recognizer.py
# Picks the STT backend at call time (lazy import so only one model loads).

from config import STT_BACKEND


def listen_and_transcribe(start_timeout_s=None):
    """Transcribe one utterance. Returns None if nothing was spoken within
    start_timeout_s (used by the realtime loop to detect idleness)."""
    if STT_BACKEND == "vosk":
        from stt.vosk_stt import listen_and_transcribe as _impl
        return _impl()  # vosk backend: no idle-timeout support
    from stt.whisper_stt import listen_and_transcribe as _impl
    return _impl(start_timeout_s=start_timeout_s)

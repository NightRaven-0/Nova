# stt/whisper_stt.py
# Offline speech-to-text with faster-whisper (GPU-accelerated).

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    SAMPLE_RATE,
    MIC_INDEX,
)
from utils.audio_utils import record_utterance

# Load the model once. First run downloads weights to the HF cache.
print(f"Loading Whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}... (first run downloads it)")
_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


def transcribe_buffer(audio: np.ndarray) -> str:
    """Transcribe a mono float32 16kHz numpy buffer. Reusable by the realtime loop."""
    if audio is None or audio.size == 0:
        return ""
    segments, _info = _model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(seg.text for seg in segments).strip()


def listen_and_transcribe() -> str:
    """One-shot: wait for an utterance, then transcribe it."""
    print("Listening... speak now.")
    audio = record_utterance(samplerate=SAMPLE_RATE, mic_index=MIC_INDEX)
    return transcribe_buffer(audio)

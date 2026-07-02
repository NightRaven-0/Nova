# stt/whisper_stt.py
# Offline speech-to-text with faster-whisper (GPU-accelerated).

from __future__ import annotations

import os
import sys

import numpy as np


# Keep os.add_dll_directory handles alive for the whole process — if they're
# garbage-collected, the directory is REMOVED from the DLL search path, and
# ctranslate2 loads cuBLAS lazily at encode() time (long after import), so the
# dir must still be present then.
_dll_dir_handles = []


def _add_nvidia_dll_dirs() -> None:
    """ctranslate2 (faster-whisper's engine) needs CUDA cuBLAS + cuDNN DLLs.
    When those come from the nvidia-*-cu12 pip packages rather than a system CUDA
    install, their bin folders aren't on Windows' DLL search path, so add them
    before faster_whisper is imported (handles kept alive + also put on PATH so
    dependent DLLs resolve)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import nvidia  # namespace package created by the nvidia-*-cu12 wheels
    except ImportError:
        return
    bindirs = []
    for pkg_dir in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            bindir = os.path.join(pkg_dir, sub, "bin")
            if os.path.isdir(bindir):
                bindirs.append(bindir)
    for bindir in bindirs:
        _dll_dir_handles.append(os.add_dll_directory(bindir))  # MUST keep the handle
    if bindirs:
        os.environ["PATH"] = os.pathsep.join(bindirs) + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402

from config import (  # noqa: E402
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    SAMPLE_RATE,
    MIC_INDEX,
)
from utils.audio_utils import record_utterance  # noqa: E402

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


def listen_and_transcribe(start_timeout_s=None):
    """One-shot: wait for an utterance, then transcribe it. Returns None if
    nothing was spoken within `start_timeout_s` seconds (idle)."""
    print("Listening... speak now.")
    audio = record_utterance(
        samplerate=SAMPLE_RATE, mic_index=MIC_INDEX, start_timeout_s=start_timeout_s
    )
    if audio is None:
        return None
    return transcribe_buffer(audio)

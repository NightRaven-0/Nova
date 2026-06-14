# utils/audio_utils.py
# RMS calc, silence detection, and energy-VAD utterance recording.

from __future__ import annotations

import numpy as np
import sounddevice as sd


def rms(frame: np.ndarray) -> float:
    """Root-mean-square energy of a float32 audio frame in [-1, 1]."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def record_utterance(
    samplerate: int = 16000,
    mic_index=None,
    frame_ms: int = 30,
    silence_threshold: float = 0.012,
    silence_ms: int = 800,
    max_seconds: float = 15.0,
    start_timeout_s: float | None = None,
) -> np.ndarray | None:
    """Record a single spoken utterance using a simple energy VAD.

    Waits for speech to start, captures until `silence_ms` of quiet (or
    `max_seconds`), and returns a mono float32 numpy array at `samplerate`.
    Returns None if nothing was spoken before `start_timeout_s`.
    """
    frame_len = max(1, int(samplerate * frame_ms / 1000))
    silence_frames = max(1, int(silence_ms / frame_ms))
    max_frames = max(1, int(max_seconds * 1000 / frame_ms))
    start_timeout_frames = (
        None if start_timeout_s is None else int(start_timeout_s * 1000 / frame_ms)
    )

    captured: list[np.ndarray] = []
    started = False
    silent_run = 0
    waited = 0

    with sd.InputStream(
        samplerate=samplerate,
        device=mic_index,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
    ) as stream:
        while True:
            data, _ = stream.read(frame_len)
            frame = data[:, 0].copy()
            level = rms(frame)

            if not started:
                if level >= silence_threshold:
                    started = True
                    captured.append(frame)
                else:
                    waited += 1
                    if start_timeout_frames is not None and waited >= start_timeout_frames:
                        return None
                continue

            captured.append(frame)
            if level < silence_threshold:
                silent_run += 1
                if silent_run >= silence_frames:
                    break
            else:
                silent_run = 0

            if len(captured) >= max_frames:
                break

    if not captured:
        return None
    return np.concatenate(captured).astype(np.float32)

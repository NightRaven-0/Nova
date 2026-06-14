# realtime.py
# Layer 3: the realtime loop that wraps capture -> brain -> speak with optional
# wake word (Nova only engages when called) and barge-in (talk over Nova to stop
# it). Controlled by USE_WAKE_WORD / USE_BARGE_IN in config.

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from utils.cli import print_banner
from brain.gpt_llm import ask_gpt
from representation import build_phase1_processor
from stt.recognizer import listen_and_transcribe
from config import (
    SAMPLE_RATE,
    MIC_INDEX,
    USE_WAKE_WORD,
    USE_BARGE_IN,
    WAKE_WORD_MODEL,
    WAKE_WORD_THRESHOLD,
    BARGE_RMS_THRESHOLD,
    TTS_BACKEND,
)

FRAME = 1280  # 80 ms @ 16 kHz — openWakeWord's expected chunk size
EXIT_WORDS = {"quit", "exit", "stop", "goodbye"}
_BARGE_MIN_FRAMES = 4  # ~320 ms of sustained speech before we count it as barge-in


def _rms(frame_i16: np.ndarray) -> float:
    f = frame_i16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))) if f.size else 0.0


def _wait_for_wake(detector) -> None:
    """Block until the wake phrase is heard."""
    detector.reset()
    with sd.InputStream(
        samplerate=SAMPLE_RATE, device=MIC_INDEX, channels=1,
        dtype="int16", blocksize=FRAME,
    ) as stream:
        while True:
            data, _ = stream.read(FRAME)
            if detector.triggered(data[:, 0]):
                return


def _speak(text: str) -> bool:
    """Speak `text`. With barge-in enabled (Piper backend only), stop early if the
    user talks over Nova. Returns True if it was interrupted."""
    if TTS_BACKEND != "piper" or not USE_BARGE_IN:
        from tts.voice import speak
        speak(text)
        return False

    from tts.piper_tts import synthesize, play_async, stop_playback

    samples, rate = synthesize(text)
    if samples.size == 0:
        return False

    play_async(samples, rate)
    deadline = time.time() + len(samples) / float(rate)

    loud = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, device=MIC_INDEX, channels=1,
            dtype="int16", blocksize=FRAME,
        ) as mic:
            while time.time() < deadline:
                data, _ = mic.read(FRAME)
                if _rms(data[:, 0]) >= BARGE_RMS_THRESHOLD:
                    loud += 1
                    if loud >= _BARGE_MIN_FRAMES:
                        stop_playback()
                        return True
                else:
                    loud = 0
    except Exception:
        # If a second mic stream can't open during playback, just let it finish.
        sd.wait()
        return False

    sd.wait()
    return False


def run_realtime() -> None:
    print_banner()
    phase1 = build_phase1_processor()

    detector = None
    if USE_WAKE_WORD:
        from wake.wake_word import WakeWordDetector
        print(f"[wake] loading '{WAKE_WORD_MODEL}'...")
        detector = WakeWordDetector(WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD)

    skip_wake = False
    while True:
        if detector and not skip_wake:
            print(f"Waiting for wake word ('{WAKE_WORD_MODEL}')... say it to start.")
            _wait_for_wake(detector)
            print("(wake word detected)")
        skip_wake = False

        raw_text = listen_and_transcribe()
        if not raw_text:
            continue

        user_input = phase1.process(raw_text).cleaned_text
        if not user_input:
            continue

        print(f" You:  {user_input}")
        if user_input.lower().strip(" .!?") in EXIT_WORDS:
            _speak("Goodbye!")
            print(" Goodbye!")
            break

        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")
        if _speak(reply):
            print(" (interrupted — listening)")
            skip_wake = True  # user is mid-sentence; capture it without re-waking

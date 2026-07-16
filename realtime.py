# realtime.py
# Layer 3: the realtime loop that wraps capture -> brain -> speak with optional
# wake word (Nova only engages when called) and barge-in (talk over Nova to stop
# it). Controlled by USE_WAKE_WORD / USE_BARGE_IN in config.

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

import os

from utils.cli import (
    print_banner, is_exit_command, is_sleep_command,
    print_you, print_nova, print_status,
)
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
    SLEEP_AFTER_S,
    TTS_BACKEND,
    USE_PROACTIVE,
    PROACTIVE_ACTIVITY_HOURS,
    PROACTIVE_RAM_ALERT,
)

FRAME = 1280  # 80 ms @ 16 kHz — openWakeWord's expected chunk size
_BARGE_MIN_FRAMES = 4      # ~320 ms of sustained speech before we count it as barge-in
_BARGE_GRACE_SEC = 0.35   # ignore the very start (playback ramp-up / mic settling)
_BARGE_DEBUG = os.getenv("BARGE_DEBUG") == "1"  # print peak mic level to help tune the threshold
_barge_warned = False     # only warn once if the monitor mic can't open


def _rms(frame_i16: np.ndarray) -> float:
    f = frame_i16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))) if f.size else 0.0


def _wait_for_wake(detector, monitor=None) -> str:
    """Block until the wake phrase is heard, or a proactive nudge is queued.
    Returns 'wake' or 'proactive'."""
    detector.reset()
    with sd.InputStream(
        samplerate=SAMPLE_RATE, device=MIC_INDEX, channels=1,
        dtype="int16", blocksize=FRAME,
    ) as stream:
        while True:
            data, _ = stream.read(FRAME)
            if monitor is not None and monitor.has_pending():
                return "proactive"
            if detector.triggered(data[:, 0]):
                return "wake"


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

    global _barge_warned
    play_async(samples, rate)
    start = time.time()
    deadline = start + len(samples) / float(rate)

    loud = 0
    peak = 0.0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, device=MIC_INDEX, channels=1,
            dtype="int16", blocksize=FRAME,
        ) as mic:
            while time.time() < deadline:
                data, _ = mic.read(FRAME)
                level = _rms(data[:, 0])
                peak = max(peak, level)
                # Ignore the first fraction of a second (playback ramp-up / mic settling).
                if time.time() - start < _BARGE_GRACE_SEC:
                    continue
                if level >= BARGE_RMS_THRESHOLD:
                    loud += 1
                    if loud >= _BARGE_MIN_FRAMES:
                        stop_playback()
                        if _BARGE_DEBUG:
                            print(f"   [barge-in fired: peak {peak:.3f} >= {BARGE_RMS_THRESHOLD}]")
                        return True
                else:
                    loud = 0
        if _BARGE_DEBUG:
            print(f"   [barge-in: peak mic level {peak:.3f} (threshold {BARGE_RMS_THRESHOLD}) "
                  f"— lower BARGE_RMS_THRESHOLD if you spoke and it didn't stop]")
    except Exception as e:
        if not _barge_warned:
            print(f"   [barge-in disabled: couldn't open the monitor mic during playback ({e}). "
                  f"Set USE_BARGE_IN=0 to silence this, or use headphones.]")
            _barge_warned = True
        sd.wait()
        return False

    sd.wait()
    return False


def run_realtime() -> None:
    print_banner()
    phase1 = build_phase1_processor()
    from utils import afk

    detector = None
    if USE_WAKE_WORD:
        from wake.wake_word import WakeWordDetector
        print(f"[wake] loading '{WAKE_WORD_MODEL}'...")
        detector = WakeWordDetector(WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD)

    # Proactive monitor: a background thread that queues nudges (reminders,
    # long-activity, resource alerts) for us to speak. It never touches audio.
    monitor = None
    if USE_PROACTIVE:
        from utils.proactive import ProactiveMonitor
        monitor = ProactiveMonitor(
            activity_hours=PROACTIVE_ACTIVITY_HOURS, ram_alert=PROACTIVE_RAM_ALERT
        )
        monitor.start()

    # Once-a-day greeting on the first launch of the day.
    from config import GREET_ON_FIRST_LAUNCH
    if GREET_ON_FIRST_LAUNCH:
        from utils.proactive import first_launch_today, daily_greeting
        if first_launch_today():
            greeting = daily_greeting()
            print_nova(greeting)
            _speak(greeting)

    def _say_proactive() -> None:
        """Speak any nudges the monitor has queued (main thread = safe for audio)."""
        if monitor:
            for msg in monitor.drain():
                print_nova(msg)
                _speak(msg)

    # Awake = in an active conversation (listens to every turn, no wake word needed).
    # Asleep = dormant, waiting for the wake word. With no wake word configured she
    # is always awake.
    awake = detector is None

    while True:
        if detector and not awake:
            print_status("asleep — say the wake word to wake me")
            if _wait_for_wake(detector, monitor) == "proactive":
                _say_proactive()      # a nudge came due while sleeping — say it, stay asleep
                continue
            awake = True
            afk.on_active()  # user is back — clear any AFK status
            _speak("Yes?")

        _say_proactive()  # deliver any pending nudges before we start listening

        # Awake-with-wake-word: time out after SLEEP_AFTER_S of silence and sleep.
        # No wake word but proactive on: short tick so nudges can still fire.
        # Neither: wait indefinitely.
        if detector:
            timeout = SLEEP_AFTER_S
        elif monitor:
            timeout = 30.0
        else:
            timeout = None
        result = listen_and_transcribe(start_timeout_s=timeout)

        if result is None:  # nothing spoken within the window
            if detector:
                print_status("quiet for a while — dozing off")
                awake = False
                afk.on_idle()  # mark AFK (skips if you're watching a video)
            # no detector: just loop — _say_proactive() at the top delivers nudges
            continue

        user_input = phase1.process(result).cleaned_text
        if not user_input:
            continue

        print_you(user_input)

        if is_exit_command(user_input):
            _speak("Goodbye!")
            print_status("goodbye")
            break

        if is_sleep_command(user_input):
            _speak("Going to sleep. Call me when you need me.")
            print_status("going to sleep")
            awake = False
            continue

        reply = ask_gpt(user_input)
        print_nova(reply)
        _speak(reply)  # barge-in handled inside; we stay awake and loop to listen

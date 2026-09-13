# realtime.py
# Layer 3: the realtime loop that wraps capture -> brain -> speak with optional
# wake word (Nova only engages when called) and barge-in (talk over Nova to stop
# it). Replies stream, so Nova starts talking after the first sentence.
# Controlled by USE_WAKE_WORD / USE_BARGE_IN in config.

from __future__ import annotations

import sounddevice as sd

from utils.cli import (
    print_banner, is_exit_command, is_sleep_command,
    print_you, print_nova, print_status, NovaLine,
)
from brain.gpt_llm import ask_gpt_stream, cancel_reply, warm_up_model
from representation import build_phase1_processor
from stt.recognizer import listen_and_transcribe
from tts.speaker import speak, speak_stream
from config import (
    SAMPLE_RATE,
    MIC_INDEX,
    USE_WAKE_WORD,
    WAKE_WORD_MODEL,
    WAKE_WORD_THRESHOLD,
    SLEEP_AFTER_S,
    USE_PROACTIVE,
    PROACTIVE_ACTIVITY_HOURS,
    PROACTIVE_RAM_ALERT,
)

FRAME = 1280  # 80 ms @ 16 kHz — openWakeWord's expected chunk size


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
            speak(greeting)

    def _say_proactive() -> None:
        """Speak any nudges the monitor has queued (main thread = safe for audio)."""
        if monitor:
            for msg in monitor.drain():
                print_nova(msg)
                speak(msg)

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
            # Ollama unloads the model after ~5 idle minutes (about when she dozes
            # off), so start reloading it now, while you're still talking.
            warm_up_model()
            afk.on_active()  # user is back — clear any AFK status
            speak("Yes?")

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
            speak("Goodbye!")
            print_status("goodbye")
            break

        if is_sleep_command(user_input):
            speak("Going to sleep. Call me when you need me.")
            print_status("going to sleep")
            awake = False
            continue

        # Stream the reply: each sentence is spoken (and printed) as soon as the
        # brain finishes it. Talking over her stops the voice and the generation.
        line = NovaLine()
        interrupted = speak_stream(
            ask_gpt_stream(user_input), on_sentence=line, on_interrupt=cancel_reply
        )
        line.end()
        if interrupted:
            print_status("interrupted — go ahead")

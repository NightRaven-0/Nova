from utils.cli import print_banner
from stt.recognizer import listen_and_transcribe
from brain.gpt_llm import ask_gpt
from tts.voice import speak
from representation import build_phase1_processor
from config import USE_WAKE_WORD, USE_BARGE_IN
from dotenv import load_dotenv

load_dotenv()

EXIT_WORDS = {"quit", "exit", "stop", "goodbye"}


def _simple_loop():
    """Plain listen -> reply -> speak loop (no wake word, no barge-in)."""
    print_banner()
    phase1_processor = build_phase1_processor()

    while True:
        raw_text = listen_and_transcribe()
        if not raw_text:
            continue

        user_input = phase1_processor.process(raw_text).cleaned_text
        if not user_input:
            continue

        print(f" You:  {user_input}")

        if user_input.lower().strip(" .!?") in EXIT_WORDS:
            speak("Goodbye!")
            print(" Goodbye!")
            break

        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")
        speak(reply)


def main():
    # The realtime loop adds wake word and/or barge-in; fall back to the simple
    # loop only when both are disabled.
    if USE_WAKE_WORD or USE_BARGE_IN:
        from realtime import run_realtime
        run_realtime()
    else:
        _simple_loop()


if __name__ == "__main__":
    main()

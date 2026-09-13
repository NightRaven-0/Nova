from utils.cli import print_banner, is_exit_command, print_you, print_status, NovaLine
from stt.recognizer import listen_and_transcribe
from brain.gpt_llm import ask_gpt_stream
from tts.speaker import speak, speak_stream
from representation import build_phase1_processor
from config import USE_WAKE_WORD, USE_BARGE_IN
from dotenv import load_dotenv

load_dotenv()


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

        print_you(user_input)

        if is_exit_command(user_input):
            speak("Goodbye!")
            print_status("goodbye")
            break

        line = NovaLine()
        speak_stream(ask_gpt_stream(user_input), on_sentence=line)
        line.end()


def main():
    # Tidy up Nova's own old logs (>7 days) so activity logs never pile up.
    from utils.logs import purge_old_logs
    deleted, freed = purge_old_logs(retention_days=7)
    if deleted:
        print(f"[logs] removed {deleted} old log file(s), freed {freed/1e6:.1f} MB")

    # Make sure the local brain is reachable before we start (starts Ollama if
    # it isn't running — common after a reboot/power cut).
    from config import LLM_BACKEND, TTS_BACKEND
    if LLM_BACKEND == "ollama":
        from brain.gpt_llm import ensure_ollama, warm_up_model
        ensure_ollama()
        warm_up_model()  # load the model into VRAM now, not on the first question

    if TTS_BACKEND == "piper":
        from tts.piper_tts import warm_up
        warm_up()

    # The realtime loop adds wake word and/or barge-in; fall back to the simple
    # loop only when both are disabled.
    if USE_WAKE_WORD or USE_BARGE_IN:
        from realtime import run_realtime
        run_realtime()
    else:
        _simple_loop()


if __name__ == "__main__":
    main()

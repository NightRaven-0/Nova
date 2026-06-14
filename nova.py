from utils.cli import print_banner
from stt.recognizer import listen_and_transcribe
from brain.gpt_llm import ask_gpt
from tts.voice import speak
from representation import build_phase1_processor
from dotenv import load_dotenv

load_dotenv()


def main():
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

        if user_input.lower().strip(" .!?") in ("quit", "exit", "stop", "goodbye"):
            speak("Goodbye!")
            print(" Goodbye!")
            break

        # The LLM decides whether to answer or call a tool (open apps, search, etc.).
        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")
        speak(reply)


if __name__ == "__main__":
    main()

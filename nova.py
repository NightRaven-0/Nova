# nova.py
# Main entry point

from utils.cli import print_banner
from utils.system_commands import execute_command
from stt.vosk_stt import listen_and_transcribe
from brain.gpt_llm import ask_gpt
from tts.elevenlabs_tts import speak
from config import USE_TTS

def main():
    print_banner()

    while True:
        user_input = listen_and_transcribe()
        print(f" You: {user_input}")

        if not user_input:
            continue

        # Quit commands
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("👋 Goodbye!")
            break

        # Try to handle as a system command first
        sys_response = execute_command(user_input)
        if sys_response:
            print(f" Nova: {sys_response}")
            if USE_TTS:
                speak(sys_response)
            # skip GPT if a system command was executed
            continue

        # Otherwise, ask GPT
        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")

        if USE_TTS:
            speak(reply)

if __name__ == "__main__":
    main()

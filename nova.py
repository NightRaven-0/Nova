from stt.vosk_stt import listen_and_transcribe
from brain.gpt_llm import ask_gpt
from tts.elevenlabs_tts import speak
from config import USE_TTS

def main():
    print("Nova Assistant Ready!")
    while True:
        user_input = listen_and_transcribe()
        print(f" You: {user_input}")

        if user_input.lower() in ["quit", "exit", "stop"]:
            print("👋 Goodbye!")
            break

        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")

        # ⭐ NEW — Speak the reply
        if USE_TTS:
            speak(reply)

if __name__ == "__main__":
    main()

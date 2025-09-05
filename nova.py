# Main entry point
from stt.vosk_stt import listen_and_transcribe
from brain.gpt_llm import ask_gpt

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

if __name__ == "__main__":
    main()

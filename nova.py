# nova.py
# Main entry point

from utils.cli import print_banner
from utils.system_commands import execute_command
from stt.vosk_stt import listen_and_transcribe
from brain.gpt_llm import ask_gpt
from tts.elevenlabs_tts import speak
from representation import build_phase1_processor
from config import USE_TTS


def main():
    print_banner()
    phase1_processor = build_phase1_processor()

    while True:
        raw_text = listen_and_transcribe()
        phase1_result = phase1_processor.process(raw_text)
        user_input = phase1_result.reconstructed_text

        print(f" [raw]  You: {phase1_result.raw_text}")
        print(f" [vae]  You: {phase1_result.reconstructed_text}")
        print(f" [loss] reconstruction_loss={phase1_result.reconstruction_loss:.6f}")
        print(f" [sim]  nearest_score={phase1_result.nearest_score:.4f}")

        if phase1_result.latent_vector:
            print(f" [z]    latent[:8]={phase1_result.latent_vector[:8]}")

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "stop"]:
            print(" Goodbye!")
            break

        sys_response = execute_command(user_input)
        if sys_response:
            print(f" Nova: {sys_response}")
            if USE_TTS:
                speak(sys_response)
            continue

        reply = ask_gpt(user_input)
        print(f" Nova: {reply}")

        if USE_TTS:
            speak(reply)


if __name__ == "__main__":
    main()
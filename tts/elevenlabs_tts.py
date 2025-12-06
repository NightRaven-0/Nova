# tts/elevenlabs_tts.py
# Speaking (text → ElevenLabs voice)

from elevenlabs import generate, play, set_api_key
from config import ELEVENLABS_API_KEY, VOICE_ID

# Configure ElevenLabs API
set_api_key(ELEVENLABS_API_KEY)

def speak(text: str) -> None:
    """Convert text to speech using ElevenLabs."""
    text = text.strip()
    if not text:
        return

    try:
        audio = generate(
            text=text,
            voice=VOICE_ID,                # e.g. "Shelby"
            model="eleven_multilingual_v2" # good general model
        )
        play(audio)
    except Exception as e:
        print(f"[TTS error] {e}")

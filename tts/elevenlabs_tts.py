# tts/elevenlabs_tts.py
# Speaking (text → ElevenLabs voice) using the new SDK

import os
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from config import ELEVENLABS_API_KEY, VOICE_ID

# Create client once
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def speak(text: str) -> None:
    """Convert text to speech using ElevenLabs."""
    text = text.strip()
    if not text:
        return

    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,               # must be a *voice ID*, see note below
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",   # works with play()
        )
        play(audio)
    except Exception as e:
        print(f"[TTS error] {e}")

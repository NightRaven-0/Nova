# Speaking (text → ElevenLabs voice)
import os
from elevenlabs import ElevenLabs
from config import ELEVENLABS_API_KEY, VOICE_ID

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def speak(text: str):
    """Convert text to speech using ElevenLabs."""
    if not text.strip():
        return
    
    audio = client.generate(
        text=text,
        voice=VOICE_ID,
        model="eleven_multilingual_v2"
    )

    # Stream output directly to speakers
    client.stream(audio)

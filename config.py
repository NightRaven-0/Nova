# Loads .env + constants (API keys, audio settings, assistant name)

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Assistant settings
ASSISTANT_NAME = "Nova"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file")

if not ELEVENLABS_API_KEY:
    raise ValueError("❌ Missing ELEVENLABS_API_KEY in .env file")

# Paths
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-en-us-0.15")

# Audio settings
SAMPLE_RATE = 16000 
CHUNK_SIZE = 4000 

# Voice settings
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Shelby") 

# Feature toggles
# USE_CLOUD_STT: 0 = Use Vosk (offline), 1 = Use Whisper API (online)
USE_CLOUD_STT = str(os.getenv("USE_CLOUD_STT", "0")) == "1"

# USE_TTS: 0 = Text only (no speech), 1 = Speak replies aloud with ElevenLabs
USE_TTS = str(os.getenv("USE_TTS", "1")) == "1"

# USE_CLOUD_LLM: 0 = Use local GPT-4, 1 = Use OpenAI API (cloud)
USE_CLOUD_LLM = str(os.getenv("USE_CLOUD_LLM", "0")) == "1"
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
    raise ValueError("Missing OPENAI_API_KEY in .env file")

if not ELEVENLABS_API_KEY:
    raise ValueError("Missing ELEVENLABS_API_KEY in .env file")

# Paths
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-en-us-0.15")

LOCAL_ONLY_MODE = str(os.getenv("LOCAL_ONLY_MODE", "1")) == "1"

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

# ---------------------------
# Phase 1: VAE representation
# ---------------------------

USE_VAE_TEXT_CLEANER = str(os.getenv("USE_VAE_TEXT_CLEANER", "0")) == "1"

VAE_MODEL_PATH = os.getenv(
    "VAE_MODEL_PATH",
    os.path.join("models", "phase1_text_vae.pt"),
)

VAE_BANK_PATH = os.getenv(
    "VAE_BANK_PATH",
    os.path.join("models", "phase1_reconstruction_bank.json"),
)

VAE_LOG_PATH = os.getenv(
    "VAE_LOG_PATH",
    os.path.join("logs", "vae_interactions.jsonl"),
)

VAE_SENTENCE_MODEL = os.getenv(
    "VAE_SENTENCE_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

VAE_SIMILARITY_THRESHOLD = float(
    os.getenv("VAE_SIMILARITY_THRESHOLD", "0.75")
)

RAW_MATCH_THRESHOLD = float(os.getenv("RAW_MATCH_THRESHOLD", "0.78"))
RAW_MARGIN_THRESHOLD = float(os.getenv("RAW_MARGIN_THRESHOLD", "0.10"))

VAE_MATCH_THRESHOLD = float(os.getenv("VAE_MATCH_THRESHOLD", "0.75"))
VAE_MARGIN_THRESHOLD = float(os.getenv("VAE_MARGIN_THRESHOLD", "0.08"))
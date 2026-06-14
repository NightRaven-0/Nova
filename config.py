# Loads .env + constants (backend selection, model paths, audio settings)

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Assistant identity
# ---------------------------------------------------------------------------
ASSISTANT_NAME = "Nova"  # Natural-language Oriented Voice Assistant

# ---------------------------------------------------------------------------
# Backend selection  (the main "offline vs cloud" control)
#   STT_BACKEND : "whisper" (offline, GPU) | "vosk" (offline, light)
#   LLM_BACKEND : "ollama"  (offline, GPU) | "openai" (cloud)
#   TTS_BACKEND : "piper"   (offline)      | "elevenlabs" (cloud) | "none" (text only)
# Defaults below are FULLY OFFLINE.
# ---------------------------------------------------------------------------
STT_BACKEND = os.getenv("STT_BACKEND", "whisper").lower()
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper").lower()

# ---------------------------------------------------------------------------
# API keys — only required when the matching cloud backend is selected
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
    raise ValueError("LLM_BACKEND=openai but OPENAI_API_KEY is missing in .env")
if TTS_BACKEND == "elevenlabs" and not ELEVENLABS_API_KEY:
    raise ValueError("TTS_BACKEND=elevenlabs but ELEVENLABS_API_KEY is missing in .env")

# ---------------------------------------------------------------------------
# Local LLM (Ollama, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3:8b")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-en-us-0.15")

# faster-whisper: tiny[.en]/base[.en]/small[.en]/medium[.en]/large-v3
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")        # "cuda" | "cpu"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")  # cpu -> "int8"

# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")  # ElevenLabs voice *ID* hash, not a name
PIPER_MODEL_PATH = os.getenv(
    "PIPER_MODEL_PATH",
    os.path.join("models", "piper", "en_US-amy-medium.onnx"),
)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
MIC_INDEX = os.getenv("MIC_INDEX")  # None -> system default mic; run test_mic.py to list devices
if MIC_INDEX is not None:
    MIC_INDEX = int(MIC_INDEX)
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MEMORY_PATH = os.getenv("MEMORY_PATH", os.path.join("data", "memory.json"))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "20"))           # recent turns kept verbatim
MEMORY_SUMMARIZE_AFTER = int(os.getenv("MEMORY_SUMMARIZE_AFTER", "30"))  # summarize once exceeded

# ---------------------------------------------------------------------------
# Wake word + barge-in (realtime controls — used by the realtime loop)
# ---------------------------------------------------------------------------
USE_WAKE_WORD = str(os.getenv("USE_WAKE_WORD", "0")) == "1"
WAKE_WORD_MODEL = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")  # openWakeWord pretrained name
WAKE_WORD_THRESHOLD = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
USE_BARGE_IN = str(os.getenv("USE_BARGE_IN", "1")) == "1"

# ---------------------------------------------------------------------------
# Derived / legacy flags (kept for backward compatibility)
# ---------------------------------------------------------------------------
USE_CLOUD_LLM = LLM_BACKEND == "openai"
USE_CLOUD_STT = STT_BACKEND == "openai"
USE_TTS = TTS_BACKEND != "none"
LOCAL_ONLY_MODE = str(os.getenv("LOCAL_ONLY_MODE", "0")) == "1"

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

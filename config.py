# Loads .env + constants (API keys, audio settings, assistant name)

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Assistant settings
ASSISTANT_NAME = "Nova"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-en-us-0.15")

# Audio settings (can tweak later)
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

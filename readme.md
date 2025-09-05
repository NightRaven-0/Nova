# Nova – Personal AI Assistant  

Nova is a lightweight personal AI assistant powered by OpenAI and Vosk.  
Currently, Nova can **listen to your voice**, transcribe it to text (via Vosk), and **reply with text responses** (via OpenAI).  
It’s designed to be modular, so more features (like TTS replies or integrations) can be added later.  

## Features  
- 🎤 **Speech-to-Text (STT)** using [Vosk](https://alphacephei.com/vosk/) (offline models).  
- 🤖 **AI-powered replies** using OpenAI’s GPT models.  
- 📝 Text-based responses only (no voice output yet).  
- ⚡ Configurable assistant name and API keys through `.env`.  

## Setup  
1. Clone the repo  
    git clone <https://github.com/NightRaven-0/Nova>
    cd nova

2. Create a .env file:
    OPENAI_API_KEY=your_api_key_here
    
3. List your api models:
    run 'list_models.py'

4. Install dependencies:
    pip install -r requirements.txt
    
5. Download the Vosk model:
    run 'download_model.py'

## Usage

1. Run the assistant:
    run 'python nova.py'

2. Speak into your microphone → Nova will transcribe and reply in text.

Limitations(for now)

Replies are only text (no text-to-speech yet).

Requires internet for GPT responses (offline GPT not supported).

Works best with a clear microphone input.
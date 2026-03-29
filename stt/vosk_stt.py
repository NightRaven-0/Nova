# Recording + transcription (mic → text)

import queue
import sounddevice as sd
import vosk
import json
from config import SAMPLE_RATE, CHUNK_SIZE, VOSK_MODEL_PATH

# -------------------------------
# Load model ONCE (big performance improvement)
# -------------------------------
print("Loading Vosk STT model... (this happens only once)")
model = vosk.Model(VOSK_MODEL_PATH)
rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)

# Recording queue
q = queue.Queue()

def callback(indata, frames, time, status):
    print(" Received audio...")
    if status:
        print(status)
    q.put(bytes(indata))

def listen_and_transcribe():
    """Continuously listen on the mic until user speaks a phrase."""
    mic_index = 3
    with sd.RawInputStream(
        device = mic_index,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        dtype='int16',
        channels=1,
        callback=callback,
    ):
        print("Listening... speak now.")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text.strip():
                    return text
# wake/wake_word.py
# Offline wake-word detection via openWakeWord (ONNX inference).
# Pretrained phrases: hey_jarvis, alexa, hey_mycroft, hey_rhasspy, timer, weather.
# A custom "hey nova" would need training (see openWakeWord's training notebook).

from __future__ import annotations

import numpy as np


class WakeWordDetector:
    """Detects a wake phrase in a stream of 16 kHz int16 audio frames."""

    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        from openwakeword.model import Model

        # ONNX backend avoids the tflite-runtime dependency.
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self.model_name = model_name
        self.threshold = threshold

    def score(self, frame_int16: np.ndarray) -> float:
        """Feed one ~80ms (1280-sample) int16 frame; return the highest wake score."""
        scores = self.model.predict(frame_int16)
        return max(scores.values()) if scores else 0.0

    def triggered(self, frame_int16: np.ndarray) -> bool:
        return self.score(frame_int16) >= self.threshold

    def reset(self) -> None:
        """Clear internal audio buffers (call after a detection)."""
        try:
            self.model.reset()
        except Exception:
            pass

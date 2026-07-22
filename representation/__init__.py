# representation/ — the text pre-processing hook between STT and the brain.
#
# This once hosted a Phase-1 text VAE that tried to clean noisy speech-to-text by
# snapping to a phrase bank. Whisper made that redundant, so the VAE was removed.
# What remains is a passthrough plus a small result type — kept deliberately as
# the clean extension point for a future text cleaner or intent classifier
# (fill in TextResult.intent, or branch inside build_phase1_processor).

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextResult:
    raw_text: str
    cleaned_text: str
    intent: str = ""  # reserved: a future intent classifier can populate this


class PassthroughProcessor:
    """Returns the STT text unchanged."""

    def process(self, raw_text: str) -> TextResult:
        text = (raw_text or "").strip()
        return TextResult(raw_text=text, cleaned_text=text)


def build_phase1_processor():
    """The text pre-processor that sits between STT and the brain. Currently a
    passthrough; swap in a real cleaner/intent step here when there's one worth
    running."""
    return PassthroughProcessor()

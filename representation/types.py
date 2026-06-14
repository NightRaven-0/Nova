from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ReconstructionResult:
    raw_text: str
    cleaned_text: str
    vae_decoded_text: str
    latent_vector: List[float]
    reconstruction_loss: float
    raw_match_score: float
    vae_match_score: float
    used_fallback: bool

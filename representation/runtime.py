from __future__ import annotations

from pathlib import Path

from config import (
    USE_VAE_TEXT_CLEANER,
    VAE_BANK_PATH,
    VAE_LOG_PATH,
    VAE_MODEL_PATH,
    VAE_SENTENCE_MODEL,
    RAW_MATCH_THRESHOLD,
    RAW_MARGIN_THRESHOLD,
    VAE_MATCH_THRESHOLD,
    VAE_MARGIN_THRESHOLD,
)
from .types import ReconstructionResult


class PassthroughProcessor:
    def process(self, raw_text: str) -> ReconstructionResult:
        raw_text = (raw_text or "").strip()
        return ReconstructionResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            vae_decoded_text=raw_text,
            latent_vector=[],
            reconstruction_loss=0.0,
            raw_match_score=1.0,
            vae_match_score=1.0,
            used_fallback=True,
        )


def build_phase1_processor():
    if not USE_VAE_TEXT_CLEANER:
        print("[Phase1] VAE cleaner disabled. Using raw STT text.")
        return PassthroughProcessor()

    model_path = Path(VAE_MODEL_PATH)
    bank_path = Path(VAE_BANK_PATH)

    if not model_path.exists() or not bank_path.exists():
        print("[Phase1] VAE artifacts missing. Using raw STT text.")
        print(f"[Phase1] expected model: {model_path}")
        print(f"[Phase1] expected bank : {bank_path}")
        return PassthroughProcessor()

    try:
        # Imported lazily so the default (VAE-disabled) path never needs torch.
        from .vae_pipeline import VAETextProcessor

        processor = VAETextProcessor(
            model_path=str(model_path),
            bank_path=str(bank_path),
            log_path=VAE_LOG_PATH,
            model_name=VAE_SENTENCE_MODEL,
            raw_similarity_threshold=RAW_MATCH_THRESHOLD,
            raw_margin_threshold=RAW_MARGIN_THRESHOLD,
            vae_similarity_threshold=VAE_MATCH_THRESHOLD,
            vae_margin_threshold=VAE_MARGIN_THRESHOLD,
        )
        print("[Phase1] VAE cleaner loaded successfully.")
        return processor
    except Exception as exc:
        print(f"[Phase1] failed to load VAE cleaner: {exc}")
        print("[Phase1] falling back to raw STT text.")
        return PassthroughProcessor()
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn.functional as F

from .embedder import SentenceEmbedder
from .text_vae import TextEmbeddingVAE, vae_loss


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


class VAETextProcessor:
    def __init__(
        self,
        model_path: str,
        bank_path: str,
        log_path: str = "logs/vae_interactions.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        raw_similarity_threshold: float = 0.78,
        raw_margin_threshold: float = 0.10,
        vae_similarity_threshold: float = 0.75,
        vae_margin_threshold: float = 0.08,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(model_path)
        self.bank_path = Path(bank_path)
        self.log_path = Path(log_path)

        self.raw_similarity_threshold = raw_similarity_threshold
        self.raw_margin_threshold = raw_margin_threshold
        self.vae_similarity_threshold = vae_similarity_threshold
        self.vae_margin_threshold = vae_margin_threshold

        self.embedder = SentenceEmbedder(model_name=model_name)
        self.model = self._load_model()
        self.reconstruction_bank = self._load_bank()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> TextEmbeddingVAE:
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = TextEmbeddingVAE(
            input_dim=checkpoint["config"]["input_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            latent_dim=checkpoint["config"]["latent_dim"],
            dropout=checkpoint["config"].get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _load_bank(self) -> Dict[str, Any]:
        payload = json.loads(self.bank_path.read_text(encoding="utf-8"))
        payload["embeddings"] = np.asarray(payload["embeddings"], dtype=np.float32)
        return payload

    def _nearest(self, embedding: np.ndarray) -> Tuple[str, float, float]:
        bank_embeddings = self.reconstruction_bank["embeddings"]
        bank_texts = self.reconstruction_bank["texts"]

        sims = bank_embeddings @ embedding
        order = np.argsort(sims)[::-1]

        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx

        best_score = float(sims[best_idx])
        second_score = float(sims[second_idx])
        margin = best_score - second_score

        return bank_texts[best_idx], best_score, margin

    def _is_confident(self, score: float, margin: float, score_th: float, margin_th: float) -> bool:
        return score >= score_th and margin >= margin_th

    def process(self, raw_text: str) -> ReconstructionResult:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return ReconstructionResult(
                raw_text="",
                cleaned_text="",
                vae_decoded_text="",
                latent_vector=[],
                reconstruction_loss=0.0,
                raw_match_score=0.0,
                vae_match_score=0.0,
                used_fallback=True,
            )

        raw_embedding = self.embedder.encode_one(raw_text)

        x = torch.from_numpy(raw_embedding).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            _, recon_loss, _ = vae_loss(
                output.reconstruction,
                x,
                output.mu,
                output.logvar,
            )

            reconstructed_embedding = output.reconstruction.squeeze(0)
            reconstructed_embedding = F.normalize(reconstructed_embedding, dim=0)
            reconstructed_embedding = reconstructed_embedding.cpu().numpy().astype(np.float32)

            latent = output.mu.squeeze(0).cpu().numpy().astype(float).tolist()

        raw_best_text, raw_best_score, raw_margin = self._nearest(raw_embedding)
        vae_best_text, vae_best_score, vae_margin = self._nearest(reconstructed_embedding)

        raw_confident = self._is_confident(
            raw_best_score,
            raw_margin,
            self.raw_similarity_threshold,
            self.raw_margin_threshold,
        )

        vae_confident = self._is_confident(
            vae_best_score,
            vae_margin,
            self.vae_similarity_threshold,
            self.vae_margin_threshold,
        )

        # Safe runtime choice:
        # Prefer direct semantic match from the raw text.
        # Only fall back to raw text when confidence is insufficient.
        if raw_confident:
            cleaned_text = raw_best_text
            used_fallback = False
        else:
            cleaned_text = raw_text
            used_fallback = True

        # VAE output is logged, but not trusted for live routing unless you later promote it.
        vae_decoded_text = vae_best_text if vae_confident else raw_text

        result = ReconstructionResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            vae_decoded_text=vae_decoded_text,
            latent_vector=latent,
            reconstruction_loss=float(recon_loss.item()),
            raw_match_score=raw_best_score,
            vae_match_score=vae_best_score,
            used_fallback=used_fallback,
        )

        self._log_result(result)
        return result

    def _log_result(self, result: ReconstructionResult) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
import torch
import torch.nn.functional as F

from .embedder import SentenceEmbedder
from .text_vae import TextEmbeddingVAE, vae_loss


@dataclass
class ReconstructionResult:
    raw_text: str
    reconstructed_text: str
    latent_vector: List[float]
    reconstruction_loss: float
    nearest_score: float
    used_fallback: bool


class VAETextProcessor:
    def __init__(
        self,
        model_path: str,
        bank_path: str,
        log_path: str = "logs/vae_interactions.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        similarity_threshold: float = 0.55,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(model_path)
        self.bank_path = Path(bank_path)
        self.log_path = Path(log_path)
        self.similarity_threshold = similarity_threshold

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

    def process(self, raw_text: str) -> ReconstructionResult:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return ReconstructionResult(
                raw_text="",
                reconstructed_text="",
                latent_vector=[],
                reconstruction_loss=0.0,
                nearest_score=0.0,
                used_fallback=True,
            )

        input_embedding = self.embedder.encode_one(raw_text)
        x = torch.from_numpy(input_embedding).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            _, recon_loss, _ = vae_loss(
                output.reconstruction,
                x,
                output.mu,
                output.logvar,
            )

            reconstructed_embedding = output.reconstruction.squeeze(0)
            reconstructed_embedding = F.normalize(
                reconstructed_embedding,
                dim=0,
            ).cpu().numpy().astype(np.float32)

            latent = output.mu.squeeze(0).cpu().numpy().astype(float).tolist()

        bank_embeddings = self.reconstruction_bank["embeddings"]
        bank_texts = self.reconstruction_bank["texts"]

        sims = bank_embeddings @ reconstructed_embedding
        best_idx = int(np.argmax(sims))
        nearest_score = float(sims[best_idx])

        reconstructed_text = (
            bank_texts[best_idx]
            if nearest_score >= self.similarity_threshold
            else raw_text
        )
        used_fallback = nearest_score < self.similarity_threshold

        result = ReconstructionResult(
            raw_text=raw_text,
            reconstructed_text=reconstructed_text,
            latent_vector=latent,
            reconstruction_loss=float(recon_loss.item()),
            nearest_score=nearest_score,
            used_fallback=used_fallback,
        )

        self._log_result(result)
        return result

    def _log_result(self, result: ReconstructionResult) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
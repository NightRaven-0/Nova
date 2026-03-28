from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, Dataset

from representation.embedder import SentenceEmbedder
from representation.noise import inject_stt_noise
from representation.text_vae import TextEmbeddingVAE, vae_loss


BOOTSTRAP_TEXTS = [
    "open youtube",
    "open google",
    "open spotify",
    "open browser",
    "what time is it",
    "what's the weather like",
    "tell me a joke",
    "summarize this",
    "stop listening",
    "exit the assistant",
    "play some music",
    "search the web for machine learning news",
    "set a timer for ten minutes",
    "what is the capital of finland",
    "write an email draft",
    "explain variational autoencoders simply",
    "what did i just ask you",
    "remember this note for later",
    "open file explorer",
    "restart my pc",
]


class PairDataset(Dataset):
    def __init__(self, noisy_embeddings: np.ndarray, clean_embeddings: np.ndarray):
        self.noisy = torch.from_numpy(noisy_embeddings).float()
        self.clean = torch.from_numpy(clean_embeddings).float()

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def load_texts(corpus_path: Path) -> list[str]:
    texts = []
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    return texts or BOOTSTRAP_TEXTS


def build_pairs(texts: list[str], augmentations_per_text: int = 8):
    clean_texts = []
    noisy_texts = []

    for text in texts:
        clean_texts.append(text)
        noisy_texts.append(text)

        for _ in range(augmentations_per_text):
            clean_texts.append(text)
            noisy_texts.append(inject_stt_noise(text))

    return noisy_texts, clean_texts


def save_bank(path: Path, texts: list[str], embeddings: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"texts": texts, "embeddings": embeddings.tolist()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_losses(
    train_losses: list[float],
    recon_losses: list[float],
    kl_losses: list[float],
    output_path: Path,
):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="total")
    plt.plot(recon_losses, label="recon")
    plt.plot(kl_losses, label="kl")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Nova Phase 1 VAE training")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_latent(latents: np.ndarray, labels: list[str], output_path: Path, method: str):
    if len(latents) < 3:
        return

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(10, max(2, len(latents) - 1)),
            init="random",
            learning_rate="auto",
        )
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label[:30], (reduced[i, 0], reduced[i, 1]), fontsize=8)

    plt.title(f"Latent space ({method.upper()})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/phase1_corpus.txt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-out", default="models/phase1_text_vae.pt")
    parser.add_argument("--bank-out", default="models/phase1_reconstruction_bank.json")
    parser.add_argument("--plots-dir", default="artifacts/phase1")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    texts = load_texts(Path(args.corpus))
    noisy_texts, clean_texts = build_pairs(texts)

    embedder = SentenceEmbedder()
    noisy_embeddings = embedder.encode(noisy_texts)
    clean_embeddings = embedder.encode(clean_texts)

    dataset = PairDataset(noisy_embeddings, clean_embeddings)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TextEmbeddingVAE(
        input_dim=noisy_embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, recon_losses, kl_losses = [], [], []

    model.train()
    for epoch in range(args.epochs):
        total_epoch_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for noisy_batch, clean_batch in loader:
            optimizer.zero_grad()
            output = model(noisy_batch)
            loss, recon, kl = vae_loss(
                output.reconstruction,
                clean_batch,
                output.mu,
                output.logvar,
                beta=args.beta,
            )
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_recon_loss += recon.item()
            total_kl_loss += kl.item()

        num_batches = max(1, len(loader))
        train_losses.append(total_epoch_loss / num_batches)
        recon_losses.append(total_recon_loss / num_batches)
        kl_losses.append(total_kl_loss / num_batches)

        print(
            f"epoch={epoch + 1:03d} "
            f"total={train_losses[-1]:.6f} "
            f"recon={recon_losses[-1]:.6f} "
            f"kl={kl_losses[-1]:.6f}"
        )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": noisy_embeddings.shape[1],
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "dropout": 0.1,
            },
        },
        args.model_out,
    )

    clean_bank_embeddings = embedder.encode(texts)
    save_bank(Path(args.bank_out), texts, clean_bank_embeddings)

    plot_losses(
        train_losses,
        recon_losses,
        kl_losses,
        Path(args.plots_dir) / "loss_curve.png",
    )

    model.eval()
    with torch.no_grad():
        clean_tensor = torch.from_numpy(clean_bank_embeddings).float()
        outputs = model(clean_tensor)
        latents = outputs.mu.numpy()

    plot_latent(latents, texts, Path(args.plots_dir) / "latent_pca.png", method="pca")
    plot_latent(latents, texts, Path(args.plots_dir) / "latent_tsne.png", method="tsne")

    print(f"saved model -> {args.model_out}")
    print(f"saved bank  -> {args.bank_out}")
    print(f"plots dir   -> {args.plots_dir}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
import random

from representation.noise import inject_stt_noise
from representation.vae_pipeline import VAETextProcessor


def load_texts(corpus_path: Path) -> list[str]:
    if corpus_path.exists():
        return [
            line.strip()
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/phase1_corpus.txt")
    parser.add_argument("--model", default="models/phase1_text_vae.pt")
    parser.add_argument("--bank", default="models/phase1_reconstruction_bank.json")
    parser.add_argument("--samples-per-text", type=int, default=5)
    args = parser.parse_args()

    texts = load_texts(Path(args.corpus))
    if not texts:
        raise ValueError(
            "No corpus found. Add lines to data/phase1_corpus.txt "
            "or train once with bootstrap texts."
        )

    processor = VAETextProcessor(model_path=args.model, bank_path=args.bank)

    total = 0
    exact = 0
    fallbacks = 0
    examples = []

    random.seed(42)

    for clean_text in texts:
        for _ in range(args.samples_per_text):
            noisy = inject_stt_noise(clean_text)
            result = processor.process(noisy)

            total += 1

            # ✅ Use cleaned_text as primary output
            if result.cleaned_text == clean_text:
                exact += 1

            if result.used_fallback:
                fallbacks += 1

            if len(examples) < 10:
                examples.append(
                    (
                        noisy,
                        clean_text,
                        result.cleaned_text,
                        result.raw_match_score,
                        result.vae_match_score,
                    )
                )

    accuracy = exact / max(1, total)
    fallback_rate = fallbacks / max(1, total)

    print(f"eval_samples={total}")
    print(f"exact_match_accuracy={accuracy:.4f}")
    print(f"fallback_rate={fallback_rate:.4f}")
    print("examples:")

    for noisy, clean, recon, raw_score, vae_score in examples:
        print(f"  noisy={noisy!r}")
        print(f"  clean={clean!r}")
        print(f"  cleaned={recon!r}")
        print(f"  raw_score={raw_score:.4f} vae_score={vae_score:.4f}")
        print("  ---")


if __name__ == "__main__":
    main()
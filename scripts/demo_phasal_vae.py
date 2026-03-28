from __future__ import annotations

import argparse

from representation.vae_pipeline import VAETextProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phase1_text_vae.pt")
    parser.add_argument("--bank", default="models/phase1_reconstruction_bank.json")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    processor = VAETextProcessor(model_path=args.model, bank_path=args.bank)
    result = processor.process(args.text)

    print("raw_text:", result.raw_text)
    print("reconstructed_text:", result.reconstructed_text)
    print("reconstruction_loss:", result.reconstruction_loss)
    print("nearest_score:", result.nearest_score)
    print("latent_vector[:8]:", result.latent_vector[:8])
    print("used_fallback:", result.used_fallback)


if __name__ == "__main__":
    main()
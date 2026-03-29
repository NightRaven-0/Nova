from __future__ import annotations

import argparse
from unittest import result

from representation.vae_pipeline import VAETextProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/phase1_text_vae.pt")
    parser.add_argument("--bank", default="models/phase1_reconstruction_bank.json")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    processor = VAETextProcessor(model_path=args.model, bank_path=args.bank)
    result = processor.process(args.text)

    print("[input]     ", result.raw_text)
    print("[cleaned]   ", result.cleaned_text)
    print("[vae]       ", result.vae_decoded_text)
    print("[loss]      ", result.reconstruction_loss)
    print("[raw score] ", result.raw_match_score)
    print("[vae score] ", result.vae_match_score)
    print("[fallback]  ", result.used_fallback)

if __name__ == "__main__":
    main()
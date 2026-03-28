from __future__ import annotations

import random

FILLERS = ["uh", "um", "like", "you know"]


def inject_stt_noise(text: str) -> str:
    words = text.split()
    if not words:
        return text

    noisy = words[:]

    if len(noisy) > 3 and random.random() < 0.35:
        idx = random.randrange(len(noisy))
        noisy.pop(idx)

    if len(noisy) > 3 and random.random() < 0.25:
        idx = random.randrange(len(noisy) - 1)
        noisy[idx], noisy[idx + 1] = noisy[idx + 1], noisy[idx]

    if random.random() < 0.25:
        idx = random.randrange(len(noisy) + 1)
        noisy.insert(idx, random.choice(FILLERS))

    if random.random() < 0.20:
        noisy = [w[:-1] if len(w) > 4 and random.random() < 0.2 else w for w in noisy]

    return " ".join([w for w in noisy if w]).strip()
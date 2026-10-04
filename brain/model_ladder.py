# brain/model_ladder.py
# Pick the biggest brain that fits the VRAM actually free right now, and step
# down automatically when something else wants the GPU (a game, usually).
#
# The ladder is configured biggest-first as "tag:min_free_gb" in MODEL_LADDER.
# min_free_gb is the least free VRAM at which that model is worth running, NOT
# its file size: qwen3.6:35b-a3b is a mixture-of-experts model with only ~3B
# parameters active per token, so it stays usable with part of it in system RAM,
# while the dense models want to fit entirely in VRAM.
#
# Checks cost one nvidia-smi call (~100 ms), so they run between turns (on wake,
# after a reply) and never in front of one.

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from typing import List, Optional, Tuple

from config import (
    LLM_BACKEND,
    MODEL_LADDER,
    MODEL_LADDER_CHECK_S,
    OLLAMA_BASE_URL,
    USE_MODEL_LADDER,
)

_UP_MARGIN_GB = 1.0   # need this much *extra* headroom before stepping up
_COOLDOWN_S = 20.0    # never switch more often than this (anti-flapping)

_state = {"current": None, "last_switch": 0.0, "last_check": 0.0, "pinned": False,
          "installed": None, "warned": False}


def _ollama_root() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return (base[:-3] if base.endswith("/v1") else base).rstrip("/")


def _api(path: str, payload: Optional[dict] = None, timeout: float = 10.0):
    url = _ollama_root() + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers),
                                timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def ladder() -> List[Tuple[str, float]]:
    """[(tag, min_free_gb)] biggest first, as configured."""
    out = []
    for item in (MODEL_LADDER or "").split(","):
        item = item.strip()
        if not item:
            continue
        tag, _, gb = item.rpartition(":")
        try:
            out.append((tag.strip(), float(gb)))
        except ValueError:
            continue  # a tag without a threshold is a config typo; skip it
    return out


def installed_tags() -> set:
    """Model tags Ollama actually has pulled (cached; a pull mid-session is rare)."""
    if _state["installed"] is None:
        try:
            tags = _api("/api/tags").get("models", [])
            _state["installed"] = {m.get("name", "") for m in tags} | \
                                  {m.get("name", "").split(":")[0] for m in tags}
        except Exception:
            _state["installed"] = set()
    return _state["installed"]


def free_vram_gb() -> Optional[float]:
    """Free VRAM per nvidia-smi, or None if there's no NVIDIA GPU to ask."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        pass
    return None


def _our_vram_gb() -> float:
    """VRAM currently held by a loaded Nova model — ours to reuse, so it counts
    as available when sizing the next one."""
    try:
        tags = {t for t, _ in ladder()}
        loaded = _api("/api/ps").get("models", [])
        return sum(m.get("size_vram", 0) for m in loaded
                   if m.get("name", "") in tags) / 1e9
    except Exception:
        return 0.0


def choose() -> Optional[str]:
    """The best installed model for the VRAM available right now."""
    rungs = [(tag, gb) for tag, gb in ladder() if tag in installed_tags()]
    if not rungs:
        return None
    free = free_vram_gb()
    if free is None:
        return None  # no GPU telemetry: leave the configured model alone
    available = free + _our_vram_gb()
    current = _state["current"]
    for tag, need in rungs:
        # Stepping UP needs extra headroom so a borderline reading can't flap.
        going_up = current is not None and tag != current and \
            [t for t, _ in rungs].index(tag) < [t for t, _ in rungs].index(current)
        if available >= need + (_UP_MARGIN_GB if going_up else 0.0):
            return tag
    return rungs[-1][0]  # nothing fits: the smallest rung is the best we can do


def unload(tag: str) -> None:
    """Drop a model from VRAM immediately (keep_alive 0) so a game can have it."""
    try:
        _api("/api/generate", {"model": tag, "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def note_manual_switch(tag: str) -> None:
    """The user picked a model out loud — stop second-guessing them this session."""
    _state.update(current=tag, pinned=True, last_switch=time.time())


def unpin() -> None:
    _state["pinned"] = False


def apply(force: bool = False, on_switch=None, warm: bool = True) -> Optional[str]:
    """Switch to the best-fitting model if that isn't the one we're on.
    Returns the tag now in use, or None if the ladder is off/not applicable."""
    if not (USE_MODEL_LADDER and LLM_BACKEND == "ollama") or _state["pinned"]:
        return None
    now = time.time()
    if not force and now - _state["last_check"] < MODEL_LADDER_CHECK_S:
        return _state["current"]
    _state["last_check"] = now

    from brain.gpt_llm import get_active_model, set_active_model, warm_up_model

    if _state["current"] is None:
        _state["current"] = get_active_model()
    want = choose()
    if not want:
        if not _state["warned"] and free_vram_gb() is None:
            print("[ladder] no GPU telemetry (nvidia-smi) — staying on the configured model")
            _state["warned"] = True
        return _state["current"]
    if want == _state["current"]:
        return want
    if not force and now - _state["last_switch"] < _COOLDOWN_S:
        return _state["current"]

    old = _state["current"]
    set_active_model(want)
    _state.update(current=want, last_switch=now)
    if old and old != want:
        unload(old)      # free the VRAM the old one was holding
    if warm:
        warm_up_model()  # start loading the new one in the background
    if on_switch:
        on_switch(old, want)
    return want

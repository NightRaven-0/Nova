#!/usr/bin/env python
"""Compare local models for Nova.

Measures what actually matters for a voice assistant:
  * time to the FIRST SPOKEN SENTENCE (not just the first token)
  * generation speed (approximate tokens/sec)
  * whether a spoken command makes the model CALL its tool instead of narrating
    it ("Opening Steam!" with no tool call is the failure we care about), and
    how much hidden reasoning it burns getting there
  * how much of the model actually fits in VRAM

Tool calls are RECORDED, NEVER EXECUTED, so benchmarking never opens an app or
sets a real reminder.

    python scripts/bench_models.py                  # every installed ladder model
    python scripts/bench_models.py qwen3:8b qwen3:14b --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from config import OLLAMA_BASE_URL
from brain.gpt_llm import SYSTEM_PROMPT, SentenceStream, _NO_THINKING
from brain.fastpath import route_command
from brain.tools import TOOLS

# Command phrasings deliberately too fuzzy for the instant-command router, so
# they exercise the model's own tool-calling (checked at startup).
COMMANDS = [
    ("i feel like playing something, get steam going", {"open_application"}),
    ("what is the weather looking like in delhi", {"web_search"}),
    ("it is way too loud, bring it down a bit", {"media_control"}),
    ("my eggs need flipping in three minutes, give me a shout", {"set_reminder"}),
    ("remember that the router password is swordfish", {"add_note"}),
    ("any idea what the time is where i am", {"get_current_time"}),
    ("how much memory am i using right now", {"vibe_check"}),
    ("i need to step away, secure my screen", {"power_control"}),
    ("put some music on youtube for me", {"open_website", "open_application"}),
    ("kill the music", {"media_control"}),
    ("what can you actually do for me", {"list_capabilities"}),
    ("i am about to game, drop to the small brain", {"switch_model"}),
]

CHATS = [
    "tell me three fun facts about octopuses",
    "why is the sky blue? keep it short",
    "what is a good name for a pet seagull?",
    "settle this: is a hot dog a sandwich?",
]


def ollama_root() -> str:
    base = OLLAMA_BASE_URL.rstrip("/")
    return (base[:-3] if base.endswith("/v1") else base).rstrip("/")


def api(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(ollama_root() + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def installed():
    return [m.get("name") for m in api("/api/tags").get("models", [])]


def run_turn(client, model, text, think):
    """One request. Returns timings, the tool called (if any), and sizes."""
    t0 = time.time()
    kwargs = dict(model=model, temperature=0.3, stream=True, tools=TOOLS, tool_choice="auto",
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text}])
    if not think:
        kwargs["extra_body"] = _NO_THINKING
    stream = client.chat.completions.create(**kwargs)
    sentences = SentenceStream()
    first_token = first_sentence = None
    chars = reasoning = 0
    tool = None
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            extra = getattr(d, "model_extra", None) or {}
            reasoning += len(extra.get("reasoning") or extra.get("reasoning_content") or "")
            if d.content:
                chars += len(d.content)
                if first_token is None:
                    first_token = time.time() - t0
                if sentences.feed(d.content) and first_sentence is None:
                    first_sentence = time.time() - t0
            for tc in d.tool_calls or []:
                if tool is None and tc.function is not None:
                    tool = tc.function.name
                if first_token is None:
                    first_token = time.time() - t0
    finally:
        stream.close()
    if sentences.flush() and first_sentence is None:
        first_sentence = time.time() - t0
    total = time.time() - t0
    gen_window = max(total - (first_token or total), 1e-6)
    return {"total": total, "first_token": first_token, "first_sentence": first_sentence,
            "tok_s": (chars / 4) / gen_window, "reasoning": reasoning,
            "tool": tool, "chars": chars}


def throughput(model):
    """True generation speed, from Ollama's own eval counters. (Dividing an
    estimated token count by the post-first-token window, as the streaming path
    would, is pure noise on short replies.) Also reports how long Ollama takes
    to read a realistic Nova-sized prompt, which is what you actually wait for."""
    body = {"model": model, "stream": False, "think": False,
            "options": {"temperature": 0.3},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": "Write about 200 words describing a "
                                                     "thunderstorm rolling over a coastal town."}]}
    r = api("/api/chat", body)
    ns = 1e9
    gen, gen_s = r.get("eval_count", 0), r.get("eval_duration", 1) / ns
    return {"tok_s": gen / max(gen_s, 1e-6), "gen_tokens": gen,
            "prompt_tokens": r.get("prompt_eval_count", 0),
            "prompt_s": r.get("prompt_eval_duration", 0) / ns}


def bench(model, repeats):
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    print()
    print("=== " + model)
    t0 = time.time()
    api("/api/generate", {"model": model})  # load into VRAM, no generation
    load = time.time() - t0
    ps = next((m for m in api("/api/ps").get("models", []) if m.get("name") == model), {})
    size = ps.get("size", 0) / 1e9
    vram = ps.get("size_vram", 0) / 1e9
    spill = "" if size <= vram + 0.05 else "  ({:.1f} GB spilled to RAM)".format(size - vram)
    print("  loaded in {:.1f}s | {:.1f} GB total, {:.1f} GB in VRAM{} | context {}".format(
        load, size, vram, spill, ps.get("context_length", "?")))

    tp = throughput(model)
    print("  speed: {:.0f} tok/s generating, and {:.2f}s to read a {}-token prompt".format(
        tp["tok_s"], tp["prompt_s"], tp["prompt_tokens"]))

    chat_runs = [run_turn(client, model, c, think=False) for c in CHATS]
    fs = [r["first_sentence"] for r in chat_runs if r["first_sentence"]]
    print("  chat: first sentence {:.2f}s median ({:.2f}-{:.2f}s), ~{:.0f} tok/s, "
          "reasoning {} chars".format(
              statistics.median(fs), min(fs), max(fs),
              statistics.median([r["tok_s"] for r in chat_runs]),
              sum(r["reasoning"] for r in chat_runs)))

    hits = 0
    misses = []
    cmd_runs = []
    for text, expected in COMMANDS:
        for _ in range(repeats):
            r = run_turn(client, model, text, think=True)
            cmd_runs.append(r)
            if r["tool"] in expected:
                hits += 1
            else:
                misses.append([text, r["tool"] or "NO TOOL CALL"])
    attempts = len(COMMANDS) * repeats
    cfs = [r["first_sentence"] or r["total"] for r in cmd_runs]
    print("  commands: tools called correctly {}/{} ({:.0f}%), first response {:.2f}s median, "
          "reasoning {:.0f} chars median".format(
              hits, attempts, 100.0 * hits / attempts, statistics.median(cfs),
              statistics.median([r["reasoning"] for r in cmd_runs])))
    for text, got in misses[:8]:
        print("    miss: {!r} -> {}".format(text, got))

    api("/api/generate", {"model": model, "keep_alive": 0})  # free the VRAM
    return {"model": model, "load_s": load, "size_gb": size, "vram_gb": vram,
            "context": ps.get("context_length"), "throughput": tp, "chat": chat_runs,
            "commands": {"hits": hits, "attempts": attempts, "runs": cmd_runs,
                         "misses": misses}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help="Ollama tags (default: installed ladder models)")
    p.add_argument("--repeats", type=int, default=2, help="runs per command (default 2)")
    args = p.parse_args()

    have = installed()
    if args.models:
        models = [m for m in args.models if m in have] or args.models
    else:
        from config import MODEL_LADDER
        want = [e.rpartition(":")[0] for e in MODEL_LADDER.split(",") if e.strip()]
        models = [m for m in want if m in have]
    print("installed: " + ", ".join(have))
    print("benchmarking: {}  ({} run(s) per command)".format(", ".join(models), args.repeats))

    leaks = [t for t, _ in COMMANDS if route_command(t) is not None]
    if leaks:
        print("WARNING: the instant-command router already handles {} — "
              "those test the router, not the model".format(leaks))

    results = [bench(m, args.repeats) for m in models]
    os.makedirs("logs", exist_ok=True)
    out = os.path.join("logs", "bench-{:%Y-%m-%d-%H%M}.json".format(datetime.now()))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("summary")
    print("  {:22s} {:>9s} {:>6s} {:>8s} {:>9s} {:>8s}".format(
        "model", "chat 1st", "tok/s", "cmd 1st", "tools ok", "VRAM"))
    for r in results:
        fs = [c["first_sentence"] for c in r["chat"] if c["first_sentence"]]
        cfs = [c["first_sentence"] or c["total"] for c in r["commands"]["runs"]]
        print("  {:22s} {:8.2f}s {:6.0f} {:7.2f}s {:4d}/{:<4d} {:5.1f} GB".format(
            r["model"], statistics.median(fs),
            r["throughput"]["tok_s"],
            statistics.median(cfs), r["commands"]["hits"], r["commands"]["attempts"],
            r["vram_gb"]))
    print("  saved " + out)


if __name__ == "__main__":
    main()

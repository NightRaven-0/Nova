# Nova Roadmap — toward a full-fledged offline AI assistant ("like Claude, but it listens")

Nova = **N**atural-language **O**riented **V**oice **A**ssistant.
Vision: a private, offline assistant that hears you, talks back, *does real things* on
your PC (open apps, message people, read email), and gets better the more you use it.

Status legend:  ✅ done   🔨 in progress   ⬜ planned

---

## Phase 0 — Foundation ✅ (done 2026-06-14)
The full offline voice loop, on your hardware.
- Offline STT (faster-whisper, GPU), LLM (Ollama + qwen3:8b), TTS (Piper).
- Function-calling brain with a tool registry; conversation memory that persists + self-summarizes.
- Wake word (openWakeWord) + barge-in realtime loop.
**Done when:** you can hold a spoken conversation and it can open sites/apps. ✅

## Phase 1 — Nova learns from you  (RAG correction memory) 🔨 NEXT
The cheapest, safest "it learns" win — no model training.
- Capture feedback every turn (what you said → what it did → was it right?).
- When you correct it, embed the situation + correction (reuse the existing sentence-transformer).
- Retrieve similar past corrections and inject them into the prompt next time.
**Done when:** you correct a misunderstanding once and Nova stops repeating it.

## Phase 2 — Real actions: a proper skill system ✅ (core done 2026-06-26)
The hardcoded `tools.py` is now an extensible **skills** framework — the base for everything below.
- ✅ Each skill = schema + impl + examples, auto-discovered (`skills/` package, `@skill` decorator); `brain/tools.py` is a thin shim.
- ✅ Hardened the `shell=True` app-launch — no more shell-injection (allowlisted URIs + PATH-resolved exe, shell=False).
- ✅ Windows media/volume keys (`media_control`). Typing, clipboard, and window management are now easy drop-in skills (next).
**Done:** adding a capability = dropping one file in `skills/`.

## Phase 3 — Connected services (the "do things for me" leap) ⬜
Real integrations, built as Phase-2 skills.
- **Email**: read/triage/summarize, draft replies (Gmail API or IMAP).
- **Discord**: read & send DMs/messages (bot API or local automation).
- **Calendar / reminders / notifications**.
**Done when:** "read my unread emails" and "message Alex on Discord" work end to end.

## Phase 4 — Computer-use / on-screen agency ⬜
For apps with no API: let Nova *see* the screen (vision model) and act (click/type).
- Screenshot → reason → act loop, with confirmations on risky steps.
**Done when:** Nova completes a multi-step GUI task you describe out loud.

## Phase 5 — Self-improvement ⬜
The real "learns from its mistakes" + safe "writes its own code".
- Periodic **DPO + LoRA** fine-tune of qwen3:8b from collected preference pairs.
- **Self-extending skills**: on a capability gap, Nova drafts a new skill → you approve → hot-load.
**Done when:** Nova proposes and (after approval) gains a new skill by itself.

## Phase 6 — Proactivity ⬜
Background agent that anticipates instead of only reacting.
- Watches email/calendar/system, notifies, runs scheduled routines.
**Done when:** "Good morning — 3 meetings today and 2 emails need replies," unprompted.

---

## Cross-cutting (ongoing)
- **Safety/guardrails**: confirm destructive or outward-facing actions; allowlists. Grows with each phase.
- **Custom "hey Nova" wake word**: trained openWakeWord model (see `scripts/wakeword/README.md`).
- **Personality & voice**: tune the system prompt and pick/clone a voice.

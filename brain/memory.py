# brain/memory.py
# Conversation memory: keeps recent turns verbatim, persists to disk so Nova
# remembers across restarts, and rolls older turns into a running summary so
# the context window never overflows.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Dict, Optional


class ConversationMemory:
    def __init__(
        self,
        path: str,
        system_prompt: str,
        max_turns: int = 20,
        summarize_after: int = 30,
        summarize_fn: Optional[Callable[[List[Dict]], str]] = None,
    ):
        self.path = Path(path)
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.summarize_after = summarize_after
        self.summarize_fn = summarize_fn

        self.summary: str = ""
        self.messages: List[Dict[str, str]] = []  # {"role": "user"|"assistant", "content": ...}
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.summary = data.get("summary", "")
            self.messages = data.get("messages", [])
        except Exception as e:
            print(f"[memory] could not load {self.path}: {e}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary, "messages": self.messages}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- mutation ----------------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self.summary = ""
        self.messages = []
        self.save()

    # ---- read for the LLM --------------------------------------------------
    def build_messages(self) -> List[Dict[str, str]]:
        """System prompt + rolling summary + recent turns, ready for the API."""
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            msgs.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier conversation:\n{self.summary}",
                }
            )
        msgs.extend(self.messages)
        return msgs

    # ---- compaction --------------------------------------------------------
    def maybe_summarize(self) -> None:
        """Once history grows past summarize_after, fold the oldest turns into
        the running summary and keep only the most recent max_turns verbatim."""
        if self.summarize_fn is None:
            return
        if len(self.messages) <= self.summarize_after:
            return

        keep = self.messages[-self.max_turns:]
        older = self.messages[: -self.max_turns]

        to_summarize: List[Dict[str, str]] = []
        if self.summary:
            to_summarize.append(
                {"role": "system", "content": f"Existing summary:\n{self.summary}"}
            )
        to_summarize.extend(older)

        try:
            self.summary = self.summarize_fn(to_summarize)
            self.messages = keep
        except Exception as e:
            print(f"[memory] summarize failed, keeping full history: {e}")

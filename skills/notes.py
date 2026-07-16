# skills/notes.py — quick voice notes: jot things down, read them back, clear them.

import json
import time
from pathlib import Path

from skills.registry import skill

_NOTES_FILE = Path(__file__).resolve().parent.parent / "data" / "notes.json"


def _load() -> list:
    try:
        return json.loads(_NOTES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: list) -> None:
    _NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _NOTES_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


@skill(
    name="add_note",
    description=(
        "Save a quick note for the user. Use when they say 'note that X', 'jot down X', "
        "'add X to my list', or 'remember to buy X' (a note, not a timed reminder)."
    ),
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The note content."}},
        "required": ["text"],
    },
    examples=["note that the wifi password is hunter2", "add milk to my list"],
)
def add_note(args: dict) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return "There was nothing to note down."
    items = _load()
    items.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "text": text})
    _save(items)
    return f"Noted: {text}."


@skill(
    name="read_notes",
    description="Read back the user's saved notes/list.",
    parameters={"type": "object", "properties": {}},
    examples=["read my notes", "what's on my list"],
)
def read_notes(_args: dict) -> str:
    items = _load()
    if not items:
        return "Your notes are empty."
    lines = [f"{i}. {n['text']}" for i, n in enumerate(items, 1)]
    return f"You have {len(items)} note{'s' if len(items) != 1 else ''}: " + " ".join(lines)


@skill(
    name="clear_notes",
    description=(
        "Delete notes. Pass 'matching' with a keyword to delete specific notes, or omit "
        "it to clear the whole list."
    ),
    parameters={
        "type": "object",
        "properties": {
            "matching": {"type": "string", "description": "Keyword to match; empty = clear all."},
        },
    },
    examples=["clear my notes", "remove the milk note"],
)
def clear_notes(args: dict) -> str:
    match = (args.get("matching") or "").strip().lower()
    items = _load()
    keep = [n for n in items if match and match not in n["text"].lower()]
    removed = len(items) - len(keep)
    _save(keep)
    if removed == 0:
        return "Nothing matched — your notes are unchanged."
    return f"Removed {removed} note{'s' if removed != 1 else ''}."

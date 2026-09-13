# brain/fastpath.py
# Instant commands. Everyday requests like "open steam", "what time is it",
# "pause the music" and "set a timer for 5 minutes" are matched here and run
# straight through their skill, with no LLM round-trip. That makes them instant,
# and it means they can't be narrated-but-not-done (qwen3 saying "opening Steam"
# without ever calling the tool). Matching is strict on purpose: the WHOLE
# utterance must fit a pattern, and an app/site must really resolve. Anything
# else returns None and goes to the LLM exactly as before.

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

import skills  # noqa: F401  (importing the package registers every skill)
from skills.registry import dispatch

Reply = Tuple[str, bool]  # (what Nova says, whether to keep the turn in memory)

# ---------------------------------------------------------------------------
# Normalisation: "Nova, can you open Steam, please?" -> "open steam"
# ---------------------------------------------------------------------------
_LEAD = re.compile(
    r"^(?:(?:hey|ok|okay)\s+)?nova\s+"
    r"|^(?:please|can you|could you|would you|will you|go ahead and|just)\s+"
)
_TRAIL = re.compile(r"\s+(?:please|for me|right now|now|nova|thanks|thank you)$")


def _normalize(text: str) -> str:
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[^\w\s'.:-]", " ", t)  # drop , ? ! etc; keep what's / reddit.com / 18:00
    t = re.sub(r"\s+", " ", t).strip(" .")
    while True:
        s = _TRAIL.sub("", _LEAD.sub("", t)).strip(" .")
        if s == t:
            return t
        t = s


# ---------------------------------------------------------------------------
# Durations: "5 minutes", "an hour", "half an hour", "ninety seconds"
# ---------------------------------------------------------------------------
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "twenty five": 25,
    "twenty-five": 25, "thirty": 30, "forty": 40, "forty five": 45,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "half a": 0.5, "half an": 0.5,
}
_NUM = (r"(?P<n>\d+(?:\.\d+)?|"
        + "|".join(re.escape(w) for w in sorted(_NUMBER_WORDS, key=len, reverse=True))
        + ")")
_UNIT = r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)"
_UNIT_IN_MINUTES = {"s": 1 / 60, "m": 1, "h": 60}


def _minutes(m: re.Match) -> float:
    n = m.group("n")
    value = float(n) if n[0].isdigit() else _NUMBER_WORDS[n]
    return value * _UNIT_IN_MINUTES[m.group("unit")[0]]


def _pretty(minutes: float) -> str:
    if minutes < 1:
        return f"{round(minutes * 60)} second"
    if minutes < 60 or minutes % 60:
        return f"{minutes:g} minute"
    return f"{minutes / 60:g} hour"


# ---------------------------------------------------------------------------
# Handlers: (match, raw_text) -> (reply, remember) or None to fall through
# ---------------------------------------------------------------------------
def _say(skill_name: str, args: Optional[dict] = None) -> str:
    return dispatch(skill_name, args or {})


def _timer(m, _raw) -> Reply:
    minutes = _minutes(m)
    return _say("set_reminder", {"minutes": minutes,
                                 "text": f"your {_pretty(minutes)} timer is done"}), True


def _remind(m, _raw) -> Reply:
    return _say("set_reminder", {"minutes": _minutes(m), "text": m.group("text")}), True


_NOTE_RAW = re.compile(
    r"(?:take a note|make a note|add a note|note down|jot down|write down|note that)"
    r"(?:\s+that)?[\s:,]+(.+?)[\s.!?]*$",
    re.IGNORECASE,
)


def _note(m, raw) -> Reply:
    # Take the note from the raw transcript so capitals/punctuation survive
    # (passwords, names); normalisation lowercases everything.
    found = _NOTE_RAW.search(raw or "")
    return _say("add_note", {"text": found.group(1) if found else m.group("text")}), True


def _media(action: str, presses: int = 1, reply: str = "") -> Callable:
    def run(_m, _raw) -> Reply:
        out = ""
        for _ in range(presses):  # one volume key = 2%, so "turn it up" presses a few
            out = _say("media_control", {"action": action})
        return (reply or out), False
    return run


def _skill(name: str, args: Optional[dict] = None, remember: bool = False) -> Callable:
    return lambda _m, _raw: (_say(name, args), remember)


def _open(m, _raw) -> Optional[Reply]:
    from skills.apps import _APPS, _find_shortcut
    from skills.web import _WEBSITES

    target = re.sub(r"^(?:(?:up|the|my)\s+)+", "", m.group("target"))
    target = re.sub(r"\s+(?:app|application|website|site)$", "", target)
    if target in _APPS:
        return _say("open_application", {"app": target}), True
    if target in _WEBSITES:
        return _say("open_website", {"site": target}), True
    if re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:/\S*)?", target):
        return _say("open_website", {"site": target}), True
    # Installed app: only if a Start Menu shortcut contains the target as whole
    # words ("code" -> Visual Studio Code), so "open the pod bay doors" goes to
    # the LLM instead of launching something random.
    shortcut = _find_shortcut(target)
    if shortcut:
        name = os.path.splitext(os.path.basename(shortcut))[0].lower()
        if re.search(rf"\b{re.escape(target)}\b", name):
            return _say("open_application", {"app": target}), True
    return None


_TIME = (
    r"what(?:'?s| is) the (?:time|date|day)(?: today| now| right now)?"
    r"|what time is it(?: now| right now)?|what day is (?:it|today)"
    r"|what(?:'?s| is) (?:today'?s date|the date today)"
    r"|(?:tell me )?the (?:time|date)|time check|current time"
)

_RULES: List[Tuple[str, Callable]] = [
    # --- timers & reminders (before "open/start", which would grab "start a timer")
    (rf"(?:(?:set|start) )?(?:a |an )?(?:timer|alarm) for {_NUM} {_UNIT}", _timer),
    (rf"(?:(?:set|start) )?(?:a |an )?{_NUM}[ -]{_UNIT} (?:timer|alarm)", _timer),
    (rf"remind me in {_NUM} {_UNIT} to (?P<text>.+)", _remind),
    (rf"remind me to (?P<text>.+?) in {_NUM} {_UNIT}", _remind),
    (rf"in {_NUM} {_UNIT} remind me to (?P<text>.+)", _remind),
    (r"(?:list|read|show)(?: me)? my (?:reminders|timers)"
     r"|what (?:reminders|timers) do i have(?: set)?|do i have any (?:reminders|timers)",
     _skill("list_reminders")),
    (r"cancel all (?:my |the )?(?:reminders|timers)", _skill("cancel_reminders", remember=True)),
    # --- clock
    (_TIME, _skill("get_current_time")),
    # --- media & volume
    (r"(?:pause|resume|unpause|play|stop)(?: the| my| this)? (?:music|song|track|video|media|playback)"
     r"|resume|unpause|play", _media("play_pause")),
    (r"(?:next|skip)(?: the| this)?(?: song| track| video)?|(?:play )?the next (?:song|track|video)"
     r"|skip (?:it|this)", _media("next")),
    (r"(?:previous|prev)(?: song| track| video)?"
     r"|(?:play |go back to )?the (?:previous|last) (?:song|track|video)", _media("previous")),
    (r"volume up|louder|turn (?:it|the volume|the sound|the music) up"
     r"|turn up the (?:volume|sound|music)|(?:raise|increase) the volume",
     _media("volume_up", 5, "Turned it up.")),
    (r"volume down|quieter|softer|turn (?:it|the volume|the sound|the music) down"
     r"|turn down the (?:volume|sound|music)|(?:lower|decrease|reduce) the volume",
     _media("volume_down", 5, "Turned it down.")),
    (r"(?:mute|unmute)(?: it| the sound| the audio| the volume| the music| sound| audio)?",
     _media("mute")),
    # --- notes
    (r"(?:take a note|make a note|add a note|note down|jot down|write down|note that)"
     r"(?: that)?:? (?P<text>.+)", _note),
    (r"(?:read|list|show)(?: me)? my notes|what(?:'?s| is) (?:on|in) my (?:list|notes)"
     r"|what are my notes", _skill("read_notes")),
    # --- power: only the harmless ones; shutdown/restart stay with the LLM
    (r"lock (?:my |the )?(?:pc|computer|screen|laptop|workstation)|lock it",
     _skill("power_control", {"action": "lock"})),
    (r"(?:cancel|abort|stop) (?:the )?(?:shutdown|restart)",
     _skill("power_control", {"action": "cancel"})),
    # --- open apps & sites (last: only fires if the target really resolves)
    (r"(?:open|launch|start|run|pull up|bring up|fire up) (?P<target>.+)", _open),
]
_COMPILED = [(re.compile(pattern), handler) for pattern, handler in _RULES]


def route_command(text: str) -> Optional[Reply]:
    """If `text` is a clear everyday command, run it now and return
    (reply, remember). Otherwise return None so the LLM handles it."""
    t = _normalize(text)
    if not t:
        return None
    for pattern, handler in _COMPILED:
        m = pattern.fullmatch(t)
        if m:
            result = handler(m, text)
            if result is not None:
                return result
    return None

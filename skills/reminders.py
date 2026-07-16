# skills/reminders.py — voice reminders & timers that Nova speaks aloud when due.

import time

from skills.registry import skill


@skill(
    name="set_reminder",
    description=(
        "Set a reminder or timer that Nova will announce out loud after a delay. "
        "Use when the user says things like 'remind me in 20 minutes to X', 'set a "
        "timer for 5 minutes', or 'in 2 hours tell me to Y'. Convert the delay to "
        "minutes (30 seconds = 0.5, 2 hours = 120)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "minutes": {"type": "number", "description": "Delay in minutes from now."},
            "text": {"type": "string", "description": "What to remind the user about."},
        },
        "required": ["minutes", "text"],
    },
    examples=["remind me in 20 minutes to check the oven", "set a timer for 5 minutes"],
)
def set_reminder(args: dict) -> str:
    from utils.proactive import add_reminder
    try:
        minutes = float(args.get("minutes", 0))
    except (TypeError, ValueError):
        return "I need a number of minutes for that reminder."
    text = (args.get("text") or "").strip() or "your reminder"
    if minutes <= 0:
        return "The reminder needs to be for some time in the future."
    add_reminder(time.time() + minutes * 60, text)
    when = f"{int(round(minutes))} minutes" if minutes >= 1 else f"{int(round(minutes * 60))} seconds"
    return f"Got it — I'll remind you in {when}: {text}."


@skill(
    name="set_daily_reminder",
    description=(
        "Set a RECURRING reminder Nova announces at the same time every day. Use for "
        "'every day at 6 pm remind me to X' or 'remind me daily at 9 to Y'. Time must "
        "be 24-hour HH:MM (6 pm = 18:00)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "time": {"type": "string", "description": "24-hour time HH:MM, e.g. 18:00."},
            "text": {"type": "string", "description": "What to announce."},
        },
        "required": ["time", "text"],
    },
    examples=["every day at 6pm remind me to water the plants"],
)
def set_daily_reminder(args: dict) -> str:
    from utils.proactive import add_daily_reminder
    hhmm = (args.get("time") or "").strip()
    if not (len(hhmm) == 5 and hhmm[2] == ":" and hhmm[:2].isdigit() and hhmm[3:].isdigit()):
        return "I need the time as HH:MM in 24-hour format, like 18:00."
    text = (args.get("text") or "").strip() or "your daily reminder"
    add_daily_reminder(hhmm, text)
    return f"Done — every day at {hhmm} I'll say: {text}."


@skill(
    name="cancel_reminders",
    description=(
        "Cancel reminders. Pass 'matching' with a keyword to cancel specific ones "
        "(e.g. 'oven'), or omit it to cancel ALL reminders and timers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "matching": {"type": "string", "description": "Keyword to match; empty = cancel all."},
        },
    },
    examples=["cancel my oven reminder", "cancel all reminders"],
)
def cancel_reminders(args: dict) -> str:
    from utils.proactive import remove_reminders
    n = remove_reminders((args.get("matching") or "").strip())
    if n == 0:
        return "Nothing matched — no reminders were cancelled."
    return f"Cancelled {n} reminder{'s' if n != 1 else ''}."


@skill(
    name="list_reminders",
    description="List the reminders/timers the user currently has pending, including daily recurring ones.",
    parameters={"type": "object", "properties": {}},
    examples=["what reminders do I have", "list my timers"],
)
def list_reminders(_args: dict) -> str:
    from utils.proactive import pending_reminders
    items = pending_reminders()
    if not items:
        return "You have no reminders set."
    parts = []
    for r in sorted(items, key=lambda r: (0, r.get("due", 0)) if "due" in r else (1, 0)):
        if "daily" in r:
            parts.append(f"daily at {r['daily']}: {r.get('text', '')}")
        else:
            mins = max(0, round((r.get("due", 0) - time.time()) / 60))
            parts.append(f"in about {mins} minute{'s' if mins != 1 else ''}: {r.get('text', '')}")
    return "You've got " + "; ".join(parts) + "."

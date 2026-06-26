# skills/system.py — system power and clock.

import os
import sys
from datetime import datetime

from skills.registry import skill


@skill(
    name="power_control",
    description="Control system power on Windows. Use only when the user clearly asks.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["shutdown", "restart", "lock", "cancel"]}
        },
        "required": ["action"],
    },
    examples=["lock my pc", "shut down the computer", "cancel the shutdown"],
)
def power_control(args: dict) -> str:
    action = (args.get("action") or "").strip().lower()
    if not sys.platform.startswith("win"):
        return "Power control is only set up for Windows right now."
    if action == "shutdown":
        os.system("shutdown /s /t 15")
        return "Shutting down in 15 seconds. Say cancel to stop."
    if action == "restart":
        os.system("shutdown /r /t 15")
        return "Restarting in 15 seconds. Say cancel to stop."
    if action == "lock":
        os.system("rundll32.exe user32.dll,LockWorkStation")
        return "Locking the PC."
    if action in ("cancel", "abort"):
        os.system("shutdown /a")
        return "Cancelled the pending shutdown."
    return f"Unknown power action: {action}"


@skill(
    name="get_current_time",
    description="Get the current local date and time.",
    parameters={"type": "object", "properties": {}},
    examples=["what time is it", "what's the date"],
)
def get_current_time(_args: dict) -> str:
    now = datetime.now()
    return "It is " + now.strftime("%I:%M %p on %A, %B %d").lstrip("0")

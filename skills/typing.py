# skills/typing.py — dictation: type spoken text into whatever app has focus.
#
# Uses Windows SendInput with KEYEVENTF_UNICODE, so it works in any app that
# accepts keyboard input (editors, browsers, chat boxes) with no dependencies.
# Nova listens via the mic, so the user's target app keeps keyboard focus.

import sys

from skills.registry import skill


def _send_text_windows(text: str) -> int:
    import ctypes
    from ctypes import wintypes

    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002
    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_ubyte * 32)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    events = []
    for ch in text.replace("\r\n", "\n"):
        code = 0x0D if ch == "\n" else ord(ch)  # Enter for newlines
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = INPUT(type=INPUT_KEYBOARD)
            inp.ki = KEYBDINPUT(0, code, flags, 0, 0)
            events.append(inp)

    arr = (INPUT * len(events))(*events)
    sent = ctypes.windll.user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
    return sent // 2  # key-down/up pairs -> characters


@skill(
    name="type_text",
    description=(
        "TYPE text into whatever window/app the user currently has focused, as if they "
        "typed it on the keyboard (dictation). Use when the user says 'type X', 'write X "
        "here', or 'dictate X'. Type EXACTLY the text they want — expand it naturally if "
        "they ask you to compose (e.g. 'type a polite decline'), otherwise verbatim."
    ),
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to type."}},
        "required": ["text"],
    },
    examples=["type hello, running 10 minutes late", "dictate: see you tomorrow"],
)
def type_text(args: dict) -> str:
    text = args.get("text") or ""
    if not text:
        return "There was nothing to type."
    if not sys.platform.startswith("win"):
        return "Typing is only set up for Windows right now."
    try:
        n = _send_text_windows(text)
    except Exception as e:
        return f"Couldn't type that: {e}"
    if n == 0:
        return "Windows blocked the keystrokes (the focused app may be elevated/admin)."
    return f"Typed it ({n} characters)."

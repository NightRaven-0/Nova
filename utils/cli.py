# utils/cli.py
# Startup banner + robust exit-command matching.

import os
import sys

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[38;5;44m",
    "mag": "\033[38;5;177m",
    "grey": "\033[38;5;245m",
    "green": "\033[38;5;78m",
}


def _enable_ansi() -> None:
    """Turn on ANSI/VT colours and make stdout UTF-8 so the box-drawing banner
    (and any unicode Nova prints) never crashes a cp1252 Windows console."""
    if os.name == "nt":
        os.system("")  # side effect: enables virtual-terminal processing on Win10+
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _c(name: str) -> str:
    if os.getenv("NO_COLOR"):
        return ""
    return _COLORS.get(name, "")


# ---------------------------------------------------------------------------
# Exit detection
# ---------------------------------------------------------------------------
# Whole-utterance match against a broad set. We match the *entire* cleaned
# phrase (not words inside a sentence) so "stop the music" never quits by
# accident. Note: "quit" is a short plosive word Whisper often mangles into
# "with"/"it" — "exit", "stop", "goodbye" and "bye" transcribe far more
# reliably, so those are what we advertise in the banner.
_EXIT_PHRASES = {
    "quit", "exit", "stop", "goodbye", "good bye", "bye", "bye bye",
    "goodnight", "good night", "shutdown", "shut down", "quiet", "quits",
    "see you", "see ya", "that's all", "thats all", "that's it", "thats it",
    "nova stop", "stop nova", "nova exit", "turn off", "power off",
    "sign off", "log off", "close", "end", "exit nova",
}


def is_exit_command(text: str) -> bool:
    """True if the user's whole utterance is a request to leave."""
    t = (text or "").lower().strip().strip(".!?,\"' ")
    return t in _EXIT_PHRASES


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def _on(flag: bool) -> str:
    return f"{_c('green')}on{_c('reset')}" if flag else f"{_c('grey')}off{_c('reset')}"


def print_banner() -> None:
    """Print a clean, informative Nova startup banner."""
    _enable_ansi()
    from config import (
        LOCAL_LLM_MODEL, PIPER_MODEL_PATH, TTS_BACKEND,
        USE_WAKE_WORD, USE_BARGE_IN,
    )

    if TTS_BACKEND == "piper":
        voice = os.path.splitext(os.path.basename(PIPER_MODEL_PATH))[0]
        for p in ("en_US-", "en_GB-", "en_"):
            voice = voice.replace(p, "")
    else:
        voice = TTS_BACKEND

    rule = "─" * 47
    print(
        f"""
{_c('cyan')}{_c('bold')}    ◇  N O V A{_c('reset')}  {_c('dim')}·  offline voice assistant{_c('reset')}
{_c('dim')}    {rule}{_c('reset')}
    {_c('grey')}brain{_c('reset')} {LOCAL_LLM_MODEL}    {_c('grey')}voice{_c('reset')} {voice}    {_c('grey')}wake{_c('reset')} {_on(USE_WAKE_WORD)}    {_c('grey')}barge{_c('reset')} {_on(USE_BARGE_IN)}
{_c('dim')}    {rule}{_c('reset')}
    Speak naturally — I'm listening.
    Say {_c('bold')}"exit"{_c('reset')} or {_c('bold')}"stop"{_c('reset')} to leave.
"""
    )

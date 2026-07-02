# utils/afk.py
# Sets/clears an "AFK" indicator when the user goes idle.
#
# IMPORTANT — why Rich Presence and not the actual Discord "Status" text:
# Discord has NO official/safe way for a user to set their own custom status text
# programmatically. Doing so requires driving your *account token* (self-botting),
# which violates Discord's ToS and risks a ban. We deliberately do NOT do that.
# Instead we use Discord **Rich Presence** over the local client IPC (pypresence) —
# the legitimate route — which shows Nova as an activity like "AFK — away from
# keyboard" on your profile.
#
# Opt-in setup:
#   1) pip install pypresence
#   2) create a free app at https://discord.com/developers/applications
#   3) put its Application ID in .env as DISCORD_CLIENT_ID=...
# If any of that is missing, these functions no-op silently.

import os
import sys
import time

_rpc = None
_rpc_tried = False
_afk_active = False

# Windows we don't consider "away" for — i.e. the user is watching/consuming, not gone.
_WATCHING_HINTS = ("youtube", "netflix", "twitch", "prime video", "hotstar",
                   "disney", " - vlc", "mpv", "spotify")


def _foreground_title() -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        return (buf.value or "")
    except Exception:
        return ""


def _watching_something() -> bool:
    title = _foreground_title().lower()
    return any(h in title for h in _WATCHING_HINTS)


def _get_rpc():
    global _rpc, _rpc_tried
    if _rpc is not None:
        return _rpc
    if _rpc_tried:
        return None
    _rpc_tried = True
    client_id = os.getenv("DISCORD_CLIENT_ID")
    if not client_id:
        return None  # feature not configured — stay silent
    try:
        from pypresence import Presence
        r = Presence(client_id)
        r.connect()
        _rpc = r
        return _rpc
    except Exception as e:
        print(f"[afk] Discord Rich Presence unavailable ({e}); AFK indicator off.")
        return None


def on_idle() -> None:
    """User went idle — show an AFK presence, unless they're watching something."""
    global _afk_active
    if _afk_active or _watching_something():
        return
    rpc = _get_rpc()
    if rpc is None:
        return
    try:
        rpc.update(state="AFK — away from keyboard", details="Idle",
                   start=int(time.time()))
        _afk_active = True
        print("[afk] set AFK presence")
    except Exception:
        pass


def on_active() -> None:
    """User is back — clear the AFK presence."""
    global _afk_active
    if not _afk_active:
        return
    rpc = _get_rpc()
    if rpc is not None:
        try:
            rpc.clear()
        except Exception:
            pass
    _afk_active = False

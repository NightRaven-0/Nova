# skills/apps.py — launch desktop applications, safely.
#
# Security: the old version did subprocess.Popen(user_text, shell=True), which let
# a garbled/adversarial app name inject shell commands (e.g. "foo & del ..."). This
# version NEVER runs arbitrary text through a shell. It ShellExecutes an allowlisted
# URI, launches a PATH-resolved exe with shell=False, or ShellExecutes a REAL
# shortcut file we found in the Start Menu. The user's text is only ever used to
# *match* — never executed — so there is no injection surface.

import os
import shutil
import subprocess
import sys

from skills.registry import skill

# name -> (kind, target):
#   "uri" -> os.startfile (ShellExecute; the string is a protocol/URI, no cmdline)
#   "exe" -> resolved on PATH with shutil.which, launched with shell=False
_APPS = {
    "spotify": ("uri", "spotify:"),
    "steam": ("uri", "steam://open/main"),
    "settings": ("uri", "ms-settings:"),
    "camera": ("uri", "microsoft.windows.camera:"),
    "store": ("uri", "ms-windows-store:"),
    "explorer": ("exe", "explorer.exe"),
    "file explorer": ("exe", "explorer.exe"),
    "notepad": ("exe", "notepad.exe"),
    "calculator": ("exe", "calc.exe"),
    "calc": ("exe", "calc.exe"),
    "paint": ("exe", "mspaint.exe"),
    "task manager": ("exe", "taskmgr.exe"),
    "terminal": ("exe", "wt.exe"),
}


def _start_menu_dirs():
    dirs = []
    for var in ("APPDATA", "ProgramData"):
        base = os.environ.get(var)
        if base:
            dirs.append(os.path.join(base, "Microsoft", "Windows", "Start Menu", "Programs"))
    return dirs


def _find_shortcut(app: str):
    """Best-matching Start Menu shortcut (.lnk/.url) for `app`, or None.
    This is how we reach installed apps that aren't on PATH — Steam, Discord,
    YouTube Music, Epic, games, etc. all put a shortcut here."""
    best = None
    for root in _start_menu_dirs():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.lower().endswith((".lnk", ".url")):
                    continue
                stem = os.path.splitext(f)[0].lower()
                if stem == app:
                    return os.path.join(dirpath, f)  # exact match wins immediately
                if app in stem or stem in app:
                    cand = os.path.join(dirpath, f)
                    if best is None or len(stem) < best[0]:
                        best = (len(stem), cand)  # prefer the closest-length name
    return best[1] if best else None


@skill(
    name="open_application",
    description=(
        "Open an INSTALLED desktop application on Windows by name — e.g. steam, "
        "discord, spotify, youtube music, epic games, obs, notepad, calculator, "
        "task manager, settings. Use this (not the website tool) whenever the user "
        "names an app they have installed; it resolves apps via Start Menu shortcuts."
    ),
    parameters={
        "type": "object",
        "properties": {"app": {"type": "string", "description": "Name of the application."}},
        "required": ["app"],
    },
    examples=["open steam", "launch discord", "open youtube music", "open task manager"],
)
def open_application(args: dict) -> str:
    app = (args.get("app") or "").strip().lower()
    if not app:
        return "No application given."
    if not sys.platform.startswith("win"):
        return "Application launching is only set up for Windows right now."

    entry = _APPS.get(app)

    # 1) Allowlisted URI app — safe ShellExecute, never a shell command line.
    if entry and entry[0] == "uri":
        try:
            os.startfile(entry[1])
            return f"Opening {app}."
        except Exception as e:
            return f"Couldn't open {app}: {e}"

    # 2) Known executable resolvable on PATH — launch with NO shell.
    exe = entry[1] if entry else app
    path = shutil.which(exe)
    if path:
        try:
            subprocess.Popen([path])
            return f"Opening {app}."
        except Exception as e:
            return f"Couldn't open {app}: {e}"

    # 3) General case: find the app's Start Menu shortcut and ShellExecute it.
    lnk = _find_shortcut(app)
    if lnk:
        try:
            os.startfile(lnk)  # launches the real .lnk/.url we located — no injection
            return f"Opening {app}."
        except Exception as e:
            return f"Couldn't open {app}: {e}"

    return f"I couldn't find an app called '{app}'. Is it installed?"

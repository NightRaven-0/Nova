# skills/apps.py — launch desktop applications, safely.
#
# Security: the old version did subprocess.Popen(user_text, shell=True), which let
# a garbled/adversarial app name inject shell commands (e.g. "foo & del ..."). This
# version NEVER runs arbitrary text through a shell. It either ShellExecutes an
# allowlisted URI, or resolves an executable on PATH and launches it with shell=False.

import os
import shutil
import subprocess
import sys

from skills.registry import skill

# name -> (kind, target):
#   "uri" -> launched via os.startfile (ShellExecute; treats the string as a file/
#            URI, so there is no command-line to inject into)
#   "exe" -> resolved on PATH with shutil.which, launched with shell=False
_APPS = {
    "spotify": ("uri", "spotify:"),
    "settings": ("uri", "ms-settings:"),
    "camera": ("uri", "microsoft.windows.camera:"),
    "explorer": ("exe", "explorer.exe"),
    "file explorer": ("exe", "explorer.exe"),
    "notepad": ("exe", "notepad.exe"),
    "calculator": ("exe", "calc.exe"),
    "calc": ("exe", "calc.exe"),
    "paint": ("exe", "mspaint.exe"),
    "task manager": ("exe", "taskmgr.exe"),
    "terminal": ("exe", "wt.exe"),
}


@skill(
    name="open_application",
    description="Open a desktop application on Windows (e.g. spotify, notepad, calculator, file explorer, settings, paint, task manager).",
    parameters={
        "type": "object",
        "properties": {"app": {"type": "string", "description": "Name of the application."}},
        "required": ["app"],
    },
    examples=["open spotify", "launch notepad", "open task manager"],
)
def open_application(args: dict) -> str:
    app = (args.get("app") or "").strip().lower()
    if not app:
        return "No application given."
    if not sys.platform.startswith("win"):
        return "Application launching is only set up for Windows right now."

    entry = _APPS.get(app)

    # Allowlisted URI app: safe ShellExecute, never a shell command line.
    if entry and entry[0] == "uri":
        try:
            os.startfile(entry[1])
            return f"Opening {app}."
        except Exception as e:
            return f"Couldn't open {app}: {e}"

    # Resolve an executable on PATH and launch with NO shell (no injection surface).
    exe = entry[1] if entry else app
    path = shutil.which(exe)
    if path:
        try:
            subprocess.Popen([path])  # list form => shell=False
            return f"Opening {app}."
        except Exception as e:
            return f"Couldn't open {app}: {e}"

    return f"I couldn't find an app called {app}."

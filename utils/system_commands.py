# utils/system_commands.py

import webbrowser
import os
import subprocess
import sys

def execute_command(text: str):
    """
    Identify and execute simple system commands based on recognized text.
    Returns a short status string if a command was executed,
    or None if no command matched.
    """
    if not text:
        return None

    lowered = text.lower().strip()

    # ---- Browser commands ----
    if "open youtube" in lowered:
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube."

    if "open google" in lowered or "search on google" in lowered:
        webbrowser.open("https://www.google.com")
        return "Opening Google."

    if "open browser" in lowered:
        webbrowser.open("https://www.google.com")
        return "Opening your browser."

    # ---- File explorer (Windows-focused) ----
    if "open explorer" in lowered or "open file explorer" in lowered:
        if sys.platform.startswith("win"):
            subprocess.Popen("explorer")
            return "Opening File Explorer."
        return "File explorer command is only implemented for Windows."

    # ---- Simple media / apps (examples, tweak paths as you like) ----
    if "open spotify" in lowered:
        # On Windows, if Spotify is in PATH:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen("spotify")
                return "Opening Spotify."
            else:
                return "Spotify command not configured for this OS."
        except Exception:
            return "Couldn't open Spotify."

    # ---- Power commands (use with care) ----
    if "shutdown my pc" in lowered or "shut down my pc" in lowered:
        if sys.platform.startswith("win"):
            os.system("shutdown /s /t 10")
            return "Shutting down your PC in 10 seconds."
        return "Shutdown command not implemented for this OS."

    if "restart my pc" in lowered or "reboot my pc" in lowered:
        if sys.platform.startswith("win"):
            os.system("shutdown /r /t 10")
            return "Restarting your PC in 10 seconds."
        return "Restart command not implemented for this OS."

    # No known command matched
    return None

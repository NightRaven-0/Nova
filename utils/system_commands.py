import webbrowser
import os
import subprocess

def execute_command(text: str):
    """Identify and execute simple system commands based on text."""
    lowered = text.lower()

    # -------- Browser commands --------
    if "open youtube" in lowered:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube."

    if "open google" in lowered:
        webbrowser.open("https://google.com")
        return "Opening Google."

    if "open browser" in lowered:
        webbrowser.open("https://google.com")
        return "Opening browser."

    # -------- System-level commands --------
    if "shutdown" in lowered:
        os.system("shutdown /s /t 5")
        return "Shutting down your PC."

    if "restart" in lowered:
        os.system("shutdown /r /t 5")
        return "Restarting your PC."

    if "open explorer" in lowered:
        subprocess.Popen("explorer")
        return "Opening File Explorer."

    # If no command matched:
    return None
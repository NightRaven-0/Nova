# brain/tools.py
# Tool registry for LLM function-calling.
# Each tool has an OpenAI-style schema (TOOLS) and an implementation (dispatched
# by name in dispatch_tool). The LLM decides when to call these.

import os
import sys
import json
import subprocess
import webbrowser
from datetime import datetime

# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

_WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://www.github.com",
    "gmail": "https://mail.google.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "wikipedia": "https://www.wikipedia.org",
}

# Windows launch commands for common apps
_APPS = {
    "spotify": "spotify",
    "explorer": "explorer",
    "file explorer": "explorer",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "settings": "start ms-settings:",
    "camera": "start microsoft.windows.camera:",
}


def _open_website(site: str) -> str:
    site = (site or "").strip().lower()
    if not site:
        return "No site given."
    if site in _WEBSITES:
        webbrowser.open(_WEBSITES[site])
        return f"Opening {site}."
    if "." in site:  # looks like a domain/URL
        url = site if site.startswith("http") else "https://" + site
        webbrowser.open(url)
        return f"Opening {site}."
    # unknown name -> search for it instead
    webbrowser.open(f"https://www.google.com/search?q={site}")
    return f"I didn't recognise that site, so I searched for {site}."


def _web_search(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "No search query given."
    webbrowser.open("https://www.google.com/search?q=" + query.replace(" ", "+"))
    return f"Searching the web for {query}."


def _open_application(app: str) -> str:
    app = (app or "").strip().lower()
    if not app:
        return "No application given."
    if not sys.platform.startswith("win"):
        return "Application launching is only set up for Windows right now."
    cmd = _APPS.get(app, app)
    try:
        subprocess.Popen(cmd, shell=True)
        return f"Opening {app}."
    except Exception as e:
        return f"Couldn't open {app}: {e}"


def _power_control(action: str) -> str:
    action = (action or "").strip().lower()
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


def _get_current_time(_args: dict) -> str:
    now = datetime.now()
    return "It is " + now.strftime("%I:%M %p on %A, %B %d").lstrip("0")


# ---------------------------------------------------------------------------
# Schemas exposed to the LLM
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_website",
            "description": "Open a website in the default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "description": "Site name (e.g. youtube, google, github) or a domain/URL.",
                    }
                },
                "required": ["site"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and open the results in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open a desktop application on Windows (e.g. spotify, notepad, calculator, file explorer, settings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "Name of the application."}
                },
                "required": ["app"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "power_control",
            "description": "Control system power on Windows. Use only when the user clearly asks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["shutdown", "restart", "lock", "cancel"],
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


_DISPATCH = {
    "open_website": lambda a: _open_website(a.get("site")),
    "web_search": lambda a: _web_search(a.get("query")),
    "open_application": lambda a: _open_application(a.get("app")),
    "power_control": lambda a: _power_control(a.get("action")),
    "get_current_time": _get_current_time,
}


def dispatch_tool(name: str, arguments) -> str:
    """Execute a tool by name. `arguments` may be a JSON string or a dict.
    Always returns a short string to feed back to the model."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(arguments)
    except Exception as e:
        return f"Tool {name} failed: {e}"

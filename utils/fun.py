# utils/fun.py
# The "unnecessary but delightful" tools: vibe check, sass counter, Attenborough mode.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from utils.logs import log_activity

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SASS_FILE = _DATA_DIR / "sass.json"


# ---- vibe check -----------------------------------------------------------
def _gpu_stats() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            u, mu, mt, t = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
            return f"GPU {u}% ({mu}/{mt} MB, {t}°C)"
    except Exception:
        pass
    return ""


def vibe_check() -> str:
    try:
        import psutil
    except ImportError:
        return "I can't read the system vibe right now (psutil isn't installed)."
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    parts = [f"CPU {cpu:.0f}%", f"RAM {mem.percent:.0f}% ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)"]
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            parts.append(f"battery {batt.percent:.0f}%" + (", charging" if batt.power_plugged else ""))
    except Exception:
        pass
    gpu = _gpu_stats()
    if gpu:
        parts.append(gpu)
    load = max(cpu, mem.percent)
    vibe = ("immaculate — barely awake" if load < 25 else
            "chill and cruising" if load < 55 else
            "working up a sweat" if load < 80 else
            "absolutely cooking — send coolant")
    return "Stats: " + "; ".join(parts) + f". Vibe: {vibe}."


# ---- sass counter ---------------------------------------------------------
def _read_sass() -> int:
    try:
        return int(json.loads(_SASS_FILE.read_text(encoding="utf-8")).get("count", 0))
    except Exception:
        return 0


def bump_sass() -> int:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = _read_sass() + 1
    _SASS_FILE.write_text(json.dumps({"count": n}), encoding="utf-8")
    return n


def get_sass() -> int:
    return _read_sass()


# ---- Screen roast ---------------------------------------------------------
def _active_window_title() -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


def roast_observe() -> str:
    """Return context for the LLM to roast whatever is on the user's screen."""
    title = _active_window_title()
    log_activity(f"roast: {title!r}")
    if not title:
        return ("There's no foreground window at all — the desktop is empty. Roast the user "
                "in one line for having such a barren, boring screen right now.")
    return (f"The user's active window is '{title}'. In ONE or TWO short spoken sentences, "
            "roast them about it — witty, playful, a little savage, like a friend teasing "
            "them about what's on their screen. Name the specific app/window. Example vibe: "
            "\"Task Manager again? Watching me eat your RAM for breakfast, are we?\"")

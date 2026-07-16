# utils/proactive.py
# Proactive Nova — a background monitor that notices things and gives the main
# loop lines to say ON HER OWN: due reminders, long-activity nudges, and resource
# alerts. Crucial design point: the monitor thread ONLY detects and queues; the
# main (audio) thread drains the queue and does all the speaking — so the mic
# (STT) and speaker (TTS) never fight over the sound device.

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_REMINDERS_FILE = _DATA_DIR / "reminders.json"


# ---------------------------------------------------------------------------
# Reminders store (shared with the set_reminder skill)
# ---------------------------------------------------------------------------
def _load_reminders() -> list:
    try:
        return json.loads(_REMINDERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_reminders(items: list) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _REMINDERS_FILE.write_text(json.dumps(items), encoding="utf-8")


def add_reminder(due_ts: float, text: str) -> None:
    items = _load_reminders()
    items.append({"due": float(due_ts), "text": text})
    _save_reminders(items)


def add_daily_reminder(hhmm: str, text: str) -> None:
    """Recurring daily announcement at HH:MM (24h), e.g. 18:00 'drink water'."""
    items = _load_reminders()
    items.append({"daily": hhmm, "text": text, "last": ""})
    _save_reminders(items)


def remove_reminders(matching: str = "") -> int:
    """Remove all reminders, or only those whose text contains `matching`."""
    items = _load_reminders()
    if matching:
        keep = [r for r in items if matching.lower() not in r.get("text", "").lower()]
    else:
        keep = []
    _save_reminders(keep)
    return len(items) - len(keep)


def pending_reminders() -> list:
    return _load_reminders()


def _pop_due(now: float) -> list:
    """One-shot reminders whose time has come (removed), plus daily reminders
    due today that haven't fired yet (kept, stamped with today)."""
    items = _load_reminders()
    lt = time.localtime(now)
    today = time.strftime("%Y-%m-%d", lt)
    now_hhmm = time.strftime("%H:%M", lt)

    due, keep, changed = [], [], False
    for r in items:
        if "daily" in r:
            if r.get("last") != today and r["daily"] <= now_hhmm:
                due.append(r)
                r = {**r, "last": today}
                changed = True
            keep.append(r)
        elif r.get("due", 0) <= now:
            due.append(r)
            changed = True
        else:
            keep.append(r)
    if changed:
        _save_reminders(keep)
    return due


# ---------------------------------------------------------------------------
# Daily greeting helper
# ---------------------------------------------------------------------------
def first_launch_today() -> bool:
    """True once per calendar day (persists the last-seen date)."""
    stamp = _DATA_DIR / "last_launch.txt"
    today = time.strftime("%Y-%m-%d")
    try:
        if stamp.read_text(encoding="utf-8").strip() == today:
            return False
    except Exception:
        pass
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp.write_text(today, encoding="utf-8")
    return True


def daily_greeting() -> str:
    h = time.localtime().tm_hour
    part = "morning" if h < 12 else ("afternoon" if h < 17 else "evening")
    return f"Good {part}! Nova's up and listening."


# ---------------------------------------------------------------------------
# Foreground window (for the "you've been on X for a while" nudge)
# ---------------------------------------------------------------------------
def _active_window_title() -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value or ""
    except Exception:
        return ""


def _app_name(title: str) -> str:
    parts = [p.strip() for p in title.split(" - ") if p.strip()]
    return parts[-1] if parts else title


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------
class ProactiveMonitor:
    """Background thread that queues spoken nudges for the main loop to deliver."""

    def __init__(self, interval: float = 15.0, activity_hours: float = 2.0,
                 ram_alert: float = 93.0):
        self.interval = interval
        self.activity_secs = activity_hours * 3600
        self.ram_alert = ram_alert
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # per-trigger state
        self._app = None
        self._app_since = time.time()
        self._app_nudged = False
        self._ram_high_since = None
        self._ram_nudged = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def has_pending(self) -> bool:
        with self._lock:
            return len(self._queue) > 0

    def drain(self) -> list:
        with self._lock:
            msgs = list(self._queue)
            self._queue.clear()
        return msgs

    def _emit(self, msg: str) -> None:
        with self._lock:
            self._queue.append(msg)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            for check in (self._check_reminders, self._check_activity, self._check_resources):
                try:
                    check()
                except Exception:
                    pass

    def _check_reminders(self) -> None:
        for r in _pop_due(time.time()):
            self._emit(f"Reminder: {r.get('text', 'your reminder')}.")

    def _check_activity(self) -> None:
        app = _app_name(_active_window_title()) or None
        if app != self._app:
            self._app, self._app_since, self._app_nudged = app, time.time(), False
            return
        if app and not self._app_nudged and time.time() - self._app_since >= self.activity_secs:
            hrs = round((time.time() - self._app_since) / 3600, 1)
            self._app_nudged = True
            self._emit(f"Hey, you've been on {app} for about {hrs} hours — "
                       "maybe stretch or grab some water?")

    def _check_resources(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        ram = psutil.virtual_memory().percent
        now = time.time()
        if ram >= self.ram_alert:
            if self._ram_high_since is None:
                self._ram_high_since = now
            elif not self._ram_nudged and now - self._ram_high_since >= 60:
                self._ram_nudged = True
                self._emit(f"Heads up — your RAM's sitting at {ram:.0f} percent. "
                           "Might be worth closing something.")
        else:
            self._ram_high_since = None
            if ram < self.ram_alert - 8:
                self._ram_nudged = False  # re-arm once it recovers

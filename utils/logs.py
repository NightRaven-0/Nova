# utils/logs.py
# Nova's own activity/diagnostic logs, plus a STRICTLY-scoped auto-cleanup.
# The purge deletes ONLY Nova's own logs older than the retention window. It
# never recurses, never follows symlinks, only removes Nova's own log file
# types, and hard-refuses anything that isn't unmistakably Nova's logs/ dir.
# It will never touch Windows, system, or any other application's logs.

from __future__ import annotations

import json
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # .../Nova
NOVA_LOG_DIR = _PROJECT_ROOT / "logs"

# Only files Nova itself writes are ever eligible for deletion.
_OWN_LOG_SUFFIXES = {".log", ".jsonl"}


def _is_clearly_novas_log_dir(d: Path) -> bool:
    """Belt-and-suspenders guard: only ever operate on Nova's own logs/ folder."""
    d = d.resolve()
    if d != NOVA_LOG_DIR.resolve():
        return False
    if d.name.lower() != "logs":
        return False
    parts_lower = [p.lower() for p in d.parts]
    if any(p in parts_lower for p in ("windows", "system32", "system")):
        return False
    if len(d.parts) <= 2:           # refuse anything as shallow as a drive root
        return False
    return True


def log_activity(text: str) -> None:
    """Append a timestamped activity line to today's Nova activity log."""
    NOVA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "activity": text}
    path = NOVA_LOG_DIR / f"activity-{time.strftime('%Y-%m-%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def purge_old_logs(retention_days: int = 7) -> tuple[int, int]:
    """Delete Nova's OWN log files older than `retention_days`.
    Non-recursive; files only; no symlinks; suffix-whitelisted; path-contained.
    Returns (files_deleted, bytes_freed). Never raises on individual failures."""
    if not NOVA_LOG_DIR.exists() or not _is_clearly_novas_log_dir(NOVA_LOG_DIR):
        return (0, 0)

    cutoff = time.time() - retention_days * 86400
    base = NOVA_LOG_DIR.resolve()
    deleted = 0
    freed = 0
    for f in NOVA_LOG_DIR.iterdir():            # non-recursive: direct children only
        try:
            if not f.is_file() or f.is_symlink():            # never dirs, never symlinks
                continue
            if f.suffix.lower() not in _OWN_LOG_SUFFIXES:     # only our own log types
                continue
            if f.resolve().parent != base:                    # must stay inside logs/
                continue
            st = f.stat()
            if st.st_mtime < cutoff:
                size = st.st_size
                f.unlink()
                deleted += 1
                freed += size
        except Exception:
            continue
    return (deleted, freed)

# skills/ — Nova's capabilities, one file per area.
# To add a capability: drop a new module here and decorate a function with
# @skill(...). It's auto-discovered on startup; no other edits needed.

from __future__ import annotations

import importlib
import pkgutil

from skills.registry import (  # noqa: F401
    Skill,
    all_skills,
    dispatch,
    get_tool_schemas,
    skill,
)


def discover() -> None:
    """Import every skill module so its @skill decorators register."""
    import skills as _pkg

    for _, modname, _ in pkgutil.iter_modules(_pkg.__path__):
        if modname == "registry" or modname.startswith("_"):
            continue
        importlib.import_module(f"skills.{modname}")


discover()

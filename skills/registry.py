# skills/registry.py
# Tiny skill registry. A "skill" is one capability the LLM can call: a name, an
# OpenAI-style parameter schema, and a function (dict args -> short string reply).
# Skills register themselves with the @skill decorator; the brain pulls them all
# via get_tool_schemas() / dispatch().

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict
    func: Callable[[dict], str]
    examples: List[str] = field(default_factory=list)


_REGISTRY: Dict[str, Skill] = {}


def skill(name: str, description: str, parameters: dict | None = None,
          examples: List[str] | None = None):
    """Decorator: register a function as a callable skill."""
    params = parameters or {"type": "object", "properties": {}}

    def decorator(func: Callable[[dict], str]) -> Callable[[dict], str]:
        _REGISTRY[name] = Skill(name, description, params, func, examples or [])
        return func

    return decorator


def get_tool_schemas() -> List[dict]:
    """OpenAI-format tool schemas for every registered skill."""
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in _REGISTRY.values()
    ]


def all_skills() -> List[Skill]:
    return list(_REGISTRY.values())


def dispatch(name: str, arguments: Any) -> str:
    """Run a skill by name. `arguments` may be a JSON string or a dict.
    Always returns a short string to feed back to the model."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    s = _REGISTRY.get(name)
    if s is None:
        return f"Unknown skill: {name}"
    try:
        return s.func(arguments)
    except Exception as e:
        return f"Skill {name} failed: {e}"

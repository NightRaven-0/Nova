# skills/fun.py — the unnecessary-but-delightful skills (impls live in utils/fun.py).

from skills.registry import skill


@skill(
    name="vibe_check",
    description="Read the computer's live stats (CPU, RAM, GPU, battery) and report 'the vibe'. Use when the user asks for a vibe check or how the PC/system is doing.",
    parameters={"type": "object", "properties": {}},
    examples=["vibe check", "how's the system doing"],
)
def vibe_check(_args: dict) -> str:
    from utils.fun import vibe_check as _vibe
    return _vibe()


@skill(
    name="sass_counter",
    description="Track how sassy/rude the user is being. Call with action 'bump' when the user is rude, dismissive, or tells Nova to shush; call with 'get' to report the running tally (great for teasing them later).",
    parameters={
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["bump", "get"]}},
        "required": ["action"],
    },
    examples=["how sassy have I been", "(bump when the user is rude)"],
)
def sass_counter(args: dict) -> str:
    from utils.fun import bump_sass, get_sass
    action = (args.get("action") or "get").strip().lower()
    if action in ("bump", "increment", "add", "plus"):
        return f"Noted. Sass counter is now {bump_sass()}."
    return f"You've been sassy {get_sass()} time(s) so far."


@skill(
    name="attenborough",
    description="Observe what the user is currently doing on screen (the active window) so Nova can narrate it like a David Attenborough nature documentary. Use for 'Attenborough mode' or 'narrate what I'm doing'.",
    parameters={"type": "object", "properties": {}},
    examples=["attenborough mode", "narrate what I'm doing"],
)
def attenborough(_args: dict) -> str:
    from utils.fun import attenborough_observe
    return attenborough_observe()

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
    name="roast_screen",
    description="Look at what's currently on the user's screen (the active window) and roast them about it — a witty, playful jab about whatever they're doing. Use for 'roast me', 'roast my screen', or 'make fun of what I'm doing'.",
    parameters={"type": "object", "properties": {}},
    examples=["roast me", "roast my screen", "make fun of what I'm doing"],
)
def roast_screen(_args: dict) -> str:
    from utils.fun import roast_observe
    return roast_observe()

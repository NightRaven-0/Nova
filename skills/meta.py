# skills/meta.py — Nova introspecting its own skills.

from skills.registry import skill, all_skills


@skill(
    name="list_capabilities",
    description="List what Nova can do (its available skills). Use when the user asks what you can do or for help.",
    parameters={"type": "object", "properties": {}},
    examples=["what can you do", "help", "list your skills"],
)
def list_capabilities(_args: dict) -> str:
    names = [s.name.replace("_", " ") for s in all_skills() if s.name != "list_capabilities"]
    return "I can: " + ", ".join(names) + "."

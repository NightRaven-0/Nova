# skills/models.py — switch which local LLM powers Nova.

from skills.registry import skill


@skill(
    name="switch_model",
    description="Switch which local LLM powers Nova's replies (e.g. 'gemma', 'gemma 4', or 'qwen'). Use when the user asks to change the model, brain, or AI.",
    parameters={
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Model to switch to: a friendly name like gemma or qwen, or an exact Ollama tag like gemma4:12b.",
            }
        },
        "required": ["model"],
    },
    examples=["switch to gemma", "use qwen", "change your brain to gemma 4"],
)
def switch_model(args: dict) -> str:
    model = (args.get("model") or "").strip()
    if not model:
        return "Which model? For example gemma or qwen."
    from brain.gpt_llm import set_active_model  # late import avoids a circular import
    tag = set_active_model(model)
    return f"Okay, switching my brain to {tag}. It takes effect on my next reply."

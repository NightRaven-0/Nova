# skills/models.py — switch which local LLM powers Nova.

from skills.registry import skill


@skill(
    name="switch_model",
    description=(
        "Switch which local LLM powers Nova's replies: 'fast' / 'small' (qwen3:4b, quicker) "
        "or 'big' / 'qwen' (qwen3:8b, smarter). Use when the user asks to change the "
        "model, brain, or AI."
    ),
    parameters={
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "A friendly name (fast, small, big, qwen) or an exact Ollama tag like qwen3:4b.",
            }
        },
        "required": ["model"],
    },
    examples=["switch to the fast model", "use the big model", "go back to automatic"],
)
def switch_model(args: dict) -> str:
    model = (args.get("model") or "").strip()
    if not model:
        return "Which model? For example fast, medium, big, or automatic."
    from brain import model_ladder  # late imports avoid a circular import
    from brain.gpt_llm import set_active_model

    if model.lower() in ("auto", "automatic", "ladder", "whatever fits", "default"):
        model_ladder.unpin()
        tag = model_ladder.apply(force=True)
        if not tag:
            return "Automatic model picking is off, so I'm staying on the configured brain."
        return f"Back on automatic — running {tag}, and I'll step down if the GPU gets busy."

    tag = set_active_model(model)
    installed = model_ladder.installed_tags()
    if installed and tag not in installed:
        return f"I don't have {tag} pulled yet, so I'm staying on what I've got."
    model_ladder.note_manual_switch(tag)
    return f"Okay, switching my brain to {tag}. It takes effect on my next reply."

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
    examples=["switch to the fast model", "use the big model", "use qwen"],
)
def switch_model(args: dict) -> str:
    model = (args.get("model") or "").strip()
    if not model:
        return "Which model? For example fast or big."
    from brain.gpt_llm import set_active_model  # late import avoids a circular import
    tag = set_active_model(model)
    return f"Okay, switching my brain to {tag}. It takes effect on my next reply."

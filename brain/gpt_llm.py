# brain/gpt_llm.py
# Nova's brain: talks to a local LLM (Ollama, OpenAI-compatible) or OpenAI,
# runs the function-calling loop, and keeps conversation memory.

import re

from openai import OpenAI

from config import (
    ASSISTANT_NAME,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    LOCAL_LLM_MODEL,
    OPENAI_LLM_MODEL,
    OPENAI_API_KEY,
    MEMORY_PATH,
    MEMORY_MAX_TURNS,
    MEMORY_SUMMARIZE_AFTER,
)
from brain.tools import TOOLS, dispatch_tool
from brain.memory import ConversationMemory

MAX_TOOL_ROUNDS = 5

# qwen3 and other reasoning models emit <think>...</think> blocks. Strip them so
# Nova never speaks its internal monologue aloud.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()

SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME} (Natural-language Oriented Voice Assistant), a friendly, "
    "concise voice assistant for the user's Windows PC.\n"
    "Your replies are spoken aloud, so keep them short and conversational. Do NOT use "
    "markdown, bullet points, code blocks, headings, or emoji — write the way you'd say it.\n"
    "You can control the PC with the provided tools: open websites and applications, "
    "search the web, tell the time, and power actions. Call a tool when the user wants "
    "something done on the computer; otherwise just answer.\n"
    "Speech recognition can be imperfect. If a request seems garbled or ambiguous, ask a "
    "brief clarifying question instead of guessing."
)


class Brain:
    def __init__(self):
        if LLM_BACKEND == "ollama":
            # Ollama exposes an OpenAI-compatible API; api_key is ignored but required.
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            self.model = LOCAL_LLM_MODEL
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_LLM_MODEL

        self.memory = ConversationMemory(
            path=MEMORY_PATH,
            system_prompt=SYSTEM_PROMPT,
            max_turns=MEMORY_MAX_TURNS,
            summarize_after=MEMORY_SUMMARIZE_AFTER,
            summarize_fn=self._summarize,
        )

    def _summarize(self, messages):
        """Condense old turns into a factual paragraph (no tools)."""
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation into a short factual paragraph, "
                    "capturing the user's preferences, facts about them, and any context "
                    "worth remembering later. Be concise.",
                },
                {"role": "user", "content": convo},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def ask(self, user_text: str) -> str:
        self.memory.add_user(user_text)
        working = self.memory.build_messages()

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=working,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                msg = resp.choices[0].message

                if not msg.tool_calls:
                    reply = _strip_think(msg.content)
                    self.memory.add_assistant(reply)
                    self.memory.maybe_summarize()
                    self.memory.save()
                    return reply

                # Record the assistant's tool-call request, then run each tool.
                working.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    result = dispatch_tool(tc.function.name, tc.function.arguments)
                    working.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

            # Ran out of tool rounds.
            reply = "Sorry, I got a bit tangled up there. Could you say that again?"
            self.memory.add_assistant(reply)
            self.memory.save()
            return reply

        except Exception as e:
            if LLM_BACKEND == "ollama":
                return (
                    f"(Brain error: {e}) — is Ollama running? Start it with 'ollama serve' "
                    f"and make sure you've pulled '{self.model}'."
                )
            return f"(Brain error: {e})"


# Lazily-built singleton so importing this module doesn't connect to the LLM.
_brain = None


def _get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


def ask_gpt(user_text: str) -> str:
    """Backward-compatible entry point used by nova.py."""
    return _get_brain().ask(user_text)

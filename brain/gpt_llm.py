# Reply generation (OpenAI or local fallback)
from openai import OpenAI
from config import OPENAI_API_KEY, ASSISTANT_NAME, USE_CLOUD_LLM, LOCAL_ONLY_MODE

client = OpenAI(api_key=OPENAI_API_KEY)

_conversation_history = [
    {"role": "system", "content": f"You are {ASSISTANT_NAME}, a helpful personal AI assistant."}
]

def ask_gpt(user_text: str) -> str:
    """Send user input to GPT and return reply, maintaining conversation history."""
    if LOCAL_ONLY_MODE:
        return f"[LOCAL_STUB_REPLY] I heard: {user_text}"

    try:
        if USE_CLOUD_LLM:
            _conversation_history.append({"role": "user", "content": user_text})
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=_conversation_history,
            )
            reply = response.choices[0].message.content.strip()
            _conversation_history.append({"role": "assistant", "content": reply})
            return reply
        else:
            return f"(Local LLM not yet integrated) You said: {user_text}"

    except Exception as e:
        return f"(Error: {e})"

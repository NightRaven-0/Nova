# Reply generation (OpenAI or local fallback)
from openai import OpenAI
from config import OPENAI_API_KEY, ASSISTANT_NAME, USE_CLOUD_LLM

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_gpt(user_text: str) -> str:
    """Send user input to GPT (cloud or local) and return reply"""
    try:
        if USE_CLOUD_LLM:
            # Cloud: OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are {ASSISTANT_NAME}, a helpful personal AI assistant."},
                    {"role": "user", "content": user_text}
                ]
            )
            return response.choices[0].message.content.strip()
        else:
            # Local fallback (placeholder for now)
            return f"(Local LLM not yet integrated) You said: {user_text}"

    except Exception as e:
        return f"(Error: {e})"

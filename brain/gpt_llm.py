# Reply generation (OpenAI or fallback rules)

from openai import OpenAI
from config import OPENAI_API_KEY, ASSISTANT_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_gpt(user_text: str) -> str:
    """Send user input to GPT and return reply"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # you can use v1-compatible model
            messages=[
                {"role": "system", "content": f"You are {ASSISTANT_NAME}, a helpful personal AI assistant."},
                {"role": "user", "content": user_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"

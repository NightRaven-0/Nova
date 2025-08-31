from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
models = client.models.list()
for m in models:
    print(m.id)

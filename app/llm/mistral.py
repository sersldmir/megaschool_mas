import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MistralLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1"
        )
        self.model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=400,
        )
        return response.choices[0].message.content
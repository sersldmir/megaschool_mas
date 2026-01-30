import os
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage, SystemMessage

load_dotenv()


class MistralLLM:
    def __init__(self):
        self.client = Mistral(
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        self.model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                SystemMessage(role="system", content=system_prompt),
                UserMessage(role="user", content=user_prompt),
            ],
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
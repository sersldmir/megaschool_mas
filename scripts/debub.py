import os
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage, SystemMessage

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY"),
)
model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

response = client.chat.complete(
    model=model,
    messages=[
        UserMessage(role="user", content="Define 'consciousnes'"),
    ],
    temperature=0.2,
    max_tokens=500,
)


print(response.choices[0].message.content)
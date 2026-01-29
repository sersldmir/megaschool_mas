import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.mistral.ai/v1"
)

MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

def main():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Ты вежливый и краткий технический ассистент."
            },
            {
                "role": "user",
                "content": "Объясни индексы в PostgreSQL."
            }
        ],
        # temperature=0.2,
        # max_tokens=300
    )

    print("MODEL:", MODEL)
    print("RESPONSE:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
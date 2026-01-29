import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = os.getenv("OPENROUTER_MODEL")

def main():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ты вежливый и краткий ассистент."},
            {"role": "user", "content": "Объясни разницу между JOIN и UNION в SQL."}
        ],
        temperature=0.2
    )

    print("MODEL:", MODEL)
    print("RESPONSE:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
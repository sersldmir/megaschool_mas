import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

if not API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")

# Инициализация клиента
client = Mistral(api_key=API_KEY)

# Константы
MAX_TURNS = 15  # Защита от бесконечного цикла
LOG_FILE = "interview_log.json"
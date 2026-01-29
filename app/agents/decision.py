from app.llm.mistral import MistralLLM

class DecisionAgent:
    SYSTEM_PROMPT = """
Ты — Hiring Manager.

Ты принимаешь финальное решение по кандидату на основе интервью.

Правила:
- ты НЕ задаёшь вопросы
- ты НЕ проверяешь факты
- ты анализируешь агрегированные данные интервью

Отвечай СТРОГО в JSON.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def make_decision(self, state: dict) -> dict:
        prompt = f"""
Позиция: {state["position"]}
Целевой грейд: {state["target_grade"]}
Опыт (лет): {state["experience_years"]}

Подтверждённые навыки:
{state["confirmed_skills"]}

Пробелы в знаниях (с правильными ответами):
{state["knowledge_gaps"]}

История ответов:
{state["dialog_history"]}

Оценки уверенности:
{state["confidence_scores"]}

Сформируй JSON следующего формата:

{{
  "hiring_recommendation": "Strong Hire | Hire | No Hire",
  "final_grade": "Junior | Middle | Senior",
  "confidence_score": 0-100,
  "hard_skills": {{
    "confirmed": [],
    "gaps": []
  }},
  "soft_skills": {{
    "clarity": "low | medium | high",
    "honesty": "low | medium | high",
    "engagement": "low | medium | high"
  }},
  "knowledge_gaps": [
    {{
      "topic": "",
      "expected_answer": ""
    }}
  ],
  "roadmap": [
    "конкретный шаг 1",
    "конкретный шаг 2"
  ]
}}
"""
        raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.2)
        raw = raw.replace("```json", "").replace("```", "").replace('\n', '').strip()

        return raw

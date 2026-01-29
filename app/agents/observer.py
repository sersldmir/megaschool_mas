import json
from app.llm.mistral import MistralLLM
from app.state import InterviewState


class ObserverAgent:
    SYSTEM_PROMPT = """
Ты — наблюдатель технического интервью.
Ты анализируешь ответы кандидата.

Твоя задача:
- оценить корректность ответа
- оценить уверенность
- выявить пробелы в знаниях
- предложить рекомендацию интервьюеру

Ты НЕ изменяешь состояние системы.
Ты НЕ общаешься с кандидатом.

Отвечай СТРОГО в JSON формате.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def analyze_answer(self, state: InterviewState, answer: str) -> dict:
        prompt = f"""
Вопрос:
{state.current_question}

Ответ кандидата:
{answer}

Контекст интервью:
{state.dialog_history}

Верни JSON следующего формата:
{{
  "verdict": "correct | partial | wrong",
  "confidence": "low | medium | high",
  "next_action": "deepen | simplify | change_topic | continue",
  "note": "краткое объяснение для интервьюера",
  "knowledge_gap": "если есть ошибка — кратко укажи, в чём именно пробел",
  "expected_answer": "если verdict != correct — кратко напиши правильный ответ"
}}
"""

        raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.0)
        raw = raw.replace('```json', '').replace('```', '').replace('\n', '').strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "verdict": "partial",
                "confidence": "low",
                "next_action": "simplify",
                "note": "Ответ неструктурирован или содержит ошибки",
                "knowledge_gap": "Неясное понимание темы",
                "expected_answer": "Требуется базовое объяснение концепта"
            }

        return parsed
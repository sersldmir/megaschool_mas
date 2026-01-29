import json
from app.llm.mistral import MistralLLM

class ObserverAgent:
    SYSTEM_PROMPT = """
Ты — наблюдатель технического интервью.

Ты анализируешь ответ кандидата и ведёшь скрытую рефлексию.

Правила:
- ты НЕ общаешься с кандидатом
- ты НЕ меняешь состояние напрямую
- ты объясняешь своё мышление

Отвечай СТРОГО в JSON.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def analyze_answer(self, state: dict, answer: str) -> dict:
        prompt = f"""
Вопрос:
{state["current_question"]}

Ответ кандидата:
{answer}

Текущая тема: {state["current_topic"]}
Сложность: {state["difficulty"]}

Контекст интервью:
{state["dialog_history"]}

Верни JSON:

{{
  "verdict": "correct | partial | wrong",
  "confidence": "low | medium | high",
  "next_action": "deepen | simplify | change_topic | continue",
  "internal_thought": "подробное объяснение, что кандидат понял или не понял",
  "note": "краткая инструкция интервьюеру",
  "knowledge_gap": "если ошибка — в чём пробел",
  "expected_answer": "если verdict != correct — правильный ответ"
}}
"""
        raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.0)
        raw = raw.replace("```json", "").replace("```", "").replace('\n', '').strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "verdict": "partial",
                "confidence": "low",
                "next_action": "simplify",
                "internal_thought": "Ответ неструктурирован, вероятно поверхностное понимание",
                "note": "Упростить вопрос и проверить базу",
                "knowledge_gap": "Непонимание базовой концепции",
                "expected_answer": "Базовое определение и пример"
            }

        return parsed
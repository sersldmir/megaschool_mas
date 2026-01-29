import json
from app.state import InterviewState
from app.llm.mistral import MistralLLM


class ObserverAgent:
    SYSTEM_PROMPT = """
Ты — наблюдатель технического интервью.
Ты анализируешь ответы кандидата.

Твоя задача:
- оценить корректность ответа
- понять уверенность кандидата
- решить, что делать дальше

Ты НЕ общаешься с кандидатом напрямую.
Отвечай СТРОГО в JSON формате.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def analyze_answer(self, state: InterviewState, answer: str) -> dict:
        prompt = f"""
Ответ кандидата:
{answer}

История интервью:
{state.history}

Верни JSON следующего формата:
{{
  "verdict": "correct | partial | wrong",
  "confidence": "low | medium | high",
  "next_action": "deepen | simplify | change_topic | continue",
  "note": "краткое объяснение для интервьюера"
}}
"""
        raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.0)
        raw = raw.replace('```json', '').replace('```', '').replace('\n', '')

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "verdict": "partial",
                "confidence": "low",
                "next_action": "simplify",
                "note": "Ответ неструктурирован, требуется уточнение"
            }

        state.observer_notes.append(parsed["note"])

        if parsed["verdict"] == "correct" and parsed["confidence"] == "high":
            state.difficulty = min(5, state.difficulty + 1)
        elif parsed["verdict"] == "wrong":
            state.difficulty = max(1, state.difficulty - 1)

        if parsed["next_action"] == "change_topic":
            state.current_topic = "next_topic"

        return parsed
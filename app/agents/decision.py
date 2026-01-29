from app.state import InterviewState
from app.llm.mistral import MistralLLM


class DecisionAgent:
    SYSTEM_PROMPT = """
Ты — менеджер по найму.
Ты читаешь всё интервью целиком и выносишь финальное решение.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def make_decision(self, state: InterviewState) -> str:
        prompt = f"""
Позиция: {state.position}
Целевой уровень: {state.target_grade}

История интервью:
{state.history}

Наблюдения:
{state.observer_notes}

Сформируй структурированный отчет:
A. Decision
B. Technical Review
C. Soft Skills
D. Roadmap
"""
        return self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.2)
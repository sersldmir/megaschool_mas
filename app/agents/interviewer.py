from app.llm.mistral import MistralLLM


class InterviewerAgent:
    SYSTEM_PROMPT = """
Ты — технический интервьюер
Ты ведёшь интервью с кандидатом.
Ты просто задаёшь вопросы по инструкции, НЕ проверяя фактов и НЕ оценивая уровень.
"""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def ask_question(self, state: dict[str], instruction: str) -> str:
        prompt = f"""
Позиция: {state['position']}
Ожидаемый уровень: {state['target_grade']}
Опыт: {state['experience_years']} лет

Текущая тема: {state['current_topic']}
Сложность: {state['difficulty']}

Инструкция:
{instruction}

Сформулируй ОДИН вопрос кандидату.
"""
        return self.llm.chat(self.SYSTEM_PROMPT, prompt)
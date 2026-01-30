from app.llm.mistral import MistralLLM
from app.state import InterviewState


class InterviewerAgent:
    
    SYSTEM_PROMPT = """Ты — технический интервьюер на позицию разработчика.

Твоя роль:
- Задавать технические вопросы по указанной теме и уровню сложности
- Следовать инструкциям от команды по ведению интервью
- Быть профессиональным, но дружелюбным

Что ты НЕ делаешь:
- НЕ проверяешь правильность ответов (это делает другой член команды)
- НЕ принимаешь решения о найме
- НЕ даёшь фидбэк по ответам

Формат ответа: один чёткий технический вопрос без преамбулы."""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def ask_question(self, state: InterviewState, instruction: str) -> str:
        """Формирует вопрос на основе текущего состояния и инструкции."""
        
        recent_questions = []
        if state["dialog_history"]:
            recent_questions = [
                turn["question"] 
                for turn in state["dialog_history"][-3:]
            ]
        
        prompt = f"""Информация о кандидате:
Позиция: {state['position']}
Ожидаемый уровень: {state['target_grade']}
Опыт: {state['experience_years']} лет

Текущий контекст интервью:
Тема: {state['current_topic']}
Уровень сложности: {state['difficulty']}/5
Задано вопросов: {state['questions_asked']}
Правильных ответов: {state['correct_answers']}
Неправильных ответов: {state['wrong_answers']}

Последние заданные вопросы:
{chr(10).join(recent_questions) if recent_questions else "Это первый вопрос"}

Инструкция от команды:
{instruction}

Сформулируй ОДИН технический вопрос. Вопрос должен быть:
- Конкретным и проверяемым
- Соответствовать указанному уровню сложности
- Не повторять предыдущие вопросы
- По теме: {state['current_topic']}
"""
        
        return self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.4)

    def handle_greeting(self, state: InterviewState) -> str:
        """Формирует приветственное сообщение."""
        prompt = f"""Кандидат устраивается на позицию {state['position']} уровня {state['target_grade']}.
Опыт: {state['experience_years']} лет.

Поприветствуй кандидата и начни интервью. 
Скажи, что будешь задавать технические вопросы.
Объясни, что кандидат может честно сказать "не знаю", если не уверен.
Будь дружелюбным и профессиональным."""
        
        return self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.6)

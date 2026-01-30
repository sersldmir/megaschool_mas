import json
from app.llm.mistral import MistralLLM
from app.state import InterviewState


class ObserverAgent:
    
    SYSTEM_PROMPT = """Ты — наблюдатель и ментор на техническом интервью.

Твоя роль:
- Анализировать ответы кандидата на техническую корректность
- Обнаруживать попытки уйти от темы или манипулировать беседой
- Выявлять галлюцинации (уверенные, но ложные утверждения)
- Оценивать уровень понимания и уверенности кандидата
- Давать инструкции интервьюеру о дальнейших действиях

Что ты НЕ делаешь:
- НЕ общаешься с кандидатом напрямую
- НЕ принимаешь финальное решение о найме (это делает Hiring Manager)

Ты должен быть объективным, строгим, но справедливым.

ВАЖНО: Отвечай ТОЛЬКО валидным JSON без markdown форматирования."""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def analyze_answer(self, state: InterviewState, answer: str) -> dict:
        
        history_context = ""
        if state["dialog_history"]:
            history_context = "Предыдущие ответы:\n"
            for turn in state["dialog_history"][-3:]:
                history_context += f"Q: {turn['question'][:100]}...\n"
                history_context += f"A: {turn['answer'][:150]}...\n\n"
        
        prompt = f"""Анализируй ответ кандидата на техническое интервью.

Вопрос интервьюера:
{state["current_question"]}

Ответ кандидата:
{answer}

Контекст интервью:
Позиция: {state["position"]}
Уровень: {state["target_grade"]}
Опыт: {state["experience_years"]} лет
Текущая тема: {state["current_topic"]}
Сложность вопроса: {state["difficulty"]}/5
Статистика: {state["correct_answers"]} правильных, {state["wrong_answers"]} неправильных

{history_context}

Проанализируй ответ по следующим критериям:

1. РЕЛЕВАНТНОСТЬ: Отвечает ли кандидат на вопрос или пытается уйти от темы?
2. КОРРЕКТНОСТЬ: Правильный ли ответ с технической точки зрения?
3. ПОЛНОТА: Насколько глубоко кандидат понимает тему?
4. ЧЕСТНОСТЬ: Пытается ли выкрутиться или честно признаёт незнание?
5. ГАЛЛЮЦИНАЦИИ: Есть ли уверенные, но ложные утверждения?

Верни JSON следующей структуры:
{{
  "is_on_topic": true/false,
  "off_topic_reason": "если false - почему ушёл от темы",
  "verdict": "correct" | "partial" | "wrong" | "dont_know",
  "confidence": "low" | "medium" | "high",
  "hallucination_detected": true/false,
  "hallucination_details": "описание ложных утверждений, если есть",
  "internal_thought": "подробный анализ: что кандидат понял/не понял, какие сигналы заметил",
  "next_action": "deepen" | "simplify" | "change_topic" | "continue" | "return_to_topic",
  "next_topic_suggestion": "предложение темы для следующего вопроса",
  "difficulty_adjustment": -2 | -1 | 0 | 1 | 2,
  "note_to_interviewer": "конкретная инструкция для следующего вопроса",
  "knowledge_gap": "если ошибка - в чём конкретный пробел знаний",
  "expected_answer": "правильный ответ на вопрос",
  "soft_skill_signals": {{
    "clarity": "насколько понятно излагает",
    "honesty": "оценка честности",
    "engagement": "вовлечённость в беседу"
  }}
}}

Будь строгим, но объективным. Отвечай ТОЛЬКО JSON без markdown форматирования."""

        try:
            raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.1)
            
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            parsed = json.loads(raw)
            
            required_fields = [
                "is_on_topic", "verdict", "confidence", 
                "internal_thought", "next_action", "expected_answer"
            ]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON decode error: {e}")
            print(f"[WARNING] Raw response: {raw[:200]}...")
            return self._get_fallback_analysis(answer)
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            return self._get_fallback_analysis(answer)

    def _get_default_value(self, field: str):
        defaults = {
            "is_on_topic": True,
            "off_topic_reason": "",
            "verdict": "partial",
            "confidence": "low",
            "hallucination_detected": False,
            "hallucination_details": "",
            "internal_thought": "Не удалось проанализировать ответ",
            "next_action": "continue",
            "next_topic_suggestion": "basics",
            "difficulty_adjustment": 0,
            "note_to_interviewer": "Продолжай в том же духе",
            "knowledge_gap": "",
            "expected_answer": "",
            "soft_skill_signals": {
                "clarity": "medium",
                "honesty": "medium",
                "engagement": "medium"
            }
        }
        return defaults.get(field, "")

    def _get_fallback_analysis(self, answer: str) -> dict:
        answer_lower = answer.lower()
        
        is_short = len(answer.split()) < 5
        has_uncertainty = any(word in answer_lower for word in 
                             ["не знаю", "не уверен", "может быть", "наверное"])
        
        return {
            "is_on_topic": True,
            "off_topic_reason": "",
            "verdict": "partial" if has_uncertainty or is_short else "correct",
            "confidence": "low" if has_uncertainty else "medium",
            "hallucination_detected": False,
            "hallucination_details": "",
            "internal_thought": f"Ответ {'краткий' if is_short else 'развёрнутый'}. "
                              f"{'Есть признаки неуверенности.' if has_uncertainty else 'Звучит уверенно.'}",
            "next_action": "simplify" if has_uncertainty else "continue",
            "next_topic_suggestion": "basics",
            "difficulty_adjustment": -1 if has_uncertainty else 0,
            "note_to_interviewer": "Продолжай интервью",
            "knowledge_gap": "Требуется дополнительная проверка понимания темы",
            "expected_answer": "Детальный технический ответ с примерами",
            "soft_skill_signals": {
                "clarity": "low" if is_short else "medium",
                "honesty": "high" if has_uncertainty else "medium",
                "engagement": "low" if is_short else "medium"
            }
        }

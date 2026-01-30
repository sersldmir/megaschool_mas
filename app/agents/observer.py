import json
from typing import Any, Dict
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

    def analyze_answer(self, state: InterviewState, answer: str) -> Dict[str, Any]:
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
Отвечай ТОЛЬКО JSON без markdown форматирования."""

        try:
            parsed = self._chat_json_with_retry(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature_first=0.1,
                retries=2,
            )
            return self._normalize_analysis(parsed)
        except Exception:
            return self._get_fallback_analysis(answer)

    def _chat_json_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature_first: float,
        retries: int,
    ) -> Dict[str, Any]:
        last_raw = ""
        for attempt in range(retries + 1):
            temperature = temperature_first if attempt == 0 else 0.0
            raw = self.llm.chat(system_prompt, user_prompt, temperature=temperature)
            last_raw = raw
            cleaned = self._strip_code_fences(raw)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            user_prompt = (
                "Твой предыдущий ответ был НЕ валидным JSON. "
                "Верни ТОЛЬКО валидный JSON без markdown, без пояснений, строго по требуемой структуре.\n\n"
                f"Исходное задание:\n{user_prompt}\n\n"
                f"Твой предыдущий (ошибочный) ответ:\n{last_raw}"
            )

        raise ValueError("Failed to produce valid JSON from LLM")

    def _strip_code_fences(self, raw: str) -> str:
        s = raw.strip()
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _normalize_analysis(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
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
            "note_to_interviewer": "Продолжай интервью",
            "knowledge_gap": "",
            "expected_answer": "",
            "soft_skill_signals": {
                "clarity": "medium",
                "honesty": "medium",
                "engagement": "medium",
            },
        }

        out: Dict[str, Any] = {**defaults, **parsed}

        if out["verdict"] not in {"correct", "partial", "wrong", "dont_know"}:
            out["verdict"] = defaults["verdict"]

        if out["confidence"] not in {"low", "medium", "high"}:
            out["confidence"] = defaults["confidence"]

        if out["next_action"] not in {"deepen", "simplify", "change_topic", "continue", "return_to_topic"}:
            out["next_action"] = defaults["next_action"]

        if not isinstance(out.get("difficulty_adjustment"), int) or out["difficulty_adjustment"] not in {-2, -1, 0, 1, 2}:
            out["difficulty_adjustment"] = 0

        if not isinstance(out.get("soft_skill_signals"), dict):
            out["soft_skill_signals"] = defaults["soft_skill_signals"]

        return out

    def _get_fallback_analysis(self, answer: str) -> Dict[str, Any]:
        answer_lower = answer.lower()
        is_short = len(answer.split()) < 5
        has_uncertainty = any(word in answer_lower for word in ["не знаю", "не уверен", "может быть", "наверное"])

        return {
            "is_on_topic": True,
            "off_topic_reason": "",
            "verdict": "dont_know" if has_uncertainty else ("partial" if is_short else "correct"),
            "confidence": "low" if has_uncertainty else "medium",
            "hallucination_detected": False,
            "hallucination_details": "",
            "internal_thought": (
                f"Ответ {'краткий' if is_short else 'развёрнутый'}. "
                f"{'Есть признаки неуверенности.' if has_uncertainty else 'Звучит уверенно.'}"
            ),
            "next_action": "simplify" if has_uncertainty else "continue",
            "next_topic_suggestion": "basics",
            "difficulty_adjustment": -1 if has_uncertainty else 0,
            "note_to_interviewer": "Продолжай интервью",
            "knowledge_gap": "Требуется дополнительная проверка понимания темы" if is_short or has_uncertainty else "",
            "expected_answer": "Детальный технический ответ с примерами",
            "soft_skill_signals": {
                "clarity": "low" if is_short else "medium",
                "honesty": "high" if has_uncertainty else "medium",
                "engagement": "low" if is_short else "medium",
            },
        }
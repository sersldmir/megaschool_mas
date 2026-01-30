from app.llm.mistral import MistralLLM
from app.state import InterviewState
import json


class DecisionAgent:
    
    SYSTEM_PROMPT = """Ты — Hiring Manager с многолетним опытом найма IT-специалистов всех видов.

Твоя роль:
- Проанализировать результаты технического интервью
- Принять обоснованное решение о найме
- Дать конструктивный фидбэк кандидату
- Составить персональный план развития

Ты должен быть:
- Объективным и справедливым
- Конкретным в оценках
- Полезным в рекомендациях

ВАЖНО: Отвечай ТОЛЬКО валидным JSON без markdown форматирования."""

    def __init__(self, llm: MistralLLM):
        self.llm = llm

    def make_decision(self, state: InterviewState) -> dict:
        
        total_answers = state["correct_answers"] + state["wrong_answers"] + state["partial_answers"]
        success_rate = 0
        if total_answers > 0:
            success_rate = round(
                (state["correct_answers"] + 0.5 * state["partial_answers"]) / total_answers * 100
            )
        
        avg_confidence = 0
        if state["confidence_scores"]:
            avg_confidence = round(sum(state["confidence_scores"]) / len(state["confidence_scores"]))
        
        dialog_summary = []
        for i, turn in enumerate(state["dialog_history"][-10:], 1):
            dialog_summary.append(
                f"{i}. Q: {turn['question'][:80]}...\n"
                f"   A: {turn['answer'][:100]}...\n"
                f"   Topic: {turn['topic']}, Difficulty: {turn['difficulty']}"
            )
        
        gaps_with_answers = []
        for topic, expected in state["knowledge_gaps"].items():
            gaps_with_answers.append(f"Тема: {topic}\nПравильный ответ: {expected}")
        
        prompt = f"""Проанализируй результаты технического интервью и прими решение о найме.

ИНФОРМАЦИЯ О КАНДИДАТЕ:
Позиция: {state["position"]}
Целевой уровень: {state["target_grade"]}
Заявленный опыт: {state["experience_years"]} лет

СТАТИСТИКА ИНТЕРВЬЮ:
Всего вопросов задано: {state["questions_asked"]}
Правильных ответов: {state["correct_answers"]}
Частично правильных: {state["partial_answers"]}
Неправильных ответов: {state["wrong_answers"]}
Попыток уйти от темы: {state["off_topic_attempts"]}
Success Rate: {success_rate}%
Средняя уверенность: {avg_confidence}%

ПОДТВЕРЖДЁННЫЕ НАВЫКИ:
{', '.join(state["confirmed_skills"]) if state["confirmed_skills"] else "Нет явно подтверждённых"}

ПРОБЕЛЫ В ЗНАНИЯХ:
{chr(10).join(gaps_with_answers) if gaps_with_answers else "Пробелов не выявлено"}

ПОКРЫТЫЕ ТЕМЫ:
{', '.join(state["topics_covered"]) if state["topics_covered"] else "Только базовые темы"}

ИСТОРИЯ ВОПРОСОВ И ОТВЕТОВ (последние 10):
{chr(10).join(dialog_summary)}

ВНУТРЕННИЕ ЗАМЕТКИ НАБЛЮДАТЕЛЯ (ключевые инсайты):
{chr(10).join(state["observer_notes"][-5:]) if state["observer_notes"] else "Нет заметок"}

На основе этих данных сформируй финальное решение в следующем JSON формате:

{{
  "hiring_recommendation": "Strong Hire" | "Hire" | "No Hire" | "Strong No Hire",
  "recommendation_reasoning": "детальное обоснование решения",
  "final_grade": "Junior" | "Middle" | "Senior" | "Below Junior",
  "grade_reasoning": "почему именно этот грейд",
  "confidence_score": 0-100,
  
  "hard_skills": {{
    "confirmed": ["список подтверждённых технических навыков"],
    "gaps": ["список выявленных пробелов"]
  }},
  
  "soft_skills": {{
    "clarity": "low" | "medium" | "high",
    "clarity_note": "комментарий о ясности изложения",
    "honesty": "low" | "medium" | "high",
    "honesty_note": "комментарий о честности",
    "engagement": "low" | "medium" | "high",
    "engagement_note": "комментарий о вовлечённости",
    "problem_solving": "low" | "medium" | "high",
    "problem_solving_note": "комментарий о подходе к решению задач"
  }},
  
  "knowledge_gaps_detailed": [
    {{
      "topic": "конкретная тема",
      "what_missing": "что именно не знает",
      "expected_answer": "правильный ответ на вопрос",
      "importance": "high" | "medium" | "low"
    }}
  ],
  
  "strengths": ["список сильных сторон кандидата"],
  "weaknesses": ["список слабых сторон кандидата"],
  
  "roadmap": [
    {{
      "priority": "high" | "medium" | "low",
      "topic": "что изучать",
      "reason": "почему это важно",
      "resources": ["конкретные ресурсы: книги, курсы, документация"]
    }}
  ],
  
  "interview_notes": "общие впечатления от интервью"
}}

Будь объективным, конструктивным и полезным. Твой фидбэк должен помочь кандидату расти.
Отвечай ТОЛЬКО JSON без markdown форматирования."""

        try:
            raw = self.llm.chat(self.SYSTEM_PROMPT, prompt, temperature=0.2)
            
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            parsed_json = json.loads(raw)
            
            parsed_json["meta"] = {
                "total_questions": state["questions_asked"],
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "topics_covered": len(state["topics_covered"]),
                "off_topic_attempts": state["off_topic_attempts"]
            }
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse decision JSON: {e}")
            print(f"[ERROR] Raw response: {raw[:300]}...")
            return self._get_fallback_decision(state, success_rate, avg_confidence)
        except Exception as e:
            print(f"[ERROR] Decision making failed: {e}")
            return self._get_fallback_decision(state, success_rate, avg_confidence)

    def _get_fallback_decision(self, state: InterviewState, success_rate: int, avg_confidence: int) -> dict:
        
        if success_rate >= 70 and avg_confidence >= 60:
            recommendation = "Hire"
            grade = state["target_grade"]
        elif success_rate >= 50:
            recommendation = "Hire"
            grade = "Junior" if state["target_grade"] == "Middle" else state["target_grade"]
        else:
            recommendation = "No Hire"
            grade = "Below Junior"
        
        return {
            "hiring_recommendation": recommendation,
            "recommendation_reasoning": f"На основе {success_rate}% успешных ответов",
            "final_grade": grade,
            "grade_reasoning": "Автоматическая оценка на основе статистики",
            "confidence_score": avg_confidence,
            "hard_skills": {
                "confirmed": state["confirmed_skills"],
                "gaps": list(state["knowledge_gaps"].keys())
            },
            "soft_skills": {
                "clarity": "medium",
                "clarity_note": "Требуется ручная оценка",
                "honesty": "medium",
                "honesty_note": "Требуется ручная оценка",
                "engagement": "medium",
                "engagement_note": "Требуется ручная оценка",
                "problem_solving": "medium",
                "problem_solving_note": "Требуется ручная оценка"
            },
            "knowledge_gaps_detailed": [
                {
                    "topic": topic,
                    "what_missing": "Детали требуют анализа",
                    "expected_answer": answer,
                    "importance": "medium"
                }
                for topic, answer in state["knowledge_gaps"].items()
            ],
            "strengths": ["Требуется ручной анализ"],
            "weaknesses": ["Требуется ручной анализ"],
            "roadmap": [
                {
                    "priority": "high",
                    "topic": topic,
                    "reason": "Выявлен пробел в знаниях",
                    "resources": ["Официальная документация"]
                }
                for topic in list(state["knowledge_gaps"].keys())[:3]
            ],
            "interview_notes": "Автоматически сгенерированное решение (fallback)",
            "meta": {
                "total_questions": state["questions_asked"],
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "topics_covered": len(state["topics_covered"]),
                "off_topic_attempts": state["off_topic_attempts"]
            }
        }

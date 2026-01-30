from app.state import InterviewPhase, InterviewState
from app.llm.mistral import MistralLLM
from app.graph.workflow import build_graph
from app.session_logger import SessionLogger
from datetime import datetime
import json
import os
import re

STOP_WORDS = {
    "стоп интервью",
    "остановка интервью",
    "стоп",
    "завершить интервью",
    "хватит",
    "достаточно",
}

def normalize_stop_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def print_separator(char="=", length=60):
    print(char * length)

def print_section(title: str):
    print_separator()
    print(f" {title}")
    print_separator()

def format_feedback(feedback: dict) -> str:
    if not feedback:
        return "[FEEDBACK] Фидбэк не был сгенерирован"

    lines = []

    lines.append("\n[VERDICT] ВЕРДИКТ")
    lines.append("-" * 60)
    lines.append(f"Рекомендация: {feedback.get('hiring_recommendation', 'N/A')}")
    lines.append(f"Финальный грейд: {feedback.get('final_grade', 'N/A')}")
    lines.append(f"Уверенность: {feedback.get('confidence_score', 0)}%")

    if feedback.get("recommendation_reasoning"):
        lines.append(f"\nОбоснование: \n{feedback['recommendation_reasoning']}")

    lines.append("\n[HARD SKILLS] ТЕХНИЧЕСКИЕ НАВЫКИ")
    lines.append("-" * 60)

    hard_skills = feedback.get("hard_skills", {})
    confirmed = hard_skills.get("confirmed", [])
    gaps = hard_skills.get("gaps", [])

    if confirmed:
        lines.append("[CONFIRMED] Подтверждённые:")
        for skill in confirmed:
            lines.append(f" + {skill}")

    if gaps:
        lines.append("\n[GAPS] Пробелы:")
        for gap in gaps:
            lines.append(f" - {gap}")

    lines.append("\n[SOFT SKILLS] SOFT SKILLS")
    lines.append("-" * 60)

    soft_skills = feedback.get("soft_skills", {})
    for skill_name in ["clarity", "honesty", "engagement", "problem_solving"]:
        if skill_name in soft_skills:
            value = soft_skills[skill_name]
            note = soft_skills.get(f"{skill_name}_note", "")
            marker = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}.get(value, "[N/A]")
            lines.append(f"{marker} {skill_name.capitalize()}: {value}")
            if note:
                lines.append(f" {note}")

    gaps_detailed = feedback.get("knowledge_gaps_detailed", [])
    if gaps_detailed:
        lines.append("\n[KNOWLEDGE GAPS] ДЕТАЛЬНЫЙ АНАЛИЗ ПРОБЕЛОВ")
        lines.append("-" * 60)
        for gap in gaps_detailed[:5]:
            lines.append(f"\n[TOPIC] {gap.get('topic', 'Unknown')}")
            lines.append(f" Что упущено: {gap.get('what_missing', 'N/A')}")
            lines.append(f" Важность: {gap.get('importance', 'medium')}")
            if gap.get("expected_answer"):
                lines.append(f" Правильный ответ: {gap['expected_answer'][:150]}...")

    strengths = feedback.get("strengths", [])
    weaknesses = feedback.get("weaknesses", [])

    if strengths:
        lines.append("\n[STRENGTHS] СИЛЬНЫЕ СТОРОНЫ")
        lines.append("-" * 60)
        for strength in strengths:
            lines.append(f" [+] {strength}")

    if weaknesses:
        lines.append("\n[WEAKNESSES] СЛАБЫЕ СТОРОНЫ")
        lines.append("-" * 60)
        for weakness in weaknesses:
            lines.append(f" [-] {weakness}")

    roadmap = feedback.get("roadmap", [])
    if roadmap:
        lines.append("\n[ROADMAP] ПЕРСОНАЛЬНЫЙ ROADMAP")
        lines.append("-" * 60)
        for i, item in enumerate(roadmap, 1):
            priority = item.get("priority", "medium")
            marker = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(priority, "[N/A]")
            lines.append(f"\n{i}. {marker} {item.get('topic', 'Unknown')}")
            lines.append(f" Причина: {item.get('reason', 'N/A')}")
            resources = item.get("resources", [])
            if resources:
                lines.append(" Ресурсы:")
                for resource in resources[:3]:
                    lines.append(f" * {resource}")

    if feedback.get("interview_notes"):
        lines.append("\n[NOTES] ОБЩИЕ ВПЕЧАТЛЕНИЯ")
        lines.append("-" * 60)
        lines.append(feedback["interview_notes"])

    meta = feedback.get("meta", {})
    if meta:
        lines.append("\n[META] МЕТАДАННЫЕ")
        lines.append("-" * 60)
        lines.append(f"Задано вопросов: {meta.get('total_questions', 0)}")
        lines.append(f"Success Rate: {meta.get('success_rate', 0)}%")
        lines.append(f"Средняя уверенность: {meta.get('avg_confidence', 0)}%")
        lines.append(f"Покрыто тем: {meta.get('topics_covered', 0)}")
        lines.append(f"Попытки ухода от темы: {meta.get('off_topic_attempts', 0)}")

    return "\n".join(lines)

def main():
    print_section("[START] MULTI-AGENT INTERVIEW COACH")

    print("\nВведите информацию о кандидате:")
    team_name = input("Название команды: ").strip() or "Default Team"
    position = input("Позиция (например, Python Developer): ").strip() or "Python Developer"
    grade = input("Целевой грейд (Junior/Middle/Senior): ").strip() or "Middle"
    experience_str = input("Опыт работы в годах: ").strip()

    try:
        experience = int(experience_str)
    except ValueError:
        print("[WARNING] Некорректное значение опыта, устанавливаю 1 год")
        experience = 1

    max_questions = 10

    state: InterviewState = {
        "team_name": team_name,
        "position": position,
        "target_grade": grade,
        "experience_years": experience,
        "phase": InterviewPhase.ASKING_QUESTION,
        "current_question": None,
        "current_answer": None,
        "current_instruction": None,
        "dialog_history": [],
        "observer_notes": [],
        "confirmed_skills": [],
        "knowledge_gaps": {},
        "confidence_scores": [],
        "current_topic": "Python basics",
        "difficulty": 2,
        "correct_answers": 0,
        "wrong_answers": 0,
        "partial_answers": 0,
        "off_topic_attempts": 0,
        "final_feedback": None,
        "last_observer_note": None,
        "last_analysis": None,
        "topics_covered": [],
        "questions_asked": 0,
        "max_questions": max_questions,
        "greeting_done": False,
        "current_question_difficulty": 2,
        "answered_turns": 0,
    }

    required_keys = ["team_name", "position", "target_grade", "experience_years"]
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise RuntimeError(f"Initial state missing keys: {missing}")

    try:
        llm = MistralLLM()
        graph = build_graph(llm)
        logger = SessionLogger(state["team_name"])
    except Exception as e:
        print(f"\n[ERROR] Ошибка инициализации: {e}")
        print("Проверьте наличие .env файла с MISTRAL_API_KEY")
        return

    print_section("[INTERVIEW] НАЧАЛО ИНТЕРВЬЮ")
    print("Для завершения интервью введите 'стоп интервью'\n")

    iteration = 0
    max_iterations = max_questions * 3

    while state["phase"] != InterviewPhase.FINISHED and iteration < max_iterations:
        iteration += 1

        try:
            state = graph.invoke(state)

            if state["phase"] == InterviewPhase.WAITING_FOR_ANSWER:
                print(f"\n[INTERVIEWER] Интервьюер: \n{state['current_question']}")
                print(f" [INFO] Тема: {state['current_topic']}, Сложность: {state['difficulty']}/5")

                user_answer = input("\n[YOU] Ваш ответ: ").strip()

                if normalize_stop_text(user_answer) in STOP_WORDS:
                    print("\n[STOP] Интервью остановлено по запросу кандидата.")
                    state["phase"] = InterviewPhase.DECISION
                    state = graph.invoke(state)
                    break

                if not user_answer:
                    print("[WARNING] Пустой ответ расценивается как 'не знаю'")
                    user_answer = "Я не знаю ответа на этот вопрос"

                state["current_answer"] = user_answer
                state["phase"] = InterviewPhase.ANALYZING_ANSWER

                state = graph.invoke(state)

                analysis = state.get("last_analysis") or {}
                observer_thought = analysis.get("internal_thought", "Анализирую ответ...")
                interviewer_instruction = state.get("last_observer_note", "Продолжай интервью.")

                internal_thoughts = f"[Observer]: {observer_thought}\n[Interviewer]: {interviewer_instruction}"

                logger.log_turn(
                    agent_message=state["current_question"],
                    user_message=user_answer,
                    internal_thoughts=internal_thoughts,
                    internal={
                        "observer_analysis": analysis,
                        "note_to_interviewer": interviewer_instruction,
                        "current_instruction": state.get("current_instruction"),
                        "current_topic": state.get("current_topic"),
                        "difficulty": state.get("difficulty"),
                    },
                )

        except KeyboardInterrupt:
            print("\n\n[STOP] Интервью прервано пользователем (Ctrl+C)")
            state["phase"] = InterviewPhase.DECISION
            state = graph.invoke(state)
            break
        except Exception as e:
            print(f"\n[ERROR] Ошибка во время интервью: {e}")
            import traceback
            traceback.print_exc()
            break

    if iteration >= max_iterations:
        print("\n[WARNING] Достигнут лимит итераций, завершаем интервью.")
        state["phase"] = InterviewPhase.DECISION
        state = graph.invoke(state)

    print_section("[FEEDBACK] ФИНАЛЬНЫЙ ФИДБЭК")

    if state.get("final_feedback"):
        print(format_feedback(state["final_feedback"]))
        logger.log_event("final_feedback_generated", {"final_feedback": state["final_feedback"]})
    else:
        print("[ERROR] Фидбэк не был сгенерирован")
        logger.log_event("final_feedback_missing", {"reason": "final_feedback is None"})

    print_section("[SAVE] СОХРАНЕНИЕ ЛОГОВ")

    try:
        exported = logger.export(state)

        log_dir = "./app/log_data"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/interview_log_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(exported, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Лог сохранён: {filename}")

        print(f"\n[STATS] Статистика интервью:")
        print(f" Всего вопросов: {state['questions_asked']}")
        print(f" Правильных: {state['correct_answers']}")
        print(f" Частично правильных: {state['partial_answers']}")
        print(f" Неправильных: {state['wrong_answers']}")
        print(f" Попыток ухода от темы: {state['off_topic_attempts']}")
        print(f" Покрыто тем: {len(state['topics_covered'])}")

    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении логов: {e}")

    print_section("[END] ИНТЕРВЬЮ ЗАВЕРШЕНО")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
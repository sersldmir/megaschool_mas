from app.state import InterviewState
from app.llm.mistral import MistralLLM
from app.graph.workflow import build_graph
from app.session_logger import SessionLogger
import json
from datetime import datetime

STOP_WORDS = [
    "стоп интервью",
    "остановка интервью",
    "стоп",
    "я ухожу",
    "интервью окончено"
]

def main():
    print("=== Начало интервью ===")
    team_name = input("Название команды: ")
    position = input("Позиция: ")
    grade = input("Грейд: ")
    experience = int(input("Опыт (в годах): "))

    state = InterviewState(
        team_name=team_name,
        position=position,
        target_grade=grade,
        experience_years=experience
    )

    llm = MistralLLM()
    graph = build_graph(llm)
    logger = SessionLogger(state)

    print("\n=== Интервью в процессе ===")

    while not state.finished:
        state = graph.invoke(state)

        question = state.current_question
        print(f"\nИнтервьюер: {question}")

        user_answer = input("Ваш ответ: ").strip()

        if user_answer.lower() in STOP_WORDS:
            state.finished = True
            continue

        state.current_answer = user_answer
        state.history.append({
            "question": question,
            "answer": user_answer
        })

        state = graph.invoke(state)

        internal_thoughts = ""
        if state.last_analysis:
            internal_thoughts = (
                f"[Observer]: {state.last_analysis.get('note', '')}"
            )

        logger.log_turn(
            agent_message=question,
            user_message=user_answer,
            internal_thoughts=internal_thoughts
        )

        if not state.final_feedback:
            state = graph.invoke(state)

    print("\n=== Фидбэк по интервью ===")
    print(state.final_feedback)

    print("\n=== Лог интервью ===")
    print(logger.export())
    with open(f"log_data/interview_state_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
        json.dump(state, f, indent=4)


if __name__ == "__main__":
    main()
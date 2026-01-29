from app.state import InterviewState, InterviewPhase
from app.llm.mistral import MistralLLM
from app.graph.workflow import build_graph
from app.session_logger import SessionLogger
from datetime import datetime
import json

STOP_WORDS = {
    "стоп интервью",
    "остановка интервью",
    "стоп",
    "я ухожу",
    "интервью окончено",
}


def main():
    print("=== Начало интервью ===")
    team_name = input("Название команды: ").strip()
    position = input("Позиция: ").strip()
    grade = input("Грейд: ").strip()
    experience = int(input("Опыт (в годах): ").strip())

    state = InterviewState(
        team_name=team_name,
        position=position,
        target_grade=grade,
        experience_years=experience,
    )

    llm = MistralLLM()
    graph = build_graph(llm)
    logger = SessionLogger(state)

    print("\n=== Интервью в процессе ===")

    while state['phase'] != InterviewPhase.FINISHED:

        state = graph.invoke(state)

        if state['phase'] == InterviewPhase.WAITING_FOR_ANSWER:
            print(f"\nИнтервьюер: {state['current_question']}")
            user_answer = input("Ваш ответ: ").strip()

            if user_answer.lower() in STOP_WORDS:
                state['phase'] = InterviewPhase.DECISION
                continue

            state['current_answer'] = user_answer
            state['dialog_history'].append({
                "question": state['current_question'],
                "answer": user_answer,
            })

            logger.log_turn(
                agent_message=state['current_question'],
                user_message=user_answer,
                internal_thoughts=(state['observer_notes'][-1]
                                  if state['observer_notes'] else "")
            )

            state['phase'] = InterviewPhase.ANALYZING_ANSWER

    print("\n=== Фидбэк по интервью ===")
    print(state['final_feedback'])

    print("\n=== Лог интервью ===")
    exported = logger.export()
    print(exported)

    filename = f"log_data/interview_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
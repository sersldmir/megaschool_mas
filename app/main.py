from app.state import InterviewState
from app.llm.mistral import MistralLLM
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent


def main():
    state = InterviewState(
        position="Data Engineer",
        target_grade="Middle",
        experience_years=3
    )

    llm = MistralLLM()
    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)

    instruction = "Начни интервью с базового вопроса по SQL."
    question = interviewer.ask_question(state, instruction)
    print("QUESTION:", question)

    answer = input("\nANSWER: ")

    state.history.append({
        "question": question,
        "answer": answer
    })

    analysis = observer.analyze_answer(state, answer)

    print("\n[HIDDEN OBSERVER NOTE]")
    print(analysis)


if __name__ == "__main__":
    main()
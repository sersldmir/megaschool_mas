from langgraph.graph import StateGraph, END
from app.state import InterviewState, InterviewPhase
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent
from app.agents.decision import DecisionAgent


def build_graph(llm):
    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)
    decision = DecisionAgent(llm)

    def interviewer_node(state: InterviewState) -> InterviewState:
        if state.phase != InterviewPhase.ASKING_QUESTION:
            return state

        instruction = "Продолжай интервью по текущей теме."
        question = interviewer.ask_question(state, instruction)

        state.current_question = question
        state.phase = InterviewPhase.WAITING_FOR_ANSWER
        return state

    def observer_node(state: InterviewState) -> InterviewState:
        if state.phase != InterviewPhase.ANALYZING_ANSWER:
            return state

        analysis = observer.analyze_answer(state, state.current_answer)
        state.observer_notes.append(analysis["note"])

        if analysis["verdict"] == "correct":
            state.correct_answers += 1
            if analysis["confidence"] == "high":
                state.difficulty = min(5, state.difficulty + 1)
        else:
            state.wrong_answers += 1
            state.difficulty = max(1, state.difficulty - 1)

        if analysis["next_action"] == "change_topic":
            state.current_topic = "next_topic"

        if state.correct_answers + state.wrong_answers >= 5:
            state.phase = InterviewPhase.DECISION
        else:
            state.phase = InterviewPhase.ASKING_QUESTION

        return state

    def decision_node(state: InterviewState) -> InterviewState:
        if state.phase != InterviewPhase.DECISION:
            return state

        state.final_feedback = decision.make_decision(state)
        state.phase = InterviewPhase.FINISHED
        return state

    graph = StateGraph(InterviewState)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("observer", observer_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("interviewer")

    graph.add_conditional_edges(
        "interviewer",
        lambda s: (
            END if s.phase == InterviewPhase.WAITING_FOR_ANSWER
            else "decision" if s.phase == InterviewPhase.DECISION
            else "observer"
        ),
    )

    graph.add_conditional_edges(
        "observer",
        lambda s: (
            "decision" if s.phase == InterviewPhase.DECISION
            else "interviewer"
        ),
    )

    graph.add_edge("decision", END)

    return graph.compile()
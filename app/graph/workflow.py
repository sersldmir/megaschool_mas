from langgraph.graph import StateGraph, END
from app.state import InterviewPhase
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent
from app.agents.decision import DecisionAgent


def build_graph(llm):
    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)
    decision = DecisionAgent(llm)

    def interviewer_node(state: dict) -> dict:
        if state["phase"] != InterviewPhase.ASKING_QUESTION:
            return state

        instruction = state.get(
            "current_instruction",
            "Продолжай интервью по текущей теме."
        )

        question = interviewer.ask_question(
            state=state,
            instruction=instruction
        )

        state["current_question"] = question
        state["phase"] = InterviewPhase.WAITING_FOR_ANSWER
        return state

    def observer_node(state: dict) -> dict:
        if state["phase"] != InterviewPhase.ANALYZING_ANSWER:
            return state

        analysis = observer.analyze_answer(
            state=state,
            answer=state["current_answer"]
        )

        state["observer_notes"].append(
            f"[Observer reasoning] {analysis['internal_thought']}"
        )

        state["last_observer_note"] = analysis["note"]
        state["current_instruction"] = analysis["next_action"]

        if analysis["verdict"] == "correct":
            state["correct_answers"] += 1
            state["confirmed_skills"].append(state["current_topic"])
        else:
            state["wrong_answers"] += 1
            state["knowledge_gaps"][state["current_topic"]] = analysis["expected_answer"]

        # difficulty adaption
        if analysis["confidence"] == "high":
            state["difficulty"] = min(5, state["difficulty"] + 1)
            state["confidence_scores"].append(90)
        elif analysis["confidence"] == "medium":
            state["confidence_scores"].append(60)
        else:
            state["difficulty"] = max(1, state["difficulty"] - 1)
            state["confidence_scores"].append(30)

        state["current_instruction"] = analysis["next_action"]

        # stop condition
        if state["correct_answers"] + state["wrong_answers"] >= 5:
            state["phase"] = InterviewPhase.DECISION
        else:
            state["phase"] = InterviewPhase.ASKING_QUESTION

        return state

    def decision_node(state: dict) -> dict:
        if state["phase"] != InterviewPhase.DECISION:
            return state

        state["final_feedback"] = decision.make_decision(state)
        state["phase"] = InterviewPhase.FINISHED
        return state

    graph = StateGraph(dict)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("observer", observer_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("interviewer")

    graph.add_conditional_edges(
        "interviewer",
        lambda s: (
            "observer" if s["phase"] == InterviewPhase.ANALYZING_ANSWER
            else "interviewer" if s["phase"] == InterviewPhase.ASKING_QUESTION
            else END
        ),
    )

    graph.add_conditional_edges(
        "observer",
        lambda s: (
            "decision" if s["phase"] == InterviewPhase.DECISION
            else "interviewer"
        ),
    )

    graph.add_edge("decision", END)

    return graph.compile()
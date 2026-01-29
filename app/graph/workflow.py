from langgraph.graph import StateGraph
from app.state import InterviewState
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent
from app.agents.decision import DecisionAgent


def build_graph(llm):
    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)
    decision = DecisionAgent(llm)

    def interviewer_node(state: InterviewState) -> InterviewState:
        instruction = "Продолжай интервью по текущей теме."
        question = interviewer.ask_question(state, instruction)
        state.current_question = question
        return state
    
    def observer_node(state: InterviewState) -> InterviewState:
        answer = state.current_answer
        analysis = observer.analyze_answer(state, answer)
        state.last_analysis = analysis
        return state

    def decision_node(state: InterviewState) -> InterviewState:
        state.final_feedback = decision.make_decision(state)
        state.finished = True
        return state

    graph = StateGraph(InterviewState)

    graph.add_node("interviewer", interviewer_node)
    graph.add_node("observer", observer_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("interviewer")

    graph.add_edge("interviewer", "observer")
    graph.add_conditional_edges(
        "observer",
        lambda s: "decision" if s.finished else "interviewer"
    )

    return graph.compile()
from langgraph.graph import StateGraph, END
from schemas import InterviewState
from agents import (
    profile_parser_node, 
    mentor_node, 
    interviewer_node, 
    feedback_node
)
from config import MAX_TURNS

workflow = StateGraph(InterviewState)

workflow.add_node("profile_parser", profile_parser_node)
workflow.add_node("mentor", mentor_node)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("feedback", feedback_node)


def route_step(state: InterviewState):

    if not state.get("profile"):
        return "profile_parser"
    
    last_msg = state.get("last_user_input", "").strip().lower()
    stop_words = ["стоп", "stop", "конец", "хватит", "exit"]
    
    if any(word in last_msg for word in stop_words):
        return "feedback"
    
    if state.get("turn_count", 0) >= MAX_TURNS:
        return "feedback"
    
    return "mentor"

workflow.set_conditional_entry_point(
    route_step,
    {
        "profile_parser": "profile_parser",
        "mentor": "mentor",
        "feedback": "feedback"
    }
)

workflow.add_edge("profile_parser", "interviewer")
workflow.add_edge("mentor", "interviewer")
workflow.add_edge("interviewer", END)
workflow.add_edge("feedback", END)

app = workflow.compile()
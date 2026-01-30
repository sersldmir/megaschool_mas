from langgraph.graph import StateGraph, END
from app.state import InterviewPhase, InterviewState
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent
from app.agents.decision import DecisionAgent


def build_graph(llm):

    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)
    decision = DecisionAgent(llm)
    
    def greeting_node(state: InterviewState) -> InterviewState:

        if state["questions_asked"] == 0:
            greeting = interviewer.handle_greeting(state)
            state["current_question"] = greeting
            state["phase"] = InterviewPhase.WAITING_FOR_ANSWER
        return state
    
    def interviewer_node(state: InterviewState) -> InterviewState:

        if state["phase"] != InterviewPhase.ASKING_QUESTION:
            return state
        
        instruction = state.get("current_instruction", "")
        
        if not instruction or instruction == "continue":
            instruction = f"Задай вопрос по теме '{state['current_topic']}' уровня сложности {state['difficulty']}/5"
        elif instruction == "deepen":
            instruction = f"Углуби вопрос по теме '{state['current_topic']}', увеличь сложность до {min(5, state['difficulty'] + 1)}"
        elif instruction == "simplify":
            instruction = f"Упрости вопрос по теме '{state['current_topic']}', понизь сложность до {max(1, state['difficulty'] - 1)}"
        elif instruction == "change_topic":
            next_topic = state.get("next_topic_suggestion", "algorithms")
            state["current_topic"] = next_topic
            instruction = f"Перейди к новой теме: '{next_topic}', начни с базового вопроса"
        elif instruction == "return_to_topic":
            instruction = f"Вежливо верни беседу к теме интервью: '{state['current_topic']}'"
        
        question = interviewer.ask_question(state, instruction)
        
        state["current_question"] = question
        state["questions_asked"] += 1
        state["phase"] = InterviewPhase.WAITING_FOR_ANSWER
        
        return state
    
    def observer_node(state: InterviewState) -> InterviewState:

        if state["phase"] != InterviewPhase.ANALYZING_ANSWER:
            return state
        
        answer = state["current_answer"]
        analysis = observer.analyze_answer(state, answer)
        
        state["last_analysis"] = analysis
        
        thought = (
            f"[Observer Analysis] {analysis['internal_thought']}\n"
            f"Verdict: {analysis['verdict']}, Confidence: {analysis['confidence']}\n"
            f"Action: {analysis['next_action']}"
        )
        state["observer_notes"].append(thought)
        state["last_observer_note"] = analysis["note_to_interviewer"]
        
        if not analysis["is_on_topic"]:
            state["off_topic_attempts"] += 1
            state["current_instruction"] = "return_to_topic"
            state["observer_notes"].append(
                f"[Off-topic detected] {analysis['off_topic_reason']}"
            )
        else:
            state["current_instruction"] = analysis["next_action"]
            if "next_topic_suggestion" in analysis and analysis["next_topic_suggestion"]:
                state["next_topic_suggestion"] = analysis["next_topic_suggestion"]
        
        if analysis.get("hallucination_detected", False):
            state["observer_notes"].append(
                f"[Hallucination] {analysis.get('hallucination_details', 'Detected false claims')}"
            )
        
        verdict = analysis["verdict"]
        if verdict == "correct":
            state["correct_answers"] += 1
            if state["current_topic"] not in state["confirmed_skills"]:
                state["confirmed_skills"].append(state["current_topic"])
        elif verdict == "partial":
            state["partial_answers"] += 1
        elif verdict == "wrong":
            state["wrong_answers"] += 1
            if analysis.get("knowledge_gap"):
                state["knowledge_gaps"][analysis["knowledge_gap"]] = analysis.get("expected_answer", "")
        
        difficulty_change = analysis.get("difficulty_adjustment", 0)
        state["difficulty"] = max(1, min(5, state["difficulty"] + difficulty_change))
        
        state["dialog_history"].append({
            "question": state["current_question"],
            "answer": answer,
            "topic": state["current_topic"],
            "difficulty": state["difficulty"]
        })
        
        conf_map = {"low": 30, "medium": 60, "high": 90}
        state["confidence_scores"].append(conf_map.get(analysis["confidence"], 50))
        
        if state["current_topic"] not in state["topics_covered"]:
            state["topics_covered"].append(state["current_topic"])
        
        total_answers = state["correct_answers"] + state["wrong_answers"] + state["partial_answers"]
        
        if total_answers >= state["max_questions"]:
            state["phase"] = InterviewPhase.DECISION
        elif state["off_topic_attempts"] >= 3:
            state["observer_notes"].append(
                "[Decision] Stopping interview due to repeated off-topic attempts"
            )
            state["phase"] = InterviewPhase.DECISION
        else:
            state["phase"] = InterviewPhase.ASKING_QUESTION
        
        return state
    
    def decision_node(state: InterviewState) -> InterviewState:
        if state["phase"] != InterviewPhase.DECISION:
            return state
        
        feedback = decision.make_decision(state)
        state["final_feedback"] = feedback
        state["phase"] = InterviewPhase.FINISHED
        
        return state
    
    def router(state: InterviewState) -> str:
        phase = state["phase"]
        
        if phase == InterviewPhase.WAITING_FOR_ANSWER:
            return END
        elif phase == InterviewPhase.ANALYZING_ANSWER:
            return "observer"
        elif phase == InterviewPhase.ASKING_QUESTION:
            return "interviewer"
        elif phase == InterviewPhase.DECISION:
            return "decision"
        elif phase == InterviewPhase.FINISHED:
            return END
        else:
            return END
    
    graph = StateGraph(InterviewState)
    
    graph.add_node("greeting", greeting_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("observer", observer_node)
    graph.add_node("decision", decision_node)
    
    graph.set_entry_point("greeting")
    
    graph.add_conditional_edges(
        "greeting",
        router
    )
    
    graph.add_conditional_edges(
        "interviewer",
        router
    )
    
    graph.add_conditional_edges(
        "observer",
        router
    )
    
    graph.add_conditional_edges(
        "decision",
        router
    )
    
    return graph.compile()

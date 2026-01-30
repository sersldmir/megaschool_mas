from langgraph.graph import StateGraph, END
from app.state import InterviewPhase, InterviewState
from app.agents.interviewer import InterviewerAgent
from app.agents.observer import ObserverAgent
from app.agents.decision import DecisionAgent

def _ensure_state_defaults(state: dict) -> dict:
    base = {
        "team_name": "Default Team",
        "position": "Python Developer",
        "target_grade": "Middle",
        "experience_years": 1,

        "phase": InterviewPhase.ASKING_QUESTION,

        "current_question": None,
        "current_answer": None,
        "current_instruction": None,
        "current_topic": "Python basics",
        "difficulty": 2,

        "dialog_history": [],
        "observer_notes": [],

        "confirmed_skills": [],
        "knowledge_gaps": {},
        "confidence_scores": [],

        "correct_answers": 0,
        "wrong_answers": 0,
        "partial_answers": 0,
        "off_topic_attempts": 0,

        "final_feedback": None,

        "last_observer_note": None,
        "last_analysis": None,
        "topics_covered": [],
        "questions_asked": 0,
        "max_questions": 10,

        "greeting_done": False,
        "current_question_difficulty": 2,
        "answered_turns": 0,
    }

    merged = dict(base)
    merged.update(state or {})

    if not isinstance(merged.get("dialog_history"), list):
        merged["dialog_history"] = []
    if not isinstance(merged.get("observer_notes"), list):
        merged["observer_notes"] = []
    if not isinstance(merged.get("confirmed_skills"), list):
        merged["confirmed_skills"] = []
    if not isinstance(merged.get("topics_covered"), list):
        merged["topics_covered"] = []
    if not isinstance(merged.get("knowledge_gaps"), dict):
        merged["knowledge_gaps"] = {}
    if not isinstance(merged.get("confidence_scores"), list):
        merged["confidence_scores"] = []

    if merged.get("phase") is None:
        merged["phase"] = InterviewPhase.ASKING_QUESTION

    if not isinstance(merged.get("difficulty"), int):
        merged["difficulty"] = 2
    merged["difficulty"] = max(1, min(5, merged["difficulty"]))

    if not isinstance(merged.get("max_questions"), int) or merged["max_questions"] <= 0:
        merged["max_questions"] = 10

    return merged

def build_graph(llm):
    interviewer = InterviewerAgent(llm)
    observer = ObserverAgent(llm)
    decision = DecisionAgent(llm)

    def greeting_node(state: InterviewState) -> InterviewState:
        state = _ensure_state_defaults(state)

        if state.get("greeting_done"):
            return state

        greeting = interviewer.handle_greeting(state)
        state["current_question"] = greeting
        state["questions_asked"] += 1
        state["greeting_done"] = True
        state["current_question_difficulty"] = state["difficulty"]
        state["phase"] = InterviewPhase.WAITING_FOR_ANSWER
        return state

    def interviewer_node(state: InterviewState) -> InterviewState:
        state = _ensure_state_defaults(state)

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

        asked_difficulty = state["difficulty"]
        question = interviewer.ask_question(state, instruction)

        state["current_question"] = question
        state["questions_asked"] += 1
        state["current_question_difficulty"] = asked_difficulty
        state["phase"] = InterviewPhase.WAITING_FOR_ANSWER

        return state

    def observer_node(state: InterviewState) -> InterviewState:
        state = _ensure_state_defaults(state)

        if state["phase"] != InterviewPhase.ANALYZING_ANSWER:
            return state

        answer = state["current_answer"] or ""
        analysis = observer.analyze_answer(state, answer)

        state["last_analysis"] = analysis
        state["answered_turns"] += 1

        thought = (
            f"[Observer Analysis] {analysis.get('internal_thought', '')}\n"
            f"Verdict: {analysis.get('verdict', '')}, Confidence: {analysis.get('confidence', '')}\n"
            f"Action: {analysis.get('next_action', '')}"
        )
        state["observer_notes"].append(thought)
        state["last_observer_note"] = analysis.get("note_to_interviewer")

        if not analysis.get("is_on_topic", True):
            state["off_topic_attempts"] += 1
            state["current_instruction"] = "return_to_topic"
            state["observer_notes"].append(
                f"[Off-topic detected] {analysis.get('off_topic_reason', '')}"
            )
        else:
            state["current_instruction"] = analysis.get("next_action", "continue")
            if analysis.get("next_topic_suggestion"):
                state["next_topic_suggestion"] = analysis["next_topic_suggestion"]

        if analysis.get("hallucination_detected", False):
            state["observer_notes"].append(
                f"[Hallucination] {analysis.get('hallucination_details', 'Detected false claims')}"
            )

        verdict = analysis.get("verdict", "partial")
        if verdict == "correct":
            state["correct_answers"] += 1
            if state["current_topic"] not in state["confirmed_skills"]:
                state["confirmed_skills"].append(state["current_topic"])
        elif verdict == "partial":
            state["partial_answers"] += 1
        elif verdict in {"wrong", "dont_know"}:
            state["wrong_answers"] += 1
            if analysis.get("knowledge_gap"):
                state["knowledge_gaps"][analysis["knowledge_gap"]] = analysis.get("expected_answer", "")

        asked_difficulty = state.get("current_question_difficulty", state["difficulty"])
        state["dialog_history"].append({
            "question": state["current_question"],
            "answer": answer,
            "topic": state["current_topic"],
            "difficulty": asked_difficulty
        })

        difficulty_change = analysis.get("difficulty_adjustment", 0)
        if not isinstance(difficulty_change, int):
            difficulty_change = 0
        state["difficulty"] = max(1, min(5, state["difficulty"] + difficulty_change))

        conf_map = {"low": 30, "medium": 60, "high": 90}
        state["confidence_scores"].append(conf_map.get(analysis.get("confidence", "medium"), 50))

        if state["current_topic"] not in state["topics_covered"]:
            state["topics_covered"].append(state["current_topic"])

        if state["answered_turns"] >= state["max_questions"]:
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
        state = _ensure_state_defaults(state)

        if state["phase"] != InterviewPhase.DECISION:
            return state

        feedback = decision.make_decision(state)
        state["final_feedback"] = feedback
        state["phase"] = InterviewPhase.FINISHED

        return state

    def router(state: InterviewState) -> str:
        state = _ensure_state_defaults(state)
        phase = state["phase"]

        if phase == InterviewPhase.WAITING_FOR_ANSWER:
            return END
        if phase == InterviewPhase.ANALYZING_ANSWER:
            return "observer"
        if phase == InterviewPhase.ASKING_QUESTION:
            return "interviewer"
        if phase == InterviewPhase.DECISION:
            return "decision"
        if phase == InterviewPhase.FINISHED:
            return END
        return END

    graph = StateGraph(dict)

    graph.add_node("greeting", greeting_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("observer", observer_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("greeting")

    graph.add_conditional_edges("greeting", router)
    graph.add_conditional_edges("interviewer", router)
    graph.add_conditional_edges("observer", router)
    graph.add_conditional_edges("decision", router)

    return graph.compile()
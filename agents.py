from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from schemas import InterviewState, CandidateProfile, TurnLog
from utils import call_mistral, parse_json_garbage
from prompts import (
    PROFILE_PARSER_SYSTEM, 
    MENTOR_SYSTEM, 
    INTERVIEWER_SYSTEM, 
    FEEDBACK_SYSTEM
)

def profile_parser_node(state: InterviewState):
    """Анализирует первое сообщение и создает профиль."""
    last_msg = state["messages"][-1].content
    state["last_user_input"] = last_msg
    
    messages = [
        SystemMessage(content=PROFILE_PARSER_SYSTEM),
        HumanMessage(content=last_msg)
    ]
    
    response_text = call_mistral(messages, json_mode=True)
    profile_dict = parse_json_garbage(response_text)
    
    profile = CandidateProfile(**profile_dict)
    
    initial_instruction = (
        f"Познакомься, подтверди, что понял контекст ({profile.role}, {profile.grade}). "
        f"Опыт кандидата: {profile.experience_summary}. "
        "Задай первый вводный технический вопрос по его опыту."
    )
    
    return {
        "profile": profile,
        "mentor_instruction": initial_instruction,
        "turn_count": 0,
        "difficulty": "Medium",
        "logs": [],
        "is_finished": False
    }


def mentor_node(state: InterviewState):
    
    last_user_msg = state["last_user_input"]
    profile = state["profile"]
    
    prompt = MENTOR_SYSTEM.format(
        role=profile.role,
        grade=profile.grade,
        experience=profile.experience_summary,
        difficulty=state.get("difficulty", "Medium")
    )
    
    # Собираем историю для контекста (без системных промптов предыдущих шагов, чтобы не путать)
    # Можно передавать summary, пока передаем последние N сообщений
    context_messages = [SystemMessage(content=prompt)]
    if len(state["messages"]) > 0:
        context_messages.extend(state["messages"][-4:]) 
    
    if not context_messages or not isinstance(context_messages[-1], HumanMessage):
         context_messages.append(HumanMessage(content=last_user_msg))

    response_text = call_mistral(context_messages, temperature=0.2, json_mode=True)
    mentor_data = parse_json_garbage(response_text)
    
    instruction = mentor_data.get("instruction", "Продолжай интервью.")
    internal_thoughts = mentor_data.get("analysis", "Анализ не предоставлен.")
    
    return {
        "mentor_instruction": instruction,
        "temp_thoughts": internal_thoughts 
    }

def interviewer_node(state: InterviewState):
    """Генерирует вопрос к кандидату."""
    profile = state["profile"]
    instruction = state["mentor_instruction"]
    
    prompt = INTERVIEWER_SYSTEM.format(
        role=profile.role,
        grade=profile.grade,
        instruction=instruction
    )
    
    messages = [SystemMessage(content=prompt)] + state["messages"]
    
    response_text = call_mistral(messages, temperature=0.7)
    
    current_turn = state["turn_count"] + 1
    
    internal_thoughts = state.get("temp_thoughts", "[Mentor silence]")
    
    new_log = TurnLog(
        turn_id=current_turn,
        agent_visible_message=response_text,
        user_message=state["last_user_input"],
        internal_thoughts=f"[Mentor Analysis]: {internal_thoughts} \n[Instruction]: {instruction}"
    )
    
    return {
        "messages": [AIMessage(content=response_text)],
        "turn_count": current_turn,
        "logs": [new_log]
    }

def feedback_node(state: InterviewState):
    """Генерирует финальный отчет."""
    messages = [SystemMessage(content=FEEDBACK_SYSTEM)] + state["messages"]
    
    response_text = call_mistral(messages, temperature=0.3, json_mode=True)
    feedback_data = parse_json_garbage(response_text)
    
    return {
        "final_feedback": feedback_data,
        "is_finished": True,
        "messages": [AIMessage(content="Спасибо за интервью! Результаты сохранены.")]
    }
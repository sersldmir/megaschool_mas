import json
from langchain_core.messages import HumanMessage, AIMessage

from config import LOG_FILE
from schemas import InterviewState, SessionLog
from graph import app

GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

def save_session(final_state: InterviewState):
    """Сохраняет логи и финальный отчет в JSON."""
    
    session_data = SessionLog(
        team_name="Mistral AI Agents",
        turns=final_state["logs"],
        final_feedback=final_state.get("final_feedback")
    )
    
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(json.loads(session_data.model_dump_json()), f, ensure_ascii=False, indent=2)
        print(f"\n{GREEN}Лог интервью сохранен в {LOG_FILE}{RESET}")
    except Exception as e:
        print(f"Ошибка сохранения лога: {e}")

def print_bot(text: str):
    print(f"\n{GREEN}[Interviewer]:{RESET} {text}")

def main():
    print(f"{GREEN}===  Техсобес с ИИ (Mistral) ==={RESET}")
    print("Команды: 'стоп' для завершения.")
    print("-----------------------------------------")
    
    print_bot("Доброго времени суток! Представьтесь, пожалуйста, и расскажите о своем опыте.")
    
    state = InterviewState(
        messages=[],
        profile=None,
        turn_count=0,
        logs=[],
        mentor_instruction="",
        last_user_input="",
        difficulty="Medium",
        is_finished=False,
        temp_thoughts=""
    )
    
    while True:

        try:
            user_input = input(f"\n{BLUE}[You]:{RESET} ").strip()
            if not user_input:
                continue
        except KeyboardInterrupt:
            print("\nПрерывание...")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state["last_user_input"] = user_input
        
        result = app.invoke(state)
        
        state = result
        
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print_bot(last_message.content)
            
        if state.get("is_finished", False):
            save_session(state)
            break

if __name__ == "__main__":
    main()
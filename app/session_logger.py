from typing import List, Dict, Any, Optional
from app.state import InterviewState

class SessionLogger:
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.turns: List[Dict[str, Any]] = []

    def log_turn(
        self,
        agent_message: str,
        user_message: str,
        internal_thoughts: str,
        internal: Optional[Dict[str, Any]] = None,
    ):
        turn_id = len(self.turns) + 1

        turn_data: Dict[str, Any] = {
            "turn_id": turn_id,
            "agent_visible_message": agent_message,
            "user_message": user_message,
            "internal_thoughts": internal_thoughts,
        }

        if internal is not None:
            turn_data["internal"] = internal

        self.turns.append(turn_data)

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        event_id = len(self.turns) + 1
        self.turns.append(
            {
                "turn_id": event_id,
                "event_type": event_type,
                "payload": payload,
            }
        )

    def export(self, state: InterviewState) -> Dict[str, Any]:
        return {
            "team_name": state["team_name"],
            "turns": self.turns,
            "final_feedback": state.get("final_feedback"),
        }
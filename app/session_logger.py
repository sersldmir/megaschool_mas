from app.state import InterviewState


class SessionLogger:
    def __init__(self, state: InterviewState):
        self.state = state
        self.turns = []

    def log_turn(
        self,
        agent_message: str,
        user_message: str,
        internal_thoughts: str,
    ):
        turn_id = len(self.turns) + 1

        self.turns.append({
            "turn_id": turn_id,
            "agent_visible_message": agent_message,
            "user_message": user_message,
            "internal_thoughts": internal_thoughts,
        })

    def export(self) -> dict:
        return {
            "team_name": self.state.team_name,
            "turns": self.turns,
            "final_feedback": self.state.final_feedback,
        }
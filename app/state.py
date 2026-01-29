from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class InterviewTurn:
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


@dataclass
class InterviewState:
    team_name: str
    position: str
    target_grade: str
    experience_years: int

    current_question: Optional[str] = None
    current_answer: Optional[str] = None
    last_analysis: Optional[dict] = None

    history: List[Dict[str, str]] = field(default_factory=list)
    turns: List[InterviewTurn] = field(default_factory=list)

    current_topic: str = "basics"
    difficulty: int = 2

    observer_notes: List[str] = field(default_factory=list)

    finished: bool = False
    final_feedback: Optional[str] = None
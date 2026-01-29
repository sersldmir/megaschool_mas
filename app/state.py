from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

class InterviewPhase(str, Enum):
    ASKING_QUESTION = "asking_question"
    WAITING_FOR_ANSWER = "waiting_for_answer"
    ANALYZING_ANSWER = "analyzing_answer"
    DECISION = "decision"
    FINISHED = "finished"

@dataclass
class InterviewState:

    team_name: str
    position: str
    target_grade: str
    experience_years: int

    phase: InterviewPhase = InterviewPhase.ASKING_QUESTION

    current_question: Optional[str] = None
    current_answer: Optional[str] = None

    dialog_history: List[Dict[str, str]] = field(default_factory=list)

    observer_notes: List[str] = field(default_factory=list)

    current_topic: str = "basics"
    difficulty: int = 2

    correct_answers: int = 0
    wrong_answers: int = 0

    final_feedback: Optional[str] = None
from enum import Enum
from typing import TypedDict, List, Dict, Optional, Any

class InterviewPhase(str, Enum):
    ASKING_QUESTION = "asking_question"
    WAITING_FOR_ANSWER = "waiting_for_answer"
    ANALYZING_ANSWER = "analyzing_answer"
    DECISION = "decision"
    FINISHED = "finished"

class DialogTurn(TypedDict):
    question: str
    answer: str
    topic: str
    difficulty: int

class InterviewState(TypedDict):
    team_name: str
    position: str
    target_grade: str
    experience_years: int

    phase: InterviewPhase

    current_question: Optional[str]
    current_answer: Optional[str]
    current_instruction: Optional[str]
    current_topic: str
    difficulty: int

    dialog_history: List[DialogTurn]
    observer_notes: List[str]

    confirmed_skills: List[str]
    knowledge_gaps: Dict[str, str]
    confidence_scores: List[int]

    correct_answers: int
    wrong_answers: int
    partial_answers: int
    off_topic_attempts: int

    final_feedback: Optional[Dict]

    last_observer_note: Optional[str]
    last_analysis: Optional[Dict[str, Any]]
    topics_covered: List[str]
    questions_asked: int
    max_questions: int

    greeting_done: bool
    current_question_difficulty: int

    answered_turns: int
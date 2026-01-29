from enum import Enum
class InterviewPhase(str, Enum):
    ASKING_QUESTION = "asking_question"
    WAITING_FOR_ANSWER = "waiting_for_answer"
    ANALYZING_ANSWER = "analyzing_answer"
    DECISION = "decision"
    FINISHED = "finished"

import operator
from typing import Annotated, List, TypedDict, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class TurnLog(BaseModel):
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str

class SessionLog(BaseModel):
    team_name: str = "Mistral Interview Agents"
    turns: List[TurnLog] = []
    final_feedback: Optional[dict] = None


class CandidateProfile(BaseModel):
    name: str = Field(description="Имя кандидата, если указано, иначе 'Кандидат'")
    role: str = Field(description="Целевая позиция (например, Backend Developer)")
    grade: str = Field(description="Грейд (Junior, Middle, Senior)")
    experience_summary: str = Field(
        description="Сводная информация об опыте, стеке технологий и достижениях одной строкой"
    )


class FinalFeedback(BaseModel):
    grade_assessment: str
    hiring_recommendation: str
    confidence_score: int
    confirmed_skills: List[str]
    knowledge_gaps: List[str]
    soft_skills_analysis: dict
    roadmap: List[str]


class InterviewState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    
    profile: Optional[CandidateProfile]
    turn_count: int

    logs: Annotated[List[TurnLog], operator.add]
    
    mentor_instruction: str  
    last_user_input: str     
    difficulty: str  
    
    is_finished: bool

    temp_thoughts: Optional[str]
    
    final_feedback: Optional[dict]
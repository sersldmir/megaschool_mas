from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class InterviewState:
    
    position: str
    target_grade: str
    experience_years: int

    history: List[Dict[str, str]] = field(default_factory=list)

    current_topic: str = "basics"
    difficulty: int = 2

    observer_notes: List[str] = field(default_factory=list)

    finished: bool = False
    final_feedback: Optional[Dict[str, Any]] = None
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Claim:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class Evidence:
    text: str
    source_id: str
    relevance_score: float = 0.0

@dataclass
class Argument:
    claim_id: str
    premises: List[str]

@dataclass
class DebateState:
    claim: Claim
    evidence_pool: List[Evidence] = field(default_factory=list)
    shared_evidence: List[Evidence] = field(default_factory=list)
    transcript: List[str] = field(default_factory=list)

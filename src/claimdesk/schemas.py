"""Typed messages between ClaimDesk agents. Agents talk only through these, never by free text."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Call(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


class Step(BaseModel):
    """One specialist reply: tool calls to run now, or the finished finding."""
    model_config = ConfigDict(extra="allow")     # a finding written at the top level is still read
    calls: List[Call] = []
    finding: Optional[Dict[str, Any]] = None


class Reason(BaseModel):
    text: str = ""
    cite: Any = None          # a rule id, doc id or claim id (a list is accepted too)


class IntakeReport(BaseModel):
    complete: bool
    missing_documents: List[str] = []
    notes: str = ""


class NotCovered(BaseModel):
    line: int
    rule: str = "R-NONMED"


class PolicyFinding(BaseModel):
    excluded_rule: Optional[Literal["R-EXCL-COSMETIC", "R-NETWORK-PREAUTH"]] = None
    wait_rule: Optional[Literal["R-WAIT-PED", "R-WAIT-MAT"]] = None
    covered_lines: List[int]
    not_covered: List[NotCovered] = []
    payable: Optional[float] = None
    reasons: List[Reason] = []


class CodingIssue(BaseModel):
    line: Optional[int] = None
    rule: str
    text: str = ""


class CodingFinding(BaseModel):
    invalid_lines: List[int] = []
    long_stay: bool = False
    issues: List[CodingIssue] = []


class FraudFinding(BaseModel):
    risk: Literal["low", "medium", "high"]
    duplicate_of: Optional[str] = None
    near_duplicate_of: Optional[str] = None
    inflated: bool = False
    watchlist: bool = False
    signals: List[Reason] = []


class AdjudicatorOut(BaseModel):
    decision: Literal["pay", "pend"]
    reason: str = ""
    cite: List[str] = []


class Letter(BaseModel):
    subject: str
    body: str


class Verdict(BaseModel):
    supported: bool
    reason: str = ""


class DirectDecision(BaseModel):
    """What the single-prompt and single-agent baselines answer with."""
    decision: Literal["approve", "part_pay", "reject", "pend"]
    payable: Optional[float] = None
    needs_human: bool = False
    reasons: List[Reason] = Field(default_factory=list)

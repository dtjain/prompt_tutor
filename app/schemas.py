from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Structured evaluation of a prompt."""

    specificity: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    goal_match: int = Field(ge=1, le=5)
    structure_style: int = Field(ge=1, le=5)
    advanced_techniques: int = Field(ge=1, le=5)
    strengths: List[str]
    issues: List[str]
    summary: str


class FeedbackResult(BaseModel):
    """Suggestions to improve a prompt."""

    suggestions: List[str]
    example_rewrite: str
    rationale: str


class PromptIteration(BaseModel):
    """One iteration in the tutoring loop."""

    prompt: str
    evaluation: EvaluationResult
    feedback: FeedbackResult



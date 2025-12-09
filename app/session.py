from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .schemas import FeedbackResult, PromptIteration


@dataclass
class PromptSession:
    """Track iterative improvements across revisions."""

    history: List[PromptIteration] = field(default_factory=list)

    def add(self, iteration: PromptIteration) -> None:
        self.history.append(iteration)

    def last(self) -> PromptIteration | None:
        return self.history[-1] if self.history else None

    def as_dict(self) -> List[dict]:
        return [item.model_dump() for item in self.history]

    def summary(self) -> str:
        if not self.history:
            return "No iterations yet."
        lines = []
        for idx, item in enumerate(self.history, start=1):
            avg = (
                item.evaluation.specificity
                + item.evaluation.clarity
                + item.evaluation.goal_match
                + item.evaluation.structure_style
                + item.evaluation.advanced_techniques
            ) / 5.0
            lines.append(f"#{idx} avg={avg:.1f} prompt='{item.prompt[:60]}...'")
        return "\n".join(lines)

    def trajectory(self) -> List[int]:
        """Return average scores per iteration for plotting."""
        return [
            int(
                (
                    it.evaluation.specificity
                    + it.evaluation.clarity
                    + it.evaluation.goal_match
                    + it.evaluation.structure_style
                    + it.evaluation.advanced_techniques
                )
                / 5
            )
            for it in self.history
        ]



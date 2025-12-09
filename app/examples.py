from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PromptExample:
    title: str
    original: str
    feedback: str
    improved: str


def get_examples() -> List[PromptExample]:
    """Curated examples for teaching-by-example."""
    return [
        PromptExample(
            title="Summarization",
            original="Summarize the attached article.",
            feedback="Too vague: lacks length, audience, and format guidance.",
            improved=(
                "You are a helpful analyst. Summarize the attached article in 5 bullet "
                "points for busy executives. Emphasize the key findings and "
                "implications, keep each bullet under 20 words."
            ),
        ),
        PromptExample(
            title="Creative writing",
            original="Write a story about space.",
            feedback="Missing tone, length, constraints, and perspective.",
            improved=(
                "You are a sci-fi author. Write a 250-word, first-person story about "
                "an engineer stranded on Mars. Use a hopeful tone and end with a "
                "surprising discovery."
            ),
        ),
        PromptExample(
            title="Code generation",
            original="Make a Python script that sorts numbers.",
            feedback="No input/output contract or constraints; lacks explanation.",
            improved=(
                "You are a senior Python dev. Write a Python 3.10 script that reads a "
                "list of integers from STDIN, outputs the list sorted ascending, and "
                "includes a brief docstring and inline comments explaining the steps."
            ),
        ),
    ]



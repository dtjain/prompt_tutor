from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from .schemas import EvaluationResult
from .settings import settings

EVAL_SYSTEM_PROMPT = """
You are a Prompt Tutor. Evaluate the user's prompt for quality.
Score each dimension from 1 (poor) to 5 (excellent).
Be concise and professional.
Return only a JSON object matching the specified schema.
"""

EVAL_INSTRUCTIONS = """
Required JSON fields:
- specificity: integer 1-5
- clarity: integer 1-5
- goal_match: integer 1-5
- structure_style: integer 1-5
- advanced_techniques: integer 1-5
- strengths: list of short bullet strings
- issues: list of short bullet strings
- summary: one-sentence overview
"""


def build_evaluation_chain(
    model: Optional[str] = None, temperature: Optional[float] = None
) -> RunnableSerializable:
    """Create a structured output chain for evaluating prompts."""
    llm = ChatOpenAI(
        model=model or settings.model,
        temperature=temperature if temperature is not None else settings.temperature,
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVAL_SYSTEM_PROMPT + EVAL_INSTRUCTIONS),
            (
                "human",
                "User prompt:\n{raw_prompt}\n\nReturn JSON only.",
            ),
        ]
    )

    return prompt | llm.with_structured_output(EvaluationResult)


def evaluate_prompt(raw_prompt: str) -> EvaluationResult:
    """Evaluate a raw prompt and return structured scores + notes."""
    chain = build_evaluation_chain()
    return chain.invoke({"raw_prompt": raw_prompt})



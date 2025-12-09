from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI

from .schemas import EvaluationResult, FeedbackResult
from .settings import settings

FEEDBACK_SYSTEM_PROMPT = """
You are a prompt improvement coach. Suggest 1–2 concrete improvements that use:
- role prompting or clear persona
- chain-of-thought or step-by-step scaffolding
- few-shot / exemplar inclusion when helpful
- style directives (tone, brevity) if relevant
Keep it actionable and concise.
"""


def build_feedback_chain(
    model: Optional[str] = None, temperature: Optional[float] = None
) -> RunnableSerializable:
    """Create a feedback chain that proposes improvements and an example rewrite."""
    llm = ChatOpenAI(
        model=model or settings.model,
        temperature=temperature if temperature is not None else settings.temperature,
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FEEDBACK_SYSTEM_PROMPT),
            (
                "human",
                "User prompt:\n{raw_prompt}\n\n"
                "Evaluation JSON:\n{evaluation_json}\n\n"
                "Return JSON with keys: suggestions (list of 1-2 strings), "
                "example_rewrite (single improved prompt), rationale (1-2 sentences).",
            ),
        ]
    )

    return prompt | llm.with_structured_output(FeedbackResult)


def suggest_improvements(raw_prompt: str, evaluation: EvaluationResult) -> FeedbackResult:
    """Suggest improvements based on the evaluation."""
    chain = build_feedback_chain()
    return chain.invoke(
        {"raw_prompt": raw_prompt, "evaluation_json": evaluation.model_dump()}
    )



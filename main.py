from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.evaluator import evaluate_prompt
from app.examples import get_examples
from app.feedback_agent import suggest_improvements
from app.schemas import PromptIteration
from app.session import PromptSession
from app.settings import settings

app = typer.Typer(add_completion=False, help="Prompt Tutor CLI")
console = Console()


def _print_evaluation(result) -> None:
    table = Table(title="Prompt Evaluation", show_edge=False, header_style="bold")
    table.add_column("Dimension")
    table.add_column("Score")
    scores = {
        "Specificity": result.specificity,
        "Clarity": result.clarity,
        "Goal Match": result.goal_match,
        "Structure & Style": result.structure_style,
        "Advanced Techniques": result.advanced_techniques,
    }
    for name, value in scores.items():
        table.add_row(name, str(value))
    console.print(table)
    console.print("[bold]Strengths:[/bold] " + "; ".join(result.strengths))
    console.print("[bold]Issues:[/bold] " + "; ".join(result.issues))
    console.print("[bold]Summary:[/bold] " + result.summary)


def _print_feedback(feedback) -> None:
    console.print("\n[bold]Suggestions[/bold]")
    for idx, item in enumerate(feedback.suggestions, start=1):
        console.print(f"{idx}. {item}")
    console.print("\n[bold]Example Rewrite[/bold]")
    console.print(feedback.example_rewrite)
    console.print("\n[bold]Rationale[/bold]")
    console.print(feedback.rationale)


@app.command()
def evaluate(prompt: str) -> None:
    """Evaluate a prompt and return scores + feedback."""
    settings.validate()
    result = evaluate_prompt(prompt)
    _print_evaluation(result)


@app.command()
def feedback(prompt: str) -> None:
    """Evaluate a prompt and suggest improvements."""
    settings.validate()
    evaluation = evaluate_prompt(prompt)
    _print_evaluation(evaluation)
    improvement = suggest_improvements(prompt, evaluation)
    _print_feedback(improvement)


@app.command()
def examples() -> None:
    """Show curated prompts and improved versions."""
    for item in get_examples():
        console.rule(f"[bold blue]{item.title}")
        console.print(f"[bold]Original:[/bold] {item.original}")
        console.print(f"[bold]Feedback:[/bold] {item.feedback}")
        console.print(f"[bold]Improved:[/bold] {item.improved}\n")


@app.command()
def loop(model: Optional[str] = typer.Option(None), temperature: Optional[float] = None) -> None:
    """Interactive iterative loop: prompt -> evaluation -> feedback -> revise."""
    settings.validate()
    session = PromptSession()
    console.print("[bold green]Prompt Tutor iterative loop[/bold green]")
    console.print("Enter an empty prompt to exit.\n")

    while True:
        raw_prompt = typer.prompt("Your prompt (blank to quit)")
        if not raw_prompt.strip():
            break
        eval_result = evaluate_prompt(raw_prompt)
        feedback_result = suggest_improvements(raw_prompt, eval_result)
        iteration = PromptIteration(
            prompt=raw_prompt, evaluation=eval_result, feedback=feedback_result
        )
        session.add(iteration)

        console.rule(f"Iteration {len(session.history)}")
        _print_evaluation(eval_result)
        _print_feedback(feedback_result)
        console.print("\n[bold]Trajectory:[/bold] " + ", ".join(map(str, session.trajectory())))

    console.print("\n[bold]Session summary[/bold]")
    console.print(session.summary())
    console.print("\nRaw JSON trajectory:")
    console.print_json(json.dumps(session.as_dict(), indent=2))


if __name__ == "__main__":
    app()



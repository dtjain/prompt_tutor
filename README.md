# Prompt Tutor — Learn Prompt Engineering with Feedback

Interactive tutor that scores prompts, suggests improvements, and shows curated examples. Built with LangChain + OpenAI GPT-4o. Run locally via a simple CLI; extend with Gradio/Streamlit when ready.

## Quickstart

1) Python 3.10+ recommended.  
2) Install deps:
```
pip install -r requirements.txt
```
3) Set env:
```
export OPENAI_API_KEY=sk-...
```
4) Try it:
```
python main.py evaluate "Write me a poem about the sea."
python main.py feedback "Write me a poem about the sea."
python main.py loop
```

## Streamlit Web App

Run a local web UI:
```
streamlit run streamlit_app.py
```
Features:
- Enter your OpenAI API key and model in the sidebar.
- Evaluate a prompt, see scores + improvement suggestions.
- Review teaching examples and your session history with averages charted.
- Exportable JSON trajectory at the bottom.

## Features
- Structured evaluation: scores 1–5 for specificity, clarity, goal_match, structure_style, advanced_techniques plus qualitative feedback.
- Feedback agent: 1–2 improvements using role prompting, chain-of-thought, few-shot, or style cues.
- Teaching by example: curated prompts with feedback and improved versions.
- Iterative loop: store each revision, reevaluate, and view trajectory.

## Extending
- Frontend: wrap `PromptSession` and the chains in Gradio/Streamlit (see `main.py` placeholders).
- Models: swap `model` in `settings.py` or pass `--model` to CLI commands.
- Storage: plug in a DB or filesystem persistence inside `PromptSession`.



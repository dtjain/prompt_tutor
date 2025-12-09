from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.examples import get_examples
from app.settings import settings


st.set_page_config(page_title="Prompt Tutor · Academic Workspace", page_icon="🧠", layout="wide")

# Theme state
def render_theme() -> None:
    palette = {
        "navy": "#1C2D5A",
        "ivory": "#F7F7F2" if st.session_state.light_mode else "#0F172A",
        "gold": "#C7A249",
        "ink": "#0F172A" if st.session_state.light_mode else "#E5E7EB",
        "card_bg": "#FFFFFF" if st.session_state.light_mode else "#111827",
        "border": "rgba(28,45,90,0.12)" if st.session_state.light_mode else "rgba(229,231,235,0.16)",
    }
    st.markdown(
        f"""
        <style>
        :root {{
            --navy: {palette['navy']};
            --ivory: {palette['ivory']};
            --gold: {palette['gold']};
            --ink: {palette['ink']};
        }}
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@600;700&family=Lato:wght@400;500&display=swap');
        html, body, [class*="css"]  {{
            font-family: 'Lato', sans-serif;
            background-color: var(--ivory);
            color: var(--ink);
        }}
        h1, h2, h3, h4 {{
            font-family: 'Merriweather', Georgia, serif;
            color: var(--ink);
        }}
        .prompt-card {{
            border: 1px solid {palette['border']};
            background: {palette['card_bg']};
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        }}
        .bad {{ background: #fff1f0; border-left: 4px solid #d32f2f; }}
        .mid {{ background: #fff8e1; border-left: 4px solid #f9a825; }}
        .good {{ background: #f1f8e9; border-left: 4px solid #388e3c; }}
        .sidebar-content {{ position: sticky; top: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if "light_mode" not in st.session_state:
    st.session_state.light_mode = True
render_theme()

st.title("Prompt Tutor")
st.caption("Master Prompt Engineering with Guided Feedback.")

# --- Constants and sample content ---
TASK_TYPES = ["Summarization", "Reasoning", "Brainstorming", "Creative Writing", "Q&A"]

RUBRIC = [
    ("Task", "Clear action verb + specific objective (e.g., 'Summarize the article')"),
    ("Context", "Relevant background, data, or purpose"),
    ("Role", "Explicit persona or role for the LLM (e.g., 'You are a tutor')"),
    ("Format", "Desired output format (bullets, table, JSON, email, etc.)"),
    ("Tone", "Stylistic cues like formal, casual, persuasive"),
    ("Constraints", "Rules like length, must-include points, or banned phrases"),
    ("Examples", "Few-shot prompt-response pairs to guide behavior"),
]

TEMPLATES = {
    "Writing": [
        {
            "title": "Concise blog outline",
            "prompt": (
                "You are a content strategist. Create a blog outline on {topic} with 5 sections, "
                "each with 2 bullet sub-points. Tone: clear and helpful. Keep total under 180 words."
            ),
            "why": "Defines role, format, length, and tone.",
        },
        {
            "title": "Rewrite for clarity",
            "prompt": (
                "You are an editor. Rewrite the following paragraph for clarity and brevity (under 120 words). "
                "Return before/after in a two-column markdown table. Text: {text}"
            ),
            "why": "Adds role, format, length limit, and structure.",
        },
    ],
    "Coding": [
        {
            "title": "Docstring generator",
            "prompt": (
                "You are a senior Python dev. Write a Google-style docstring for the function below. "
                "Include Args, Returns, Raises, and an example. Code:\n{code}"
            ),
            "why": "Role + structured format + examples.",
        },
        {
            "title": "Explain code",
            "prompt": (
                "You are a teacher. Explain what this code does to a beginner in 5 bullets, each under 18 words. "
                "Code:\n{code}"
            ),
            "why": "Role, audience, format, and length constraints.",
        },
    ],
    "Business": [
        {
            "title": "Product description",
            "prompt": (
                "You are a product marketer. Write a 120-word description for {product}, highlighting 3 key benefits "
                "and 1 differentiator. Tone: persuasive but concise. End with a one-line CTA."
            ),
            "why": "Role, length, format expectations, tone, and constraints.",
        },
        {
            "title": "Executive summary",
            "prompt": (
                "You are an analyst. Summarize this report for executives in 5 bullets (<18 words each), "
                "covering impact, risks, and next steps. Content:\n{content}"
            ),
            "why": "Audience, format, constraints, and focus areas.",
        },
    ],
    "Research": [
        {
            "title": "Paper abstract",
            "prompt": (
                "You are a scientific writer. Draft a 150-word abstract summarizing problem, method, results, and implications. "
                "Use accessible language and avoid jargon."
            ),
            "why": "Role, length, structure, tone constraints.",
        },
        {
            "title": "Compare studies",
            "prompt": (
                "You are a research analyst. Compare Study A vs Study B in a 2-column markdown table with rows: "
                "goal, method, dataset, metrics, key findings, limitations."
            ),
            "why": "Explicit format and structure.",
        },
    ],
    "Education": [
        {
            "title": "Lesson bullets",
            "prompt": (
                "You are a teacher. Produce 5 bullet learning objectives for a lesson on {topic}, each under 12 words, "
                "focusing on outcomes and avoiding jargon."
            ),
            "why": "Role, format, length constraints.",
        },
        {
            "title": "Explain like I'm 12",
            "prompt": (
                "You are a friendly tutor. Explain {concept} to a 12-year-old in 120 words with 1 simple example."
            ),
            "why": "Role, audience, tone, length, and example.",
        },
    ],
    "Misc": [
        {
            "title": "Brainstorm ideas",
            "prompt": (
                "You are a creative strategist. Brainstorm 8 ideas for {goal} as short bullet headlines (<10 words each). "
                "Avoid generic clichés."
            ),
            "why": "Role, format, constraints.",
        },
        {
            "title": "Pros/cons table",
            "prompt": (
                "You are an analyst. Create a markdown table of pros and cons for {topic} with 4 rows each. "
                "Keep entries under 12 words."
            ),
            "why": "Role, format, and length constraints.",
        },
    ],
}

# --- Helpers ---
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if "last_run" not in st.session_state:
    st.session_state.last_run = {"prompt": "", "task": "", "output": ""}
if "builder_prefill" not in st.session_state:
    st.session_state.builder_prefill = ""
if "improved_prompt" not in st.session_state:
    st.session_state.improved_prompt = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "template_embeddings" not in st.session_state:
    st.session_state.template_embeddings = None
if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None
if "rag_meta" not in st.session_state:
    st.session_state.rag_meta = None


def evaluate_rubric(prompt: str) -> List[Dict[str, Any]]:
    """Heuristic 0/1 scoring for the 7 rubric elements."""
    text = prompt.lower()
    task_keywords = [
        "summarize",
        "write",
        "generate",
        "analyze",
        "explain",
        "create",
        "draft",
        "classify",
        "task",
        "objective",
        "goal",
        "instruction",
        "request",
        "ask",
        "need",
        "purpose",
    ]
    checks = {
        "Task": any(v in text for v in task_keywords),
        "Context": any(v in text for v in ["context", "about", "based on", "given", "regarding", "source", "data"]),
        "Role": "you are" in text or "act as" in text,
        "Format": any(v in text for v in ["bullet", "table", "json", "list", "markdown", "outline", "sections"]),
        "Tone": any(v in text for v in ["tone", "style", "formal", "casual", "concise", "friendly", "academic", "persuasive"]),
        "Constraints": any(v in text for v in ["under", "no more than", "limit", "words", "must", "at least", "avoid"]),
        "Examples": any(v in text for v in ["example", "few-shot", "q:", "a:", "demo", "sample"]),
    }
    feedback_map = {
        "Task": "Add a clear task/objective.",
        "Context": "Add the context or source the model should use.",
        "Role": "Specify a role/persona (e.g., 'You are a ...').",
        "Format": "State the output format: bullets, table, JSON, sections.",
        "Tone": "Give a tone/style cue: concise, formal, friendly, academic.",
        "Constraints": "Add constraints: length limits, must/avoid points, structure.",
        "Examples": "Provide 1–3 short examples to guide style/output.",
    }
    results = []
    for name, _ in RUBRIC:
        present = checks[name]
        results.append(
            {
                "Element": name,
                "Present": present,
                "Feedback": "Present" if present else feedback_map[name],
            }
        )
    return results


def rubric_score(results: List[Dict[str, Any]]) -> int:
    return sum(1 for r in results if r["Present"])


def build_prompt(role: str, task: str, context: str, fmt: str, tone: str, constraints: str, examples: str) -> str:
    parts = []
    if role:
        parts.append(f"You are a {role}.")
    if task:
        parts.append(f"Your task is to {task}.")
    if context:
        parts.append(f"Context: {context}.")
    if fmt:
        parts.append(f"Please respond in {fmt}.")
    if tone:
        parts.append(f"Use a {tone} tone.")
    if constraints:
        parts.append(constraints)
    if examples:
        parts.append(f"Examples:\n{examples}")
    return " ".join(parts).strip()


def improvement_suggestions(results: List[Dict[str, Any]]) -> List[str]:
    suggestions = []
    for r in results:
        if not r["Present"]:
            if r["Element"] == "Role":
                suggestions.append("Add a role: 'You are a ...' to set persona and voice.")
            elif r["Element"] == "Task":
                suggestions.append("Start with a clear verb/objective (e.g., 'Summarize the report in 5 bullets').")
            elif r["Element"] == "Context":
                suggestions.append("Add context/source or audience so the model knows what to use.")
            elif r["Element"] == "Format":
                suggestions.append("Specify output format: bullets, table, JSON, or sections.")
            elif r["Element"] == "Tone":
                suggestions.append("State tone/style: concise, formal, friendly, persuasive, etc.")
            elif r["Element"] == "Constraints":
                suggestions.append("Add constraints: length limits, must/avoid points, structure.")
            elif r["Element"] == "Examples":
                suggestions.append("Include 1–3 short examples to illustrate desired style/output.")
    return suggestions


def apply_suggestions(prompt: str, accepted: List[str]) -> str:
    if not accepted:
        return prompt
    additions = "\n- " + "\n- ".join(accepted)
    return f"{prompt}\n\nAdd the following refinements:\n{additions}"


def call_openai(system_prompt: str, user_input: str) -> str:
    llm = ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )
    resp = llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_input),
        ]
    )
    return resp.content.strip()


def _embed_templates():
    """Create embeddings for template prompts to support semantic retrieval."""
    if st.session_state.template_embeddings is not None:
        return
    if not settings.openai_api_key:
        return
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    corpus = []
    meta = []
    for cat, items in TEMPLATES.items():
        for tpl in items:
            corpus.append(tpl["prompt"])
            meta.append({"category": cat, "title": tpl["title"], "prompt": tpl["prompt"], "why": tpl["why"]})
    vectors = embedder.embed_documents(corpus)
    st.session_state.template_embeddings = list(zip(vectors, meta))


def _semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k template matches by cosine similarity."""
    if not query.strip() or st.session_state.template_embeddings is None:
        return []
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    q_vec = embedder.embed_query(query)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)

    scored = [
        (cosine(q_vec, vec), meta)
        for vec, meta in st.session_state.template_embeddings
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


# --- Sidebar setup ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("Prompt Tutor")
    st.caption("Academic Workspace · Guided Feedback")
    default_key = st.session_state.api_key or os.getenv("OPENAI_API_KEY", "")
    # If a key exists at load time (env or prior session), sync it immediately
    if default_key and not st.session_state.api_key:
        st.session_state.api_key = default_key
        os.environ["OPENAI_API_KEY"] = default_key
        settings.openai_api_key = default_key

    api_key = st.text_input(
        "OpenAI API Key", type="password", value=default_key, help="Stored only in memory."
    )
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0 if settings.model == "gpt-4o" else 1)
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=settings.temperature, step=0.05
    )
    st.toggle("Light mode", key="light_mode")

    # Auto-sync key as soon as it is present (no second prompt later)
    if api_key.strip():
        st.session_state.api_key = api_key.strip()
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        settings.openai_api_key = st.session_state.api_key

    if st.button("Save settings"):
        if not st.session_state.api_key:
            st.error("API key is required.")
        else:
            settings.model = model.strip() or settings.model
            settings.temperature = temperature
            st.success("Settings saved.")

    st.markdown("</div>", unsafe_allow_html=True)

render_theme()

# --- Tabs ---
tab_builder, tab_evaluator, tab_improve, tab_templates, tab_agents, tab_rag = st.tabs(
    [
        "Prompt Builder",
        "Prompt Evaluator",
        "Improve My Prompt",
        "Prompt Examples & Templates",
        "Multi-Agent Refiner",
        "Ask My Docs",
    ]
)

# Tab 1: Prompt Builder
with tab_builder:
    st.subheader("Build a strong prompt (RGC + format)")
    with st.expander("📘 What makes a good prompt?"):
        st.markdown(
            "- Role: set the persona.\n"
            "- Task: start with a clear verb.\n"
            "- Context: add audience, data, or goal.\n"
            "- Format: bullets, table, JSON, sections.\n"
            "- Tone: concise, formal, friendly, academic.\n"
            "- Constraints: length, must/avoid points.\n"
            "- Examples: 1–3 short exemplars."
        )

    with st.form("builder-form"):
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Role / Persona", placeholder="e.g., product marketer, tutor")
            task = st.text_input("Task / Instruction", placeholder="e.g., summarize the report in 5 bullets")
            context = st.text_area("Context", placeholder="e.g., audience, source, goal", height=100)
            tone = st.text_input("Tone / Style", placeholder="e.g., concise, formal, friendly")
        with col2:
            fmt = st.text_input("Format", placeholder="e.g., bullets, table, JSON, sections")
            constraints = st.text_area(
                "Constraints", placeholder="e.g., under 150 words; must mention risks and next steps", height=100
            )
            examples = st.text_area("Examples (optional)", placeholder="Few-shot pairs or snippets", height=100)
        submitted = st.form_submit_button("Assemble prompt", type="primary")

    assembled_prompt = build_prompt(role, task, context, fmt, tone, constraints, examples)
    if st.session_state.builder_prefill:
        assembled_prompt = st.session_state.builder_prefill
    st.markdown("### Preview")
    st.code(assembled_prompt or "Your assembled prompt will appear here.", language="text")
    st.download_button(
        "Download Prompt (.txt)",
        assembled_prompt,
        file_name="prompt_tutor_prompt.txt",
        mime="text/plain",
        disabled=not assembled_prompt,
    )
    st.caption("Want to learn by example? Try one of our sample prompts in the Examples tab.")

# Tab 2: Prompt Evaluator
with tab_evaluator:
    st.subheader("Evaluate your prompt against the 7-point rubric")
    with st.expander("💡 How the rubric works"):
        st.markdown(
            "We score presence of Task, Context, Role, Format, Tone, Constraints, and Examples (0/1 each). "
            "Use this to spot missing elements quickly."
        )
    eval_prompt = st.text_area("Paste your prompt", height=200)
    if st.button("Run Rubric", type="primary"):
        results = evaluate_rubric(eval_prompt)
        score = rubric_score(results)
        st.metric("Total Score", f"{score}/7")
        st.divider()
        for r in results:
            icon = "✅" if r["Present"] else "❌"
            st.markdown(f"{icon} **{r['Element']}** — {r['Feedback']}")
        with st.expander("Full feedback (JSON)"):
            st.json(results)

# Tab 3: Improve My Prompt
with tab_improve:
    st.subheader("🧠 Improve My Prompt")
    st.caption("Paste any prompt and get a cleaner, structured rewrite using the 7-element framework.")
    with st.expander("💡 Tip: What’s a good constraint?"):
        st.markdown("Think length limits, required points, banned phrases, or required structure (bullets, table).")
    base_prompt = st.text_area("Original Prompt", height=200, value=st.session_state.last_run.get("prompt", ""))
    show_eval = st.checkbox("🔍 Show Evaluation Feedback", value=False)

    if st.button("✨ Improve Prompt", type="primary"):
        try:
            # Sync settings with session key so we don't prompt twice
            if st.session_state.api_key:
                settings.openai_api_key = st.session_state.api_key
                os.environ["OPENAI_API_KEY"] = st.session_state.api_key

            if not settings.openai_api_key:
                st.error("OPENAI_API_KEY is required. Set it in the sidebar.")
            else:
                with st.spinner("Improving your prompt..."):
                    llm = ChatOpenAI(
                        model=settings.model,
                        temperature=settings.temperature,
                        api_key=settings.openai_api_key,
                    )
                    system = (
                        "You are a Prompt Optimization Assistant. Rewrite the given prompt to include:\n"
                        "- Clear Task/Instruction\n- Relevant Context\n- Specified Role/Persona\n"
                        "- Desired Output Format\n- Tone/Style\n- Constraints\n- Examples if appropriate\n\n"
                        "Make it structured, specific, concise, and effective for LLMs. Return ONLY the improved prompt."
                    )
                    user = f"ORIGINAL PROMPT:\n{base_prompt}\n\nIMPROVED PROMPT:"
                    resp = llm.invoke([("system", system), ("human", user)])
                    st.session_state.improved_prompt = resp.content.strip()
        except Exception as exc:  # pragma: no cover
            st.error(f"Error: {exc}")

    if st.session_state.improved_prompt:
        st.markdown("🔹 **Before**")
        st.code(base_prompt, language="text")
        st.markdown("🔹 **After**")
        st.code(st.session_state.improved_prompt, language="text")
        st.download_button(
            "📋 Download Improved Prompt (.txt)",
            st.session_state.improved_prompt,
            file_name="prompt_tutor_improved.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if show_eval and base_prompt:
        st.divider()
        st.markdown("### Rubric check (original prompt)")
        results = evaluate_rubric(base_prompt)
        score = rubric_score(results)
        st.metric("Score", f"{score}/7")
        for r in results:
            icon = "✅" if r["Present"] else "❌"
            st.markdown(f"{icon} **{r['Element']}** — {r['Feedback']}")

    with st.expander("🗣️ Conversational coach"):
        st.caption("Ask follow-ups about improving this prompt.")
        if prompt := st.chat_input("Ask for guidance"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            if st.session_state.api_key:
                settings.openai_api_key = st.session_state.api_key
            if settings.openai_api_key:
                coach = ChatOpenAI(
                    model=settings.model,
                    temperature=settings.temperature,
                    api_key=settings.openai_api_key,
                )
                context_msgs = []
                if base_prompt:
                    context_msgs.append(("system", f"Original prompt: {base_prompt}"))
                if st.session_state.improved_prompt:
                    context_msgs.append(("system", f"Improved prompt: {st.session_state.improved_prompt}"))
                resp = coach.invoke(
                    [
                        ("system", "You are a prompt coaching assistant. Be concise and specific."),
                        *context_msgs,
                        ("human", prompt),
                    ]
                )
                st.session_state.chat_history.append({"role": "assistant", "content": resp.content})
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

# Tab 4: Prompt Examples & Templates
with tab_templates:
    st.subheader("Curated templates by domain")
    st.caption("Want to learn by example? Try one of our sample prompts.")
    category = st.selectbox("Category", list(TEMPLATES.keys()))
    query = st.text_input("Search templates (semantic)", placeholder="e.g., summarize risks; persuasive email")
    if settings.openai_api_key:
        _embed_templates()
    matches = _semantic_search(query) if query else []
    if matches:
        st.markdown("**Relevant templates**")
        for m in matches:
            st.markdown(f"- [{m['category']}] {m['title']}")
    for tpl in TEMPLATES[category]:
        st.markdown(f"#### {tpl['title']}")
        st.code(tpl["prompt"], language="text")
        st.caption(tpl["why"])
        if st.button(f"Use this prompt: {tpl['title']}", key=tpl["title"]):
            st.session_state.builder_prefill = tpl["prompt"]
            st.success("Sent to Prompt Builder tab. Open it to customize.")
        st.divider()

with tab_agents:
    st.subheader("Multi-Agent Prompt Refinement")
    orig = st.text_area("Enter your prompt", height=160, key="agent_original")
    if st.button("Run Multi-Agent Flow", type="primary"):
        if not settings.openai_api_key:
            st.error("OPENAI_API_KEY is required.")
        else:
            with st.spinner("Rewriting..."):
                rewritten = call_openai(
                    "You are a prompt engineer. Rewrite this prompt to be more clear, specific, and effective.",
                    orig,
                )
                st.subheader("📝 Rewritten Prompt")
                st.code(rewritten, language="text")

            with st.spinner("Critiquing..."):
                critique = call_openai(
                    "You are a critical reviewer. Evaluate the quality of the prompt below and offer improvements.",
                    rewritten,
                )
                st.subheader("👀 Critique")
                st.code(critique, language="text")

            with st.spinner("Finalizing..."):
                final = call_openai(
                    "You are a prompt finalizer. Improve the rewritten prompt below using the critique.",
                    f"Prompt: {rewritten}\nCritique: {critique}",
                )
                st.subheader("✅ Final Prompt")
                st.code(final, language="text")
                st.download_button(
                    "📋 Download Final Prompt (.txt)",
                    final,
                    file_name="final_prompt.txt",
                    mime="text/plain",
                )

with tab_rag:
    st.subheader("Ask My Docs (PDF RAG)")
    st.caption("Upload a PDF and ask questions grounded in the document.")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and settings.openai_api_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.rag_retriever = vectorstore.as_retriever()
        st.session_state.rag_meta = {"pages": len(pages), "chunks": len(chunks)}
        st.success(f"Loaded {len(pages)} pages → {len(chunks)} chunks.")

    question = st.text_input("Ask a question about your PDF")
    if st.button("Answer", key="rag-answer", type="primary"):
        if not settings.openai_api_key:
            st.error("OPENAI_API_KEY is required.")
        elif not st.session_state.rag_retriever:
            st.error("Upload a PDF first.")
        else:
            docs = st.session_state.rag_retriever.get_relevant_documents(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            with st.spinner("Answering from your document..."):
                answer = call_openai(
                    "You are a helpful assistant. Use ONLY the provided context to answer.",
                    f"Context:\n{context}\n\nQuestion: {question}",
                )
            st.markdown("**Answer**")
            st.write(answer)
            if docs:
                st.markdown("**Context used (top snippets)**")
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"*Snippet {i}:* {doc.page_content[:400]}{'...' if len(doc.page_content)>400 else ''}")


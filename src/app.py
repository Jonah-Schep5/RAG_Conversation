"""
app.py  –  Telecom RAG Chatbot (Streamlit)
==========================================
Workflow:
  1. User pastes a partial/in-progress call transcript.
  2. The app retrieves the top-k most similar past conversations from FAISS.
  3. A prompt is built with the retrieved context + the pasted transcript.
  4. gemma3:12b (via LM Studio OpenAI-compatible API) predicts:
       • The likely next event in the conversation
       • The customer's current sentiment
       • The best agent script line to improve sentiment
"""

import textwrap
from typing import Any

import requests
import streamlit as st

from rag_retriever import RAGRetriever

# ─── Configuration ────────────────────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_MODEL = "google/gemma-3-12b"   # adjust to exact model name in LM Studio
TOP_K = 5
MAX_CONTEXT_CHARS = 2000  # truncate each retrieved transcript in the prompt


# ─── LM Studio call ───────────────────────────────────────────────────────────
def call_lm_studio(messages: list[dict], temperature: float = 0.3) -> str:
    """Send a chat-completion request to LM Studio and return the reply text."""
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "**Error:** Could not connect to LM Studio. "
            "Make sure LM Studio is running on localhost:1234 with gemma3:12b loaded."
        )
    except Exception as exc:
        return f"**Error:** {exc}"


# ─── Prompt builder ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert telecom call center coach and conversation analyst.
    You will be given:
      - A set of SIMILAR PAST CONVERSATIONS retrieved from a knowledge base.
      - A CURRENT CONVERSATION that is in progress or just ended.

    Respond using EXACTLY this format (keep the headers verbatim):

    NEXT ACTION: <one sentence — the single most important thing the agent should say or do right now>

    **1. Next Event Prediction**
    <What is the most likely next event in this call? e.g., verification, hold, transfer, resolution, callback>

    **2. Customer Sentiment**
    <Positive / Neutral / Negative / Escalating — and a 1–2 sentence explanation>

    **3. Recommended Agent Script**
    <1–3 specific sentences the agent should say to de-escalate, improve sentiment,
    or guide the call toward a successful resolution>

    Base your analysis on patterns from the similar past conversations.
    Be specific, empathetic, and actionable.
""")


def build_prompt(
    current_transcript: str,
    retrieved: list[dict[str, Any]],
) -> list[dict]:
    context_blocks = []
    for i, item in enumerate(retrieved, 1):
        snippet = item["transcript_text"][:MAX_CONTEXT_CHARS]
        if len(item["transcript_text"]) > MAX_CONTEXT_CHARS:
            snippet += "\n[… truncated …]"
        context_blocks.append(
            f"--- Similar Call {i} "
            f"(Category: {item['category']} | {item['sub_category']} | "
            f"Similarity: {item['similarity']:.3f}) ---\n{snippet}"
        )

    context_str = "\n\n".join(context_blocks)

    user_content = (
        "## SIMILAR PAST CONVERSATIONS\n\n"
        + context_str
        + "\n\n## CURRENT CONVERSATION (in progress)\n\n"
        + current_transcript
        + "\n\n## YOUR ANALYSIS\n"
        "Please provide the three sections described in your instructions."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ─── Parse next action from LLM output ───────────────────────────────────────
def extract_next_action(analysis: str) -> str | None:
    """Pull the NEXT ACTION line out of the structured LLM response."""
    for line in analysis.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("NEXT ACTION:"):
            action = stripped[len("NEXT ACTION:"):].strip()
            return action if action else None
    return None


# ─── Human-readable transcript renderer ──────────────────────────────────────
_SPEAKER_STYLE = {
    "agent": {
        "label": "Agent",
        "bg": "#1e3a5f",
        "color": "#7ec8e3",
        "align": "left",
    },
    "customer": {
        "label": "Customer",
        "bg": "#2d1f3d",
        "color": "#c97bdb",
        "align": "right",
    },
}
_EVENT_LABELS = {
    "verification": "🔐 Verification",
    "hold_start": "⏸ Hold started",
    "hold_end": "▶ Hold ended",
    "transfer": "↗ Transfer",
}


def _render_transcript(retrieved_item: dict[str, Any]) -> None:
    """Render a retrieved call's transcript_turns as a styled chat UI."""
    turns = retrieved_item.get("transcript_turns", [])
    if not turns:
        st.text(retrieved_item.get("transcript_text", "No transcript available."))
        return

    html_parts = ['<div style="font-family: sans-serif; padding: 4px 0;">']

    for turn in turns:
        speaker = turn.get("speaker", "unknown").lower()
        text = turn.get("text", "").strip()
        event = turn.get("event", "null")

        # Show event badges between turns
        if event and event != "null":
            label = _EVENT_LABELS.get(event, f"• {event}")
            html_parts.append(
                f'<div style="text-align:center; margin: 8px 0;">'
                f'<span style="background:#333; color:#aaa; font-size:11px; '
                f'padding:2px 10px; border-radius:20px;">{label}</span></div>'
            )

        if not text:
            continue

        style = _SPEAKER_STYLE.get(speaker, {"label": speaker.capitalize(),
                                              "bg": "#2a2a2a", "color": "#ccc",
                                              "align": "left"})
        is_right = style["align"] == "right"
        bubble_margin = "margin-left: auto; margin-right: 8px;" if is_right else "margin-left: 8px; margin-right: auto;"

        html_parts.append(f"""
            <div style="display:flex; flex-direction:column;
                        align-items:{'flex-end' if is_right else 'flex-start'};
                        margin: 6px 0;">
                <div style="font-size:10px; color:#888; margin-bottom:3px;
                            text-align:{'right' if is_right else 'left'};">
                    {style['label']}
                </div>
                <div style="background:{style['bg']}; color:{style['color']};
                            border-radius:12px; padding:10px 14px;
                            max-width:80%; font-size:14px; line-height:1.5;
                            {bubble_margin}">
                    {text}
                </div>
            </div>
        """)

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Telecom RAG Chatbot",
        page_icon="📞",
        layout="wide",
    )

    st.title("📞 Telecom Call Center AI Coach")
    st.caption(
        "Paste a call transcript to predict the next event, assess customer "
        "sentiment, and get agent script recommendations."
    )

    # Load retriever once per session
    if "retriever" not in st.session_state:
        with st.spinner("Loading RAG index …"):
            try:
                st.session_state.retriever = RAGRetriever()
                st.success("RAG index loaded.", icon="✅")
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

    retriever: RAGRetriever = st.session_state.retriever

    # ── Sidebar settings ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Similar calls to retrieve (top-k)", 1, 10, TOP_K)
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.3, step=0.05)
        show_retrieved = st.checkbox("Show all retrieved similar calls", value=False)
        st.divider()
        st.caption(f"Model: {LM_STUDIO_MODEL}")
        st.caption(f"LM Studio: {LM_STUDIO_BASE_URL}")

    # ── Chat history ──────────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {transcript, analysis, retrieved}

    # ── Main input area ───────────────────────────────────────────────────────
    st.subheader("Current Call Transcript")
    placeholder = (
        "Agent: Thank you for calling, how can I help you today?\n"
        "Customer: My bill is way higher than usual and nobody can explain why.\n"
        "Agent: I'm sorry to hear that. Let me pull up your account …"
    )
    transcript_input = st.text_area(
        label="Paste the conversation so far (format: 'Speaker: text' per line)",
        placeholder=placeholder,
        height=250,
        key="transcript_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("Analyze ▶", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear history", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.rerun()

    if analyze_btn and transcript_input.strip():
        with st.spinner("Retrieving similar calls …"):
            retrieved = retriever.retrieve(transcript_input.strip(), top_k=top_k)

        with st.spinner("Generating analysis with gemma3:12b …"):
            messages = build_prompt(transcript_input.strip(), retrieved)
            analysis = call_lm_studio(messages, temperature=temperature)

        st.session_state.history.append(
            {
                "transcript": transcript_input.strip(),
                "retrieved": retrieved,
                "analysis": analysis,
            }
        )

    elif analyze_btn and not transcript_input.strip():
        st.warning("Please paste a transcript before analyzing.")

    # ── Display history (newest first) ────────────────────────────────────────
    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        with st.expander(f"Analysis #{idx}", expanded=(i == 0)):

            # ── Glanceable next action banner ──────────────────────────────
            next_action = extract_next_action(entry["analysis"])
            if next_action:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1a1a2e;
                        border-left: 6px solid #00d4aa;
                        border-radius: 8px;
                        padding: 18px 22px;
                        margin-bottom: 18px;
                    ">
                        <div style="
                            color: #00d4aa;
                            font-size: 11px;
                            font-weight: 700;
                            letter-spacing: 2px;
                            text-transform: uppercase;
                            margin-bottom: 8px;
                        ">⚡ Next Best Action</div>
                        <div style="
                            color: #ffffff;
                            font-size: 20px;
                            font-weight: 600;
                            line-height: 1.4;
                        ">{next_action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Detail columns ─────────────────────────────────────────────
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("**Input Transcript**")
                st.text(entry["transcript"])

            with col_right:
                st.markdown("**Full Analysis**")
                st.markdown(entry["analysis"])

            # ── Most similar call — always shown, human-readable ───────────
            if entry["retrieved"]:
                st.divider()
                top = entry["retrieved"][0]
                st.markdown(
                    f"**Most Similar Past Call** &nbsp;·&nbsp; "
                    f"`{top['call_id']}` &nbsp;·&nbsp; "
                    f"{top['category']} / {top['sub_category']} &nbsp;·&nbsp; "
                    f"Similarity **{top['similarity']:.3f}** &nbsp;·&nbsp; "
                    f"Transfer: {'Yes' if top['call_transfer'] else 'No'} &nbsp;·&nbsp; "
                    f"Callback 7d: {'Yes' if top['callback_7day'] else 'No'}"
                )
                _render_transcript(top)

            # ── Additional retrieved calls (collapsed) ─────────────────────
            if show_retrieved and len(entry["retrieved"]) > 1:
                st.markdown(f"**Other Similar Calls ({len(entry['retrieved']) - 1})**")
                for j, r in enumerate(entry["retrieved"][1:], 2):
                    with st.expander(
                        f"Call {j}: {r['category']} / {r['sub_category']} "
                        f"(sim={r['similarity']:.3f})"
                    ):
                        _render_transcript(r)


if __name__ == "__main__":
    main()

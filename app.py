
import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()   

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not found.\n\n"
        "Create a `.env` file in the same folder as `app.py` with:\n\n"
        "OPENAI_API_KEY=your_key_here"
    )

@st.cache_resource(show_spinner=False)
def get_client():
    """Initialize OpenAI client (cached once)."""
    return OpenAI(api_key=OPENAI_API_KEY)

def call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Unified OpenAI call (ChatCompletion API)."""
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def build_system_prompt(constitution_text: str) -> str:
    """Convert user-defined constitution into a numbered system prompt."""
    lines = [l.strip() for l in constitution_text.splitlines() if l.strip()]

    if not lines:
        lines = [
            "Avoid harm.",
            "Be honest.",
            "Respect user autonomy."
        ]

    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))

    return (
        "You are an AI assistant governed by the following constitution.\n\n"
        "CONSTITUTION:\n"
        f"{numbered}\n\n"
        "All responses must follow ALL principles."
    )


def build_critique_prompt(base_resp: str, constitution: str) -> str:
    return f"""
Evaluate the ASSISTANT_RESPONSE against the USER_CONSTITUTION.

USER_CONSTITUTION:
{constitution}

ASSISTANT_RESPONSE:
{base_resp}

Return JSON ONLY in this format:

[
  {{
    "principle_id": "Principle text or short name",
    "violated": true/false,
    "evidence": "quote or paraphrase",
    "note": "optional"
  }}
]
"""


def build_revision_prompt(base_resp: str, critique_json: str, constitution: str) -> str:
    return f"""
Revise the BASE_RESPONSE so that it satisfies ALL constitution principles.

Follow the issues listed inside CRITIQUE_JSON.

Do NOT mention critics or constitutions.
Keep improvements minimal and concise.

USER_CONSTITUTION:
{constitution}

BASE_RESPONSE:
{base_resp}

CRITIQUE_JSON:
{critique_json}

Return answer in this structure:

<REVISION_REASONING>
(short explanation)
</REVISION_REASONING>

<REVISED_RESPONSE>
(final improved response)
</REVISED_RESPONSE>
"""

def extract_revised_response(full_text: str):
    reasoning = ""
    revised = full_text.strip()

    if "<REVISION_REASONING>" in full_text:
        reasoning = full_text.split("<REVISION_REASONING>", 1)[1].split("</REVISION_REASONING>", 1)[0].strip()

    if "<REVISED_RESPONSE>" in full_text:
        revised = full_text.split("<REVISED_RESPONSE>", 1)[1].split("</REVISED_RESPONSE>", 1)[0].strip()

    return reasoning, revised


# PIPELINE
def run_pipeline(
    model_name: str,
    user_prompt: str,
    constitution: str,
    cycles: int,
    base_temp: float,
    revision_temp: float,
):
    system_prompt = build_system_prompt(constitution)

    # BASE RESPONSE
    base_response = call_openai(
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=base_temp,
    )

    current = base_response
    critiques = []
    reasons = []

    for _ in range(cycles):
        # CRITIQUE
        critique_prompt = build_critique_prompt(current, constitution)
        critique_json = call_openai(
            model=model_name,
            system_prompt="Return JSON ONLY.",
            user_prompt=critique_prompt,
            temperature=0.0,
        )
        critiques.append(critique_json)

        # REVISION
        revision_prompt = build_revision_prompt(current, critique_json, constitution)
        revision_raw = call_openai(
            model=model_name,
            system_prompt="You revise responses carefully.",
            user_prompt=revision_prompt,
            temperature=revision_temp,
        )
        reasoning, revised = extract_revised_response(revision_raw)

        reasons.append(reasoning)
        current = revised

    return base_response, critiques, reasons, current


# STREAMLIT UI
def main():
    st.set_page_config(page_title="Reflect", layout="wide")
    st.title("🧠 Reflect: In-context alignment through model self-critique and revision.")

    if not OPENAI_API_KEY:
        st.stop()

    st.markdown("""
This demo applies **Base → Critique → Revision** using *your own constitution*.
""")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    st.sidebar.header("Settings")

    model_name = st.sidebar.selectbox(
        "Model",
        ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-preview"],
        index=0,
    )

    cycles = st.sidebar.slider("Critique+Revision cycles", 1, 3, 1)
    base_temp = st.sidebar.slider("Base temp", 0.0, 1.2, 0.7)
    revision_temp = st.sidebar.slider("Revision temp", 0.0, 1.2, 0.3)

    constitution_default = (
        "Avoid harmful or dangerous instructions.\n"
        "Be honest and non-deceptive.\n"
        "Respect user autonomy.\n"
        "Be helpful and provide useful information.\n"
        "Avoid discrimination or bias.\n"
    )

    constitution = st.sidebar.text_area(
        "Your Constitution",
        value=constitution_default,
        height=180,
    )

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Applying constitution…"):
                base, crits, reasons, final = run_pipeline(
                    model_name=model_name,
                    user_prompt=user_input,
                    constitution=constitution,
                    cycles=cycles,
                    base_temp=base_temp,
                    revision_temp=revision_temp,
                )

            st.write(final)
            st.session_state.history.append({"role": "assistant", "content": final})

            
            st.markdown("---")
            tabs = st.tabs([
                "Base Response",
                "Critique JSON",
                "Revision Reasoning",
                "Final Response",
                "Raw JSON"
            ])

            with tabs[0]:
                st.write(base)

            with tabs[1]:
                for i, c in enumerate(crits, 1):
                    st.markdown(f"### Critique Cycle {i}")
                    st.code(c, language="json")

            with tabs[2]:
                for i, r in enumerate(reasons, 1):
                    st.markdown(f"### Revision Reasoning {i}")
                    st.write(r)

            with tabs[3]:
                st.write(final)

            with tabs[4]:
                raw = {
                    "user_prompt": user_input,
                    "constitution": constitution,
                    "base_response": base,
                    "critiques": crits,
                    "revision_reasonings": reasons,
                    "final_response": final,
                }
                st.code(json.dumps(raw, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

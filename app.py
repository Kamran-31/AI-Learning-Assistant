"""
CogniLearn AI - Intelligent Learning Companion
Modern Interface with Document Processing & Resource Linking
Configured for Streamlit Cloud Deployment Secrets
"""

import streamlit as st
from groq import Groq
import pypdf
import docx

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CogniLearn AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main chat canvas width & spacing */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Clean modern app header */
    .chat-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .chat-header h1 {
        font-size: 1.45rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .chat-header span {
        font-size: 0.82rem;
        color: #94a3b8;
    }

/* Clean modern logo-style app header */
    .brand-container {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-top: 10px;
        padding-bottom: 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .brand-logo-text {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.35; /* Prevents text clipping at the top */
        padding-top: 4px;
        padding-bottom: 2px;
        display: inline-block;
        background: linear-gradient(135deg, #1d4ed8 0%, #6366f1 50%, #9333ea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-tagline {
        font-size: 0.88rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 2px;
    }
    .brand-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 9999px;
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        border: 1px solid rgba(37, 99, 235, 0.25);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    
    /* Streamlit Chat Messages polish */
    .stChatMessage {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    /* Code blocks */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* File uploader container styling */
    div[data-testid="stFileUploader"] {
        border-radius: 10px;
        padding: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Secret Management from Streamlit Settings
# ---------------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ `GROQ_API_KEY` not found in Streamlit Secrets. Please add `GROQ_API_KEY = \"your_key\"` under Manage App ➔ Settings ➔ Secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------------------------------------------------
# Document Parser Utilities
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text content from PDF, DOCX, TXT, or MD files."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""
    try:
        if file_type == "pdf":
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type in ["txt", "md"]:
            text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.sidebar.error(f"Failed to parse file: {str(e)}")
    return text.strip()

# ---------------------------------------------------------
# Sidebar Controls & File Upload
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Study Session Settings")

    learning_mode = st.selectbox(
        "Learning Mode",
        [
            "Comprehensive Conceptual Explanation",
            "Socratic Discussion & Guidance",
            "In-Depth Technical & Code Walkthrough",
            "Structured Curriculum Roadmap",
            "Active Recall & Quiz Generator",
            "Feynman Technique (Simple Analogies)",
        ],
        index=0,
    )

    expertise_level = st.select_slider(
        "Target Level",
        options=["Beginner", "Intermediate", "Advanced", "Domain Specialist"],
        value="Intermediate",
    )

    st.markdown("---")
    st.markdown("### 📎 Attach Knowledge Material")
    uploaded_files = st.file_uploader(
        "Upload notes, textbooks, slides, or assignments (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    document_context = ""
    if uploaded_files:
        parsed_docs = []
        for file in uploaded_files:
            content = extract_text_from_file(file)
            if content:
                parsed_docs.append(f"--- START OF ATTACHED FILE: {file.name} ---\n{content[:8000]}\n--- END OF FILE ---")
        document_context = "\n\n".join(parsed_docs)
        st.success(f"{len(uploaded_files)} file(s) attached to memory context.")

    st.markdown("---")
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# System Instructions Engine
# ---------------------------------------------------------
def construct_system_prompt(mode: str, level: str, doc_ctx: str) -> str:
    instructions = [
        "You are CogniLearn AI, an elite educational mentor and subject-matter expert.",
        f"Learner experience level: {level}.",
        f"Active engagement style: {mode}.",
        "Communication standards:",
        "- Provide deep, articulated, clear, and comprehensive explanations.",
        "- Use Markdown formatting generously (structured tables, clear bullet points, bold markers, and code snippets).",
        "- For architectural designs, system workflows, or lifecycles, output a structured Mermaid.js diagram block.",
        "- Provide curated resource recommendations (documentation, standard papers, books, or reputable websites with markdown links) at the end of comprehensive explanations.",
        "- When the user attaches files or notes, prioritize synthesizing and answering based on the provided material.",
    ]

    if mode == "Socratic Discussion & Guidance":
        instructions.append(
            "Do not hand over answers immediately. Scaffold the learning by asking guiding, thought-provoking questions and offering constructive hints."
        )
    elif mode == "Active Recall & Quiz Generator":
        instructions.append(
            "Formulate realistic exam-style questions, scenario dilemmas, and self-evaluation rubrics. Hide answers inside `<details><summary>Click to view solution</summary>...</details>` blocks."
        )

    if doc_ctx:
        instructions.append(f"\nCONTEXT FROM ATTACHED DOCUMENTS:\n{doc_ctx}")

    return "\n".join(instructions)

# ---------------------------------------------------------
# Main Chat Canvas
# ---------------------------------------------------------
st.markdown(
    """
    <div class="brand-container">
        <div>
            <div class="brand-logo-text">SYNAPSE.AI</div>
            <div class="brand-tagline">Autonomous Cognitive Tutor & Knowledge Synthesizer</div>
        </div>
        <span class="brand-badge">Adaptive Engine</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! What would you like to explore or learn today?\n\n"
                "Feel free to upload textbooks, slide decks, or lecture notes in the sidebar, "
                "or ask deep conceptual questions, request roadmaps, or prepare for exams."
            ),
        }
    ]

# Render Message History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# User Interaction & Streaming
user_prompt = st.chat_input("Ask a question, request a breakdown, or reference uploaded documents...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_prompt)

    system_prompt = construct_system_prompt(learning_mode, expertise_level, document_context)
    api_messages = [{"role": "system", "content": system_prompt}]

    for m in st.session_state.messages[-10:]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant", avatar="🎓"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=api_messages,
                temperature=0.35,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error communicating with API: {str(e)}")

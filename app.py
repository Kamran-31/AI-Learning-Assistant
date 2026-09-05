"""
AI Learning Assistant - Intelligent Adaptive Tutor
Built with Streamlit & Groq API
Model: openai/gpt-oss-120b
"""

import os
import streamlit as st
from groq import Groq

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="CogniLearn AI | Adaptive Learning Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header banner */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin: 0;
    }

    /* Mode Pill Card */
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 12px;
    }

    /* Metric card widgets */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: left;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Quick action buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }

    /* Chat bubble polish */
    .stChatMessage {
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 8px;
    }

    /* Code block styling */
    code, pre {
        font-family: 'Fira Code', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API Initialization & Secret Fetching
# ---------------------------------------------------------
def get_groq_client():
    api_key = None
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    elif os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None
    return Groq(api_key=api_key)

# ---------------------------------------------------------
# Sidebar Configuration & Learning Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Learning Engine Controls")
    st.markdown("Customize pedagogical style & depth for your session.")

    # API Status Check
    client = get_groq_client()
    if client:
        st.success("✅ Groq API Connected", icon="⚡")
    else:
        st.warning("⚠️ GROQ_API_KEY not detected in secrets or env.")
        manual_key = st.text_input("Enter Groq API Key:", type="password")
        if manual_key:
            client = Groq(api_key=manual_key)
            st.success("Key set for current session!")

    st.markdown("---")

    # Learning Fields & Mode Selectors
    learning_mode = st.selectbox(
        "🎯 Pedagogical Mode",
        [
            "Interactive Conceptual Explanation",
            "Socratic Questioning & Reasoning",
            "Deep Dive with Code & Architecture",
            "Curriculum / Roadmap Builder",
            "Active Recall & Quiz Generator",
            "Feynman Technique (Explain like I am 12)",
        ],
        index=0,
    )

    expertise_level = st.select_slider(
        "📈 Learner Background",
        options=["Absolute Beginner", "Intermediate", "Advanced Engineer", "Domain Specialist"],
        value="Intermediate",
    )

    include_diagrams = st.checkbox("Include Mermaid.js Flowcharts/Diagrams", value=True)
    include_examples = st.checkbox("Generate Real-World Analogies & Case Studies", value=True)
    generate_exercises = st.checkbox("Attach Practice Questions & Challenges", value=True)

    st.markdown("---")
    temperature = st.slider("Response Creativity (Temp)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 24px; text-align: center;">
            Powered by <b>openai/gpt-oss-120b</b> on Groq LPU™ Engine.<br>
            Ultra-low latency inference.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# System Prompt Constructor
# ---------------------------------------------------------
def construct_system_prompt(mode: str, level: str, diagrams: bool, analogies: bool, practice: bool) -> str:
    instructions = [
        f"You are CogniLearn AI, a world-class educational AI architect and personalized tutor.",
        f"The learner's current background knowledge level is: {level}.",
        f"Active Learning Mode: {mode}.",
        "Your responses should be structured, crystal-clear, beautifully formatted in Markdown, and engaging.",
    ]

    if mode == "Interactive Conceptual Explanation":
        instructions.append(
            "Structure your response with: 1. Core Concept Overview, 2. The 'Why It Matters', 3. How It Works step-by-step, 4. Key Takeaways."
        )
    elif mode == "Socratic Questioning & Reasoning":
        instructions.append(
            "Adopt the Socratic method. Do not simply hand over complete answers; guide the learner through thoughtful probing questions, hint ladders, and reflective deductions."
        )
    elif mode == "Deep Dive with Code & Architecture":
        instructions.append(
            "Provide production-grade code snippets, architecture breakdowns, edge cases, time/space complexity analysis, and modern engineering best practices."
        )
    elif mode == "Curriculum / Roadmap Builder":
        instructions.append(
            "Design a modular, week-by-week or milestone-based mastery roadmap with learning objectives, curated project ideas, and verification checkpoints."
        )
    elif mode == "Active Recall & Quiz Generator":
        instructions.append(
            "Generate a structured knowledge check containing: 3 multiple-choice questions with answer keys hidden in collapsible details, 2 scenario-based conceptual challenges, and self-evaluation rubrics."
        )
    elif mode == "Feynman Technique (Explain like I am 12)":
        instructions.append(
            "Explain complex concepts using intuitive, vivid analogies, everyday vocabulary, and zero unnecessary jargon. Keep it thoroughly accessible without compromising core truth."
        )

    if diagrams:
        instructions.append("Whenever depicting flows, architectures, states, or relationships, provide a clean `mermaid` code block.")

    if analogies:
        instructions.append("Include at least one vivid real-world comparison or industrial case study.")

    if practice:
        instructions.append("Conclude your response with a '🧠 Mini Challenge' or 'Self-Review Question' to solidify understanding.")

    return "\n".join(instructions)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <span class="mode-badge">{learning_mode.upper()}</span>
        <div class="hero-title">CogniLearn AI Studio</div>
        <p class="hero-subtitle">Adaptive Intelligence for Deep Technical & Conceptual Mastery • Groq Acceleration</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Metric Highlights
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-val">gpt-oss-120b</div><div class="metric-lbl">Target Model</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{expertise_level}</div><div class="metric-lbl">Learner Depth</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-val">Groq LPU</div><div class="metric-lbl">Inference Stack</div></div>', unsafe_allow_html=True)
with c4:
    total_turns = len(st.session_state.get("messages", [])) // 2
    st.markdown(f'<div class="metric-card"><div class="metric-val">{total_turns}</div><div class="metric-lbl">Discussions Held</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Chat State Management
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your AI Learning Companion. What topic, system, or concept would you like to master today?\n\n"
                "💡 *Quick prompt ideas:*\n"
                "- *'Explain Distributed Consensus (Raft & Paxos) with diagrams'*\n"
                "- *'Design a 6-week curriculum to learn Generative AI Agents'*\n"
                "- *'Help me debug and optimize Transformer Attention mechanisms'*\n"
                "- *'Quiz me on System Design concepts like Sharding and CAP theorem'*"
            ),
        }
    ]

# Display Conversation History
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# User Input & Streaming Generation
# ---------------------------------------------------------
user_prompt = st.chat_input("Enter a topic, question, or learning objective...")

if user_prompt:
    if not client:
        st.error("Please provide a valid Groq API Key in `.streamlit/secrets.toml` or via the sidebar.")
        st.stop()

    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_prompt)

    # Build conversation payload
    system_prompt = construct_system_prompt(
        learning_mode, expertise_level, include_diagrams, include_examples, generate_exercises
    )

    api_messages = [{"role": "system", "content": system_prompt}]
    # Keep last 10 messages for contextual continuity without token overflow
    for m in st.session_state.messages[-10:]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Stream AI response
    with st.chat_message("assistant", avatar="🎓"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=api_messages,
                temperature=temperature,
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
            st.error(f"Error communicating with Groq API: {str(e)}")

import streamlit as st
import os
from html import escape
from dotenv import load_dotenv
from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage

def format_message_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)

load_dotenv()

st.set_page_config(
    page_title="FreightIQ | Carrier Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0d1b2a 0%, #0a0f1e 50%, #070b14 100%);
}

/* ── Header ── */
.fiq-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 28px 0 8px 0;
    border-bottom: 1px solid rgba(0, 212, 255, 0.12);
    margin-bottom: 24px;
}
.fiq-logo {
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(0, 212, 255, 0.6));
}
.fiq-title {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff 0%, #00d4ff 60%, #7b61ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0;
}
.fiq-subtitle {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.3px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"][data-type="user"] {
    border-color: rgba(0, 212, 255, 0.2) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

/* ── Tool execution card ── */
.tool-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.05) 0%, rgba(123,97,255,0.05) 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-left: 3px solid #00d4ff;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    position: relative;
    overflow: hidden;
}
.tool-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, #00d4ff, transparent);
}
.tool-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.tool-badge {
    background: rgba(0, 212, 255, 0.15);
    color: #00d4ff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.tool-name {
    color: #00d4ff;
    font-size: 0.85rem;
    font-weight: 700;
}
.tool-output {
    color: rgba(255,255,255,0.45);
    font-size: 0.75rem;
    line-height: 1.5;
    white-space: pre-wrap;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
}
.sidebar-section {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 500;
}
.status-ok {
    background: rgba(0, 255, 136, 0.12);
    color: #00ff88;
    border: 1px solid rgba(0, 255, 136, 0.25);
}
.status-err {
    background: rgba(255, 80, 80, 0.12);
    color: #ff5050;
    border: 1px solid rgba(255, 80, 80, 0.25);
}
.tool-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.78rem;
    color: rgba(255,255,255,0.7);
}
.tool-item:last-child { border-bottom: none; }
.tool-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.tool-item-name { font-weight: 600; color: rgba(255,255,255,0.9); font-size: 0.78rem; }
.tool-item-desc { color: rgba(255,255,255,0.4); font-size: 0.72rem; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(0, 212, 255, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #00d4ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0, 212, 255, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.4); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="fiq-header">
    <span class="fiq-logo">🚚</span>
    <div>
        <div class="fiq-title">FreightIQ</div>
        <div class="fiq-subtitle">Agentic Carrier Intelligence · LangGraph · PyTorch · ChromaDB · SQLite</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
TOOLS = [
    ("🔍", "Semantic Search", "Vector DB + PyTorch MLP re-ranker"),
    ("🗄️", "SQL Database", "Structured carrier lookups via SQLite"),
    ("🌐", "Web Search", "Live DuckDuckGo market research"),
    ("🔢", "Freight Class", "NMFC density calculator"),
]

with st.sidebar:
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    status_class = "status-ok" if api_key_set else "status-err"
    status_dot = "●" if api_key_set else "●"
    status_text = "Gemini 2.0 Flash · Connected" if api_key_set else "API Key Missing"

    tool_items_html = "".join([
        f'<div class="tool-item"><span class="tool-icon">{icon}</span>'
        f'<div><div class="tool-item-name">{name}</div>'
        f'<div class="tool-item-desc">{desc}</div></div></div>'
        for icon, name, desc in TOOLS
    ])

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">Agent Status</div>
        <span class="status-badge {status_class}">{status_dot} {status_text}</span>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-title">Available Tools</div>
        {tool_items_html}
    </div>
    """, unsafe_allow_html=True)

    if not api_key_set:
        st.warning("Set `GEMINI_API_KEY` in your `.env` file to enable the agent.")

    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(format_message_content(message.content))
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.write(format_message_content(message.content))

# Chat input
if user_query := st.chat_input("Ask about carriers, rates, or calculate freight class…"):
    with st.chat_message("user"):
        st.write(user_query)

    st.session_state.messages.append(HumanMessage(content=user_query))

    with st.chat_message("assistant"):
        step_container = st.container()
        response_container = st.empty()

        final_answer = ""
        try:
            with st.spinner("Reasoning…"):
                for event in graph.stream({"messages": st.session_state.messages}, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "tools":
                            for msg in node_output.get("messages", []):
                                safe_name = escape(str(msg.name))
                                safe_output = escape(str(msg.content)[:400])
                                with step_container:
                                    st.markdown(f"""
                                    <div class="tool-card">
                                        <div class="tool-card-header">
                                            <span class="tool-badge">TOOL</span>
                                            <span class="tool-name">{safe_name}</span>
                                        </div>
                                        <div class="tool-output">{safe_output}</div>
                                    </div>""", unsafe_allow_html=True)

                        elif node_name == "agent":
                            messages = node_output.get("messages", [])
                            if messages and messages[-1].content:
                                final_answer = format_message_content(messages[-1].content)
                                response_container.write(final_answer)

            if final_answer:
                st.session_state.messages.append(AIMessage(content=final_answer))

        except Exception as e:
            st.error(f"Agent error: {str(e)}")

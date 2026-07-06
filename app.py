from dotenv import load_dotenv
# 1. Load environment variables first before importing any project packages that instantiate LLMs
load_dotenv()

import logging
import streamlit as st
import os
import json
import textwrap
from datetime import datetime, timezone
from html import escape
from agent.graph import build_graph
from agent.locks import setup_lock, feedback_lock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from rag.utils import save_feedback, format_message_content
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 2. Custom LangChain callback handler to stream text tokens directly to Streamlit
class StreamlitTokenCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder, initial_text=""):
        self.placeholder = placeholder
        self.tokens = [initial_text] if initial_text else []
        if initial_text:
            self.placeholder.write(initial_text)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Ignore tokens belonging to tool calls to avoid printing JSON payloads in the chat box
        try:
            chunk = kwargs.get("chunk")
            if chunk and hasattr(chunk, "message") and hasattr(chunk.message, "tool_call_chunks") and chunk.message.tool_call_chunks:
                return
        except Exception:
            pass
        
        self.tokens.append(token)
        self.placeholder.write("".join(self.tokens))

# 3. Helper to serialize user feedback on query-response matches
# save_feedback helper has been relocated to rag/utils.py to avoid Streamlit module import errors on CLI tasks.

# Helper to dynamically trim conversation history without orphaning ToolMessages or starting with a non-HumanMessage
def get_windowed_messages(messages, max_messages=8):
    if len(messages) <= max_messages:
        return messages
    
    slice_idx = -max_messages
    
    # Trace backward to ensure no orphaned tool/assistant sequences
    while abs(slice_idx) < len(messages):
        first_msg = messages[slice_idx]
        if isinstance(first_msg, ToolMessage):
            slice_idx -= 1
        elif isinstance(first_msg, AIMessage) and first_msg.tool_calls:
            slice_idx -= 1
        else:
            break
            
    # Always initiate the message sequence with a HumanMessage to satisfy API validation
    while abs(slice_idx) < len(messages) and not isinstance(messages[slice_idx], HumanMessage):
        slice_idx -= 1
        
    return messages[slice_idx:]

# 4. Cache graph compilation inside Streamlit to prevent hot-reload compilation loops
@st.cache_resource
def get_graph():
    logger.info("Compiling agent LangGraph workflow...")
    return build_graph()

# 5. Thread-safe database auto-initialization utilizing the global locks module
if not os.path.exists(config.DB_PATH) or os.path.getsize(config.DB_PATH) == 0 or not os.path.exists(config.CHROMA_PATH):
    with setup_lock:
        # Double-check condition once lock is acquired
        if not os.path.exists(config.DB_PATH) or os.path.getsize(config.DB_PATH) == 0 or not os.path.exists(config.CHROMA_PATH):
            logger.info("Database or vector index missing or empty. Triggering thread-safe auto-setup...")
            try:
                from setup import main as run_setup
                run_setup()
            except Exception as e:
                logger.error(f"Failed to auto-initialize data environment: {e}")

# 6. Initialize session state variables safely at the module level
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "voted_message_index" not in st.session_state:
    st.session_state.voted_message_index = -1

if "tool_executions" not in st.session_state:
    st.session_state.tool_executions = {}

# format_message_content has been relocated to rag/utils.py

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
    max-height: 180px;
    overflow-y: auto;
}

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
.status-warning {
    background: rgba(255, 165, 0, 0.12);
    color: #ffa500;
    border: 1px solid rgba(255, 165, 0, 0.25);
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
.query-counter {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    text-align: right;
    padding: 4px 0 0 0;
}

[data-testid="stChatInput"] {
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.03) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(0, 212, 255, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
}

[data-testid="stSpinner"] { color: #00d4ff !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0, 212, 255, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.4); }

/* Premium Table Styling for structured carrier tables */
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 15px 0 !important;
    font-size: 0.82rem !important;
    background-color: rgba(255, 255, 255, 0.01) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0, 212, 255, 0.1) !important;
}
th {
    background-color: rgba(0, 212, 255, 0.08) !important;
    color: #00d4ff !important;
    text-align: left !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(0, 212, 255, 0.2) !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.5px !important;
}
td {
    padding: 10px 14px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
    color: rgba(255, 255, 255, 0.8) !important;
}
tr:hover {
    background-color: rgba(0, 212, 255, 0.03) !important;
}

/* Premium Markdown Links styling */
a {
    color: #00d4ff !important;
    text-decoration: none !important;
    transition: color 0.2s ease !important;
}
a:hover {
    color: #7b61ff !important;
    text-decoration: underline !important;
}

/* Premium suggestion chips and buttons styling */
div[data-testid="stButton"] button {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 8px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    font-size: 0.78rem !important;
    padding: 6px 12px !important;
    font-weight: 500 !important;
}
div[data-testid="stButton"] button:hover {
    background: rgba(0, 212, 255, 0.05) !important;
    border-color: #00d4ff !important;
    color: #00d4ff !important;
    box-shadow: 0 0 14px rgba(0, 212, 255, 0.2) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] button:active {
    transform: translateY(1px) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fiq-header">
    <span class="fiq-logo">🚚</span>
    <div>
        <div class="fiq-title">FreightIQ</div>
        <div class="fiq-subtitle">Agentic Carrier Intelligence · LangGraph · PyTorch · ChromaDB · SQLite</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Dynamic checks for reranker model configuration & feedback logs size
weights_path = os.path.join(config.BASE_DIR, "rag", "data", "reranker_weights.pt")
has_weights = os.path.exists(weights_path)
reranker_status = "PyTorch MLP (Fine-tuned)" if has_weights else "Cosine Similarity (Fallback)"
reranker_class = "status-ok" if has_weights else "status-warning"

feedback_path = os.path.join(config.BASE_DIR, "rag", "data", "feedback.json")
feedback_count = 0
if os.path.exists(feedback_path):
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
            feedback_count = len(feedback_data)
    except Exception:
        pass

semantic_search_desc = "Vector DB + Fine-tuned PyTorch MLP" if has_weights else "Vector DB + Cosine Reranker fallback"

TOOLS = [
    ("🔍", "Semantic Search", semantic_search_desc),
    ("🗄️", "SQL Database", "Structured carrier lookups via SQLite"),
    ("🌐", "Web Search", "Live DuckDuckGo market research"),
    ("🔢", "Freight Class", "NMFC density calculator"),
]

with st.sidebar:
    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    status_class = "status-ok" if api_key_set else "status-err"
    status_text = "Llama 3.1 8B · Connected" if api_key_set else "Groq API Key Missing"

    tool_items_html = "".join([
        f'<div class="tool-item"><span class="tool-icon">{icon}</span>'
        f'<div><div class="tool-item-name">{name}</div>'
        f'<div class="tool-item-desc">{desc}</div></div></div>'
        for icon, name, desc in TOOLS
    ])

    remaining = config.MAX_QUERIES_PER_SESSION - st.session_state.query_count
    counter_html = f'<div class="query-counter">{remaining}/{config.MAX_QUERIES_PER_SESSION} queries remaining</div>'

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-title">Agent Status</div>
        <div style="margin-bottom: 8px;"><span class="status-badge {status_class}">● LLM: {status_text}</span></div>
        <div style="margin-bottom: 8px;"><span class="status-badge {reranker_class}">● Re-ranker: {reranker_status}</span></div>
        <div style="font-size: 0.72rem; color: rgba(255,255,255,0.4); margin-top: 4px; margin-left: 2px;">
            Feedback Logs: {feedback_count} entries
        </div>
        {counter_html}
    </div>
    <div class="sidebar-section">
        <div class="sidebar-title">Available Tools</div>
        {tool_items_html}
    </div>
    """, unsafe_allow_html=True)

    if not api_key_set:
        st.warning("Set `GROQ_API_KEY` in your `.env` file to enable the agent.")

    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.voted_message_index = -1
        st.session_state.tool_executions = {}
        st.rerun()

for idx, message in enumerate(st.session_state.messages):
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(format_message_content(message.content))
            
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            # Render persistent tool executions associated with the preceding human query (idx - 1)
            query_idx = idx - 1
            if query_idx in st.session_state.tool_executions:
                for tool_call in st.session_state.tool_executions[query_idx]:
                    st.markdown(f"""
                    <div class="tool-card">
                        <div class="tool-card-header">
                            <span class="tool-badge">TOOL</span>
                            <span class="tool-name">{tool_call["name"]}</span>
                        </div>
                        <div class="tool-output">{tool_call["output"]}</div>
                    </div>""", unsafe_allow_html=True)
            st.write(format_message_content(message.content))

# Render Query Suggestion Chips only on landing (empty chat history)
if not st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    clicked_query = None
    with col1:
        if st.button("🚛 FL produce & class", key="chip_fl", use_container_width=True):
            clicked_query = "Find a carrier located in Florida (FL) that handles fresh produce. What are their DOT and MC numbers, and how many years have they been operating? Also, what is the freight class for a 220 lbs crate of fresh produce measuring 36x36x36 inches? Be detailed."
    with col2:
        if st.button("🗄️ Midwest Hazmat Flatbeds", key="chip_midwest", use_container_width=True):
            clicked_query = "We need flatbed carriers that handle hazardous materials in the Midwest."
    with col3:
        if st.button("🧭 OH safety check & class", key="chip_ohio", use_container_width=True):
            clicked_query = "Find a carrier headquartered in Ohio with a satisfactory safety rating. Also, what would the freight class be for a 150 lbs crate measuring 24x24x24 inches?"
else:
    clicked_query = None

chat_input_val = st.chat_input("Ask about carriers, rates, or calculate freight class…")
user_query = clicked_query if clicked_query else chat_input_val

if user_query:
    if st.session_state.query_count >= config.MAX_QUERIES_PER_SESSION:
        st.error(f"Session query limit reached ({config.MAX_QUERIES_PER_SESSION} queries). Click **Reset Conversation** to continue.")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)

    st.session_state.messages.append(HumanMessage(content=user_query))
    st.session_state.query_count += 1
    current_query_idx = len(st.session_state.messages) - 1
    logger.info(f"User query #{st.session_state.query_count}: '{user_query[:80]}'")

    with st.chat_message("assistant"):
        step_container = st.container()
        response_container = st.empty()

        try:
            # Sliding window: only send the last N messages to the LLM (turn-aware to avoid sequence errors)
            windowed_messages = get_windowed_messages(st.session_state.messages, config.CONVERSATION_WINDOW)

            # Retrieve compiled cached graph resource
            graph = get_graph()

            # Instantiate streaming callback handler
            stream_handler = StreamlitTokenCallbackHandler(response_container)
            accumulated_responses = []
            final_answer = ""

            with st.spinner("Reasoning…"):
                for event in graph.stream(
                    {"messages": windowed_messages}, 
                    config={"callbacks": [stream_handler]}, 
                    stream_mode="updates"
                ):
                    for node_name, node_output in event.items():
                        if node_name == "tools":
                            for msg in node_output.get("messages", []):
                                safe_name = escape(str(msg.name))
                                # Clean mid-word truncation using textwrap.shorten
                                raw_output = str(msg.content)
                                if len(raw_output) > config.TOOL_TRUNCATION_LIMIT:
                                    truncated_output = textwrap.shorten(
                                        raw_output, 
                                        width=config.TOOL_TRUNCATION_LIMIT, 
                                        placeholder="..."
                                    )
                                else:
                                    truncated_output = raw_output
                                
                                safe_output = escape(truncated_output)
                                # Persist the tool card execution to state
                                if current_query_idx not in st.session_state.tool_executions:
                                    st.session_state.tool_executions[current_query_idx] = []
                                st.session_state.tool_executions[current_query_idx].append({
                                    "name": safe_name,
                                    "output": safe_output
                                })

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
                                # Force final write to make sure text formatting is clean
                                response_container.write(final_answer)

            if final_answer:
                st.session_state.messages.append(AIMessage(content=final_answer))
                logger.info(f"Agent response complete. Session total: {st.session_state.query_count} queries.")
                st.rerun()

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            st.error(f"Agent error: {str(e)}")

# 7. Render feedback button loop only for the final agent message block
if st.session_state.messages and isinstance(st.session_state.messages[-1], AIMessage):
    last_response = st.session_state.messages[-1].content
    last_query = ""
    for msg in reversed(st.session_state.messages[:-1]):
        if isinstance(msg, HumanMessage):
            last_query = msg.content
            break
            
    if last_query:
        st.write("---")
        last_msg_idx = len(st.session_state.messages) - 1
        if st.session_state.voted_message_index != last_msg_idx:
            st.caption("Was this response helpful? Saves feedback to improve future reranker training:")
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 10])
            with fb_col1:
                if st.button("👍 Yes", key="thumbs_up", use_container_width=True):
                    save_feedback(last_query, last_response, "up")
                    st.session_state.voted_message_index = last_msg_idx
                    st.toast("Thank you! Feedback saved to feedback.json.")
                    st.rerun()
            with fb_col2:
                if st.button("👎 No", key="thumbs_down", use_container_width=True):
                    save_feedback(last_query, last_response, "down")
                    st.session_state.voted_message_index = last_msg_idx
                    st.toast("Thank you! Feedback saved to feedback.json.")
                    st.rerun()
        else:
            st.success("Feedback logged successfully! Thank you for helping train the re-ranker.")

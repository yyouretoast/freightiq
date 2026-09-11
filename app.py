from dotenv import load_dotenv
load_dotenv()

import logging
import streamlit as st
import os
import textwrap
from html import escape
from agent.graph import build_graph
from agent.nodes import get_windowed_messages
from utils.locks import setup_lock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from rag.utils import save_feedback, load_feedback, format_message_content
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class StreamlitTokenCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder, initial_text=""):
        self.placeholder = placeholder
        self.tokens = [initial_text] if initial_text else []
        if initial_text:
            self.placeholder.write(initial_text)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        try:
            chunk = kwargs.get("chunk")
            if chunk and hasattr(chunk, "message") and hasattr(chunk.message, "tool_call_chunks") and chunk.message.tool_call_chunks:
                return
        except Exception as e:
            logger.debug(f"Error checking tool call chunk: {e}")
        
        self.tokens.append(token)
        self.placeholder.write("".join(self.tokens))

@st.cache_resource
def get_graph():
    logger.info("Compiling agent LangGraph workflow...")
    return build_graph()

init_sentinel = os.path.join(config.DATA_DIR, ".init_complete")
if not os.path.exists(init_sentinel) or not os.path.exists(config.DB_PATH) or not os.path.exists(config.CHROMA_PATH):
    with setup_lock:
        if not os.path.exists(init_sentinel) or not os.path.exists(config.DB_PATH) or not os.path.exists(config.CHROMA_PATH):
            logger.info("Database or vector index missing. Triggering auto-setup...")
            try:
                from scripts.seed_db import main as run_setup
                run_setup()
                with open(init_sentinel, "w", encoding="utf-8") as f:
                    f.write("OK")
            except Exception as e:
                logger.error(f"Failed to auto-initialize data environment: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "voted_message_index" not in st.session_state:
    st.session_state.voted_message_index = -1

st.set_page_config(
    page_title="FreightIQ | Carrier Operations Router",
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
}

/* Modality 1: SQL (Electric Blue) */
.tool-card.tool-sql {
    background: linear-gradient(135deg, rgba(0, 132, 255, 0.08) 0%, rgba(0, 132, 255, 0.02) 100%);
    border: 1px solid rgba(0, 132, 255, 0.28);
    border-left: 4px solid #0084ff;
}
.tool-card.tool-sql::before { background: linear-gradient(90deg, #0084ff, transparent); }
.tool-card.tool-sql .tool-badge { background: rgba(0, 132, 255, 0.20); color: #00a2ff; }
.tool-card.tool-sql .tool-name { color: #00a2ff; }

/* Modality 2: Hybrid Semantic (Neon Purple) */
.tool-card.tool-semantic {
    background: linear-gradient(135deg, rgba(179, 71, 255, 0.08) 0%, rgba(179, 71, 255, 0.02) 100%);
    border: 1px solid rgba(179, 71, 255, 0.28);
    border-left: 4px solid #b347ff;
}
.tool-card.tool-semantic::before { background: linear-gradient(90deg, #b347ff, transparent); }
.tool-card.tool-semantic .tool-badge { background: rgba(179, 71, 255, 0.20); color: #ca75ff; }
.tool-card.tool-semantic .tool-name { color: #ca75ff; }

/* Modality 3: FMCSA SAFER (Emerald Green) */
.tool-card.tool-fmcsa {
    background: linear-gradient(135deg, rgba(0, 230, 118, 0.08) 0%, rgba(0, 230, 118, 0.02) 100%);
    border: 1px solid rgba(0, 230, 118, 0.28);
    border-left: 4px solid #00e676;
}
.tool-card.tool-fmcsa::before { background: linear-gradient(90deg, #00e676, transparent); }
.tool-card.tool-fmcsa .tool-badge { background: rgba(0, 230, 118, 0.20); color: #00e676; }
.tool-card.tool-fmcsa .tool-name { color: #00e676; }

/* Modality 4: Calculator (Amber Orange) */
.tool-card.tool-calc {
    background: linear-gradient(135deg, rgba(255, 145, 0, 0.08) 0%, rgba(255, 145, 0, 0.02) 100%);
    border: 1px solid rgba(255, 145, 0, 0.28);
    border-left: 4px solid #ff9100;
}
.tool-card.tool-calc::before { background: linear-gradient(90deg, #ff9100, transparent); }
.tool-card.tool-calc .tool-badge { background: rgba(255, 145, 0, 0.20); color: #ffaa33; }
.tool-card.tool-calc .tool-name { color: #ffaa33; }

/* Modality 5: Web API (Cyan) */
.tool-card.tool-web {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.08) 0%, rgba(0, 229, 255, 0.02) 100%);
    border: 1px solid rgba(0, 229, 255, 0.28);
    border-left: 4px solid #00e5ff;
}
.tool-card.tool-web::before { background: linear-gradient(90deg, #00e5ff, transparent); }
.tool-card.tool-web .tool-badge { background: rgba(0, 229, 255, 0.20); color: #00e5ff; }
.tool-card.tool-web .tool-name { color: #00e5ff; }

.tool-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.tool-badge {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.tool-name {
    font-size: 0.82rem;
    font-weight: 700;
}
.tool-output {
    color: rgba(255,255,255,0.55);
    font-size: 0.74rem;
    line-height: 1.5;
    white-space: pre-wrap;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
    max-height: 180px;
    overflow-y: auto;
}

.fiq-telemetry {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: -14px 0 20px 0;
    padding-bottom: 12px;
}
.telemetry-pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.70rem;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.2px;
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
    <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 14px rgba(0, 212, 255, 0.6)); flex-shrink: 0;">
        <rect width="46" height="46" rx="12" fill="url(#fiq_bg)" fill-opacity="0.2" stroke="url(#fiq_border)" stroke-width="1.6"/>
        <path d="M14 23L23 14L32 23L23 32L14 23Z" stroke="#00d4ff" stroke-width="2.2" stroke-linejoin="round"/>
        <circle cx="23" cy="23" r="4" fill="#7b61ff"/>
        <path d="M23 8V14M23 32V38M8 23H14M32 23H38" stroke="rgba(0, 212, 255, 0.7)" stroke-width="1.5" stroke-linecap="round"/>
        <defs>
            <linearGradient id="fiq_bg" x1="0" y1="0" x2="46" y2="46" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00d4ff"/>
                <stop offset="1" stop-color="#7b61ff"/>
            </linearGradient>
            <linearGradient id="fiq_border" x1="0" y1="0" x2="46" y2="46" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00d4ff"/>
                <stop offset="1" stop-color="#7b61ff" stop-opacity="0.3"/>
            </linearGradient>
        </defs>
    </svg>
    <div>
        <div class="fiq-title">FreightIQ</div>
        <div class="fiq-subtitle">Freight Carrier Lookup & Operations Router</div>
    </div>
</div>
<div class="fiq-telemetry">
    <span class="telemetry-pill">500 Carrier Profiles</span>
    <span class="telemetry-pill">SQLite (WAL)</span>
    <span class="telemetry-pill">FTS5 BM25 + ChromaDB</span>
    <span class="telemetry-pill">Cross-Encoder Re-ranking</span>
</div>
""", unsafe_allow_html=True)

TOOL_META = {
    "carrier_sql_query": {
        "label": "SQL DATABASE",
        "icon": "🗄️",
        "class": "tool-sql",
        "accent": "#0084ff",
        "desc": "Executing deterministic relational query on SQLite (WAL)"
    },
    "carrier_semantic_search": {
        "label": "HYBRID SEARCH",
        "icon": "🔍",
        "class": "tool-semantic",
        "accent": "#b347ff",
        "desc": "Fusing FTS5 BM25 + dense ChromaDB via RRF (k=60) & Cross-Encoder"
    },
    "check_fmcsa_authority": {
        "label": "FMCSA SAFER",
        "icon": "🛡️",
        "class": "tool-fmcsa",
        "accent": "#00e676",
        "desc": "Verifying USDOT safety compliance & operating authority"
    },
    "freight_class_calculator": {
        "label": "NMFC CALCULATOR",
        "icon": "🔢",
        "class": "tool-calc",
        "accent": "#ff9100",
        "desc": "Calculating volume, density, and exception class tiers"
    },
    "web_search": {
        "label": "WEB MARKET API",
        "icon": "🌐",
        "class": "tool-web",
        "accent": "#00e5ff",
        "desc": "Querying spot rate benchmarks & freight corridor intelligence"
    }
}

# Dynamic checks for reranker model configuration & feedback logs size
cross_encoder_name = getattr(config, "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2").split("/")[-1]
reranker_status = f"Cross-Encoder ({cross_encoder_name})"
reranker_class = "status-ok"

feedback_data = load_feedback()
feedback_count = len(feedback_data)

semantic_search_desc = f"FTS5 BM25 + ChromaDB + {reranker_status}"

TOOLS = [
    ("🔍", "Hybrid Search", semantic_search_desc),
    ("🗄️", "SQL Database", "Structured carrier lookups via SQLite"),
    ("🌐", "Web Search", "Live Tavily & DuckDuckGo market research"),
    ("🔢", "Freight Class", "NMFC density & exception calculator"),
    ("🛡️", "FMCSA Safety", "USDOT authority & insurance verification"),
]

with st.sidebar:
    # Custom Key & Model Configuration
    st.markdown('<div class="sidebar-section"><div class="sidebar-title">Model & Provider Configuration</div></div>', unsafe_allow_html=True)
    
    selected_provider_label = st.selectbox(
        "LLM Provider",
        ["Groq (Default)", "OpenAI", "Ollama / Local"],
        index=0,
        help="Select inference provider for this session."
    )
    provider_map = {
        "Groq (Default)": "groq",
        "OpenAI": "openai",
        "Ollama / Local": "ollama"
    }
    current_provider = provider_map[selected_provider_label]
    st.session_state["session_provider"] = current_provider

    if current_provider == "groq":
        model_options = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]
        key_label = "Groq API Key (Optional)"
        default_key_exists = bool(os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY)
    elif current_provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o"]
        key_label = "OpenAI API Key"
        default_key_exists = bool(os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None))
    else:
        model_options = ["qwen2.5:14b", "llama3.2:latest"]
        key_label = "API Key (Optional for Local)"
        default_key_exists = True

    user_api_key = st.text_input(
        key_label,
        type="password",
        value=st.session_state.get(f"custom_key_{current_provider}", ""),
        help="Session-specific API key."
    )
    if user_api_key != st.session_state.get(f"custom_key_{current_provider}"):
        st.session_state[f"custom_key_{current_provider}"] = user_api_key

    selected_model = st.selectbox(
        "Model",
        model_options,
        index=0,
        help="Select active model for this session."
    )
    st.session_state["session_model"] = selected_model

    has_active_key = default_key_exists or bool(user_api_key)
    status_class = "status-ok" if has_active_key else "status-err"
    status_text = f"{selected_model} · Connected" if has_active_key else f"{selected_provider_label.split()[0]} Key Missing"

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
        <div class="sidebar-title">System Status</div>
        <div style="margin-bottom: 8px;"><span class="status-badge {status_class}">Model: {status_text}</span></div>
        <div style="margin-bottom: 8px;"><span class="status-badge {reranker_class}">Re-ranker: {reranker_status}</span></div>
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

    if not has_active_key:
        st.warning(f"Provide an API key for {selected_provider_label} above or set it in your `.env` to enable the agent.")

    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.voted_message_index = -1
        st.rerun()

for idx, message in enumerate(st.session_state.messages):
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(format_message_content(message.content))
            
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            tool_cards = message.additional_kwargs.get("tool_executions", [])
            for tool_call in tool_cards:
                t_name = tool_call.get("name", "")
                t_meta = TOOL_META.get(t_name, {
                    "label": "TOOL",
                    "icon": "🔧",
                    "class": "tool-default"
                })
                card_class = tool_call.get("class", t_meta["class"])
                badge_label = tool_call.get("label", t_meta["label"])
                tool_icon = tool_call.get("icon", t_meta["icon"])
                st.markdown(f"""
                <div class="tool-card {card_class}">
                    <div class="tool-card-header">
                        <span class="tool-badge">{badge_label}</span>
                        <span class="tool-name">{tool_icon} {t_name}</span>
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
        if st.button("🚛 FL Produce & Freight Class", key="chip_fl", use_container_width=True):
            clicked_query = "Find a carrier located in Florida (FL) that handles fresh produce. What are their DOT and MC numbers, and how many years have they been operating? Also, what is the freight class for a 220 lbs crate of fresh produce measuring 36x36x36 inches? Be detailed."
    with col2:
        if st.button("🛡️ FMCSA Authority Verification", key="chip_fmcsa", use_container_width=True):
            clicked_query = "Verify the FMCSA operating authority, active insurance, and safety rating for USDOT 2942444."
    with col3:
        if st.button("🌐 Reefer Rates & GA Capacity", key="chip_market", use_container_width=True):
            clicked_query = "What are current refrigerated spot freight rates from Atlanta to Chicago, and do we have active refrigerated carriers in Georgia in our database?"
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
    logger.info(f"User query #{st.session_state.query_count}: '{user_query[:80]}'")

    with st.chat_message("assistant"):
        step_container = st.container()
        response_container = st.empty()
        turn_tool_cards = []

        try:
            # Sliding window: only send the last N messages to the LLM (turn-aware to avoid sequence errors)
            windowed_messages = get_windowed_messages(st.session_state.messages, config.CONVERSATION_WINDOW)

            # Retrieve compiled cached graph resource
            graph = get_graph()

            # Instantiate streaming callback handler
            stream_handler = StreamlitTokenCallbackHandler(response_container)
            accumulated_responses = []
            final_answer = ""

            session_provider = st.session_state.get("session_provider", "groq")
            session_config = {
                "callbacks": [stream_handler],
                "recursion_limit": 10,
                "configurable": {
                    "provider": session_provider,
                    "model": st.session_state.get("session_model", "qwen/qwen3.8-27b"),
                    "api_key": st.session_state.get(f"custom_key_{session_provider}") or None
                }
            }

            with st.status("Executing query...", expanded=True) as status_box:
                for event in graph.stream(
                    {"messages": windowed_messages}, 
                    config=session_config, 
                    stream_mode="updates"
                ):
                    for node_name, node_output in event.items():
                        if node_name == "tools":
                            for msg in node_output.get("messages", []):
                                safe_name = escape(str(msg.name))
                                t_meta = TOOL_META.get(msg.name, {
                                    "label": "TOOL",
                                    "icon": "🔧",
                                    "class": "tool-default",
                                    "accent": "#00d4ff",
                                    "desc": "Executing tool action"
                                })
                                status_box.write(f"{t_meta['icon']} **{t_meta['label']}**: {t_meta['desc']}")
                                
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
                                turn_tool_cards.append({
                                    "name": safe_name,
                                    "label": t_meta["label"],
                                    "icon": t_meta["icon"],
                                    "class": t_meta["class"],
                                    "output": safe_output
                                })

                                with step_container:
                                    st.markdown(f"""
                                    <div class="tool-card {t_meta['class']}">
                                        <div class="tool-card-header">
                                            <span class="tool-badge">{t_meta['label']}</span>
                                            <span class="tool-name">{t_meta['icon']} {safe_name}</span>
                                        </div>
                                        <div class="tool-output">{safe_output}</div>
                                    </div>""", unsafe_allow_html=True)

                        elif node_name == "agent":
                            messages = node_output.get("messages", [])
                            if messages and messages[-1].content:
                                final_answer = format_message_content(messages[-1].content)
                                # Force final write to make sure text formatting is clean
                                response_container.write(final_answer)

                status_box.update(label="✓ Complete", state="complete", expanded=False)

            if final_answer:
                st.session_state.messages.append(AIMessage(
                    content=final_answer,
                    additional_kwargs={"tool_executions": turn_tool_cards}
                ))
                logger.info(f"Agent response complete. Session total: {st.session_state.query_count} queries.")
                st.rerun()

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            err_msg = f"Agent error: {str(e)}"
            st.error(err_msg)
            st.session_state.messages.append(AIMessage(
                content=err_msg,
                additional_kwargs={"tool_executions": turn_tool_cards}
            ))

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
            st.caption("Was this response helpful?")
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 10])
            with fb_col1:
                if st.button("👍 Yes", key="thumbs_up", use_container_width=True):
                    save_feedback(last_query, last_response, "up")
                    st.session_state.voted_message_index = last_msg_idx
                    st.toast("Feedback saved.")
                    st.rerun()
            with fb_col2:
                if st.button("👎 No", key="thumbs_down", use_container_width=True):
                    save_feedback(last_query, last_response, "down")
                    st.session_state.voted_message_index = last_msg_idx
                    st.toast("Feedback saved.")
                    st.rerun()
        else:
            st.success("Feedback saved.")

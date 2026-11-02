from dotenv import load_dotenv

load_dotenv()

import logging
import os
from html import escape

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

import config
from agent.graph import build_graph
from agent.nodes import get_windowed_messages
from rag.utils import format_message_content, load_feedback, save_feedback
from utils.locks import setup_lock

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
            if chunk:
                if hasattr(chunk, "message") and hasattr(chunk.message, "tool_call_chunks") and chunk.message.tool_call_chunks:
                    return
                if (hasattr(chunk, "message") and hasattr(chunk.message, "additional_kwargs")
                        and "tool_calls" in chunk.message.additional_kwargs):
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

# UI styling lives in assets/styles.css (kept out of the Python frontend file)
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "styles.css")
with open(_css_path, "r", encoding="utf-8") as _css_file:
    st.markdown(f"<style>\n{_css_file.read()}\n</style>", unsafe_allow_html=True)

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
        "desc": "Executing deterministic relational query on SQLite (WAL)"
    },
    "carrier_semantic_search": {
        "label": "HYBRID SEARCH",
        "icon": "🔍",
        "class": "tool-semantic",
        "desc": "Fusing FTS5 BM25 + dense ChromaDB via RRF (k=60) & Cross-Encoder"
    },
    "check_fmcsa_authority": {
        "label": "FMCSA SAFER",
        "icon": "🛡️",
        "class": "tool-fmcsa",
        "desc": "Verifying USDOT safety compliance & operating authority"
    },
    "freight_class_calculator": {
        "label": "NMFC CALCULATOR",
        "icon": "🔢",
        "class": "tool-calc",
        "desc": "Calculating volume, density, and exception class tiers"
    },
    "web_search": {
        "label": "WEB MARKET API",
        "icon": "🌐",
        "class": "tool-web",
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
        model_options = list(dict.fromkeys([config.AGENT_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]))
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
            clicked_query = "Find a carrier located in Florida (FL) that handles fresh produce. What are their DOT and MC numbers, and how many years have they been operating? Also, what is the freight class for a 220 lbs crate of fresh produce measuring 36x36x36 inches?"
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
                    "model": st.session_state.get("session_model", config.AGENT_MODEL),
                    "api_key": st.session_state.get(f"custom_key_{session_provider}") or None,
                    "session_state": st.session_state
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
                                    "desc": "Executing tool action"
                                })
                                status_box.write(f"{t_meta['icon']} **{t_meta['label']}**: {t_meta['desc']}")
                                
                                # Newline-preserving truncation to maintain table and bullet formatting
                                raw_output = str(msg.content)
                                if len(raw_output) > config.TOOL_TRUNCATION_LIMIT:
                                    truncated_output = raw_output[:config.TOOL_TRUNCATION_LIMIT] + "\n\n... [Output truncated to stay within UI display limit]"
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
            err_msg = f"Agent error: {e!s}"
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

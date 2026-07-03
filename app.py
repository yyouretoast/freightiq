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
    page_title="FreightIQ | Carrier Intelligence Agent",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stTextInput input {
        background-color: #1a1d2e !important;
        color: #ffffff !important;
        border-color: #3b3f5c !important;
    }
    .stChatMessage {
        background-color: #141724;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #1f233b;
    }
    .stChatMessage.user {
        background-color: #1a1d2e;
        border: 1px solid #2e3456;
    }
    .tool-box {
        background-color: #121f2d;
        border-left: 5px solid #00d2ff;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚚 FreightIQ: Carrier Intelligence Agent")
st.markdown(
    "A multi-agent logistics assistant built using **LangGraph**, **LangChain**, and **ChromaDB**, "
    "powered by **Gemini 2.0 Flash** and custom **PyTorch** and **SQL** retrieval engines."
)

with st.sidebar:
    st.header("⚙️ Agent Status")
    
    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini API Key: Configured")
    else:
        st.error("Gemini API Key: Missing")
        st.info("Please set GEMINI_API_KEY in your .env file to enable the agent.")
        
    st.markdown("---")
    st.markdown(
        "### Available Tools:\n"
        "- 🔍 **Carrier Semantic Search**: Vector DB semantic search + custom PyTorch MLP re-ranker.\n"
        "- 🗄️ **Carrier SQL Database**: Direct SQLite queries for structured lookups.\n"
        "- 🌐 **Web Search**: Real-time DuckDuckGo carrier & market research.\n"
        "- 🔢 **NMFC Freight Class Calculator**: Precise shipment density calculator."
    )
    
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(format_message_content(message.content))
    elif isinstance(message, AIMessage):
        if message.content:
            with st.chat_message("assistant"):
                st.write(format_message_content(message.content))

if user_query := st.chat_input("Ask about carrier search, rates, or calculate freight class..."):
    with st.chat_message("user"):
        st.write(user_query)
        
    st.session_state.messages.append(HumanMessage(content=user_query))
    
    with st.chat_message("assistant"):
        step_container = st.container()
        response_container = st.empty()
        
        final_answer = ""
        try:
            with st.spinner("Agent is reasoning and querying database..."):
                for event in graph.stream({"messages": st.session_state.messages}, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "tools":
                            for msg in node_output.get("messages", []):
                                safe_result = escape(str(msg.content)[:300])
                                with step_container:
                                    st.markdown(
                                        f'<div class="tool-box">'
                                        f'<strong>🔧 Tool Executed:</strong> {escape(msg.name)}<br/>'
                                        f'<strong>Output Sample:</strong><br/>{safe_result}...'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                        elif node_name == "agent":
                            messages = node_output.get("messages", [])
                            if messages and messages[-1].content:
                                final_answer = format_message_content(messages[-1].content)
                                response_container.write(final_answer)
                                
            if final_answer:
                st.session_state.messages.append(AIMessage(content=final_answer))
                
        except Exception as e:
            st.error(f"Error during agent execution: {str(e)}")

import streamlit as tf_streamlit
import os
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

tf_streamlit.set_page_config(
    page_title="FreightIQ | Carrier Intelligence Agent",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

tf_streamlit.markdown(
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

tf_streamlit.title("🚚 FreightIQ: Carrier Intelligence Agent")
tf_streamlit.markdown(
    "A multi-agent logistics assistant built using **LangGraph**, **LangChain**, and **ChromaDB**, "
    "powered by **Gemini 2.0 Flash** and custom **PyTorch** and **SQL** retrieval engines."
)

with tf_streamlit.sidebar:
    tf_streamlit.header("⚙️ Agent Status")
    
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    if api_key_set:
        tf_streamlit.success("Gemini API Key: Configured")
    else:
        tf_streamlit.error("Gemini API Key: Missing")
        tf_streamlit.info("Please set GEMINI_API_KEY in your .env file to enable the agent.")
        
    tf_streamlit.markdown("---")
    tf_streamlit.markdown(
        "### Available Tools:\n"
        "- 🔍 **Carrier Semantic Search**: Vector DB semantic search + custom PyTorch MLP re-ranker.\n"
        "- 🗄️ **Carrier SQL Database**: Direct SQLite queries for structured lookups.\n"
        "- 🌐 **Web Search**: Real-time DuckDuckGo carrier & market research.\n"
        "- 🔢 **NMFC Freight Class Calculator**: Precise shipment density calculator."
    )
    
    if tf_streamlit.button("Reset Conversation", use_container_width=True):
        tf_streamlit.session_state.messages = []
        tf_streamlit.rerun()

if "messages" not in tf_streamlit.session_state:
    tf_streamlit.session_state.messages = []

for message in tf_streamlit.session_state.messages:
    if isinstance(message, HumanMessage):
        with tf_streamlit.chat_message("user"):
            tf_streamlit.write(format_message_content(message.content))
    elif isinstance(message, AIMessage):
        if message.content:
            with tf_streamlit.chat_message("assistant"):
                tf_streamlit.write(format_message_content(message.content))

if user_query := tf_streamlit.chat_input("Ask about carrier search, rates, or calculate freight class..."):
    with tf_streamlit.chat_message("user"):
        tf_streamlit.write(user_query)
        
    human_msg = HumanMessage(content=user_query)
    tf_streamlit.session_state.messages.append(human_msg)
    
    with tf_streamlit.chat_message("assistant"):
        step_container = tf_streamlit.container()
        response_container = tf_streamlit.empty()
        
        final_answer = ""
        try:
            with tf_streamlit.spinner("Agent is reasoning and querying database..."):
                for event in graph.stream({"messages": tf_streamlit.session_state.messages}, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "tools":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                tool_name = msg.name
                                tool_result = msg.content
                                
                                with step_container:
                                    tf_streamlit.markdown(
                                        f"""<div class="tool-box">
                                        <strong>🔧 Tool Executed:</strong> {tool_name}<br/>
                                        <strong>Output Sample:</strong><br/>
                                        {tool_result[:300]}...
                                        </div>""", 
                                        unsafe_allow_html=True
                                    )
                                    
                        elif node_name == "agent":
                            messages = node_output.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if last_msg.content:
                                    final_answer = format_message_content(last_msg.content)
                                    response_container.write(final_answer)
                                
            if final_answer:
                tf_streamlit.session_state.messages.append(AIMessage(content=final_answer))
                
        except Exception as e:
            tf_streamlit.error(f"Error during agent execution: {str(e)}")

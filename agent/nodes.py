import logging
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from agent.state import AgentState
from agent.tools import tools

logger = logging.getLogger(__name__)

# System prompt defining agent instructions, tools, capabilities, and guidance
SYSTEM_PROMPT = """You are FreightIQ, a senior logistics coordinator and agentic assistant. Your task is to resolve user queries about carriers, shipping options, freight classes, and real-time market trends.

You have access to the following specialized tools:
1. `carrier_sql_query`: Best for exact lookups, filter matching (hq_state, safety_rating, dot_number, mc_number, years_operating), or aggregate math (averages, counts). You must query the 'carriers' table.
2. `carrier_semantic_search`: Best for qualitative queries, regional match searches, or general capability matching (e.g. "haulers specializing in heavy cargo in the Midwest").
3. `web_search`: Best for querying real-time market spot rates, logistics industry news, and active shipping rates.
4. `freight_class_calculator`: Best for calculating shipment density (lbs/cu ft) and mapping it to the appropriate NMFC freight class.

Guidelines for Tool Selection:
- If a query calls for exact attributes (e.g., specific states, safety ratings, or specific MC/DOT numbers), prefer using `carrier_sql_query`.
- If a query describes a carrier's qualitative specialization or primary service lanes in natural language, prefer `carrier_semantic_search`.
- For current market rate trends or news, use `web_search`.
- Be concise and structure your responses with markdown tables or bullet points where appropriate to showcase readability.

CRITICAL TOOL CALL RULE:
- Do not generate duplicate, redundant, or identical tool calls in a single turn. If you need to query the database, generate exactly ONE tool call.
"""

# Primary LLM: Groq Llama-3.3-70b
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    logger.info(f"Agent invoked with {len(state['messages'])} messages in context.")
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}

tool_node = ToolNode(tools)

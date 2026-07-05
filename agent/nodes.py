import logging
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage
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

Formatting Rule:
- Always list the actual names and details of the carriers returned by the tools. Do not output placeholders, disclaimers, or generic templates.
"""

# Primary LLM: Groq Llama-3.1-8b (Highly efficient, large token quota to prevent TPD 429 limits)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    streaming=True
)

# Bind tools to the LLM and strictly disable parallel tool calls at the API schema level.
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

def agent_node(state: AgentState):
    logger.info(f"Agent invoked with {len(state['messages'])} messages in context.")
    messages = state["messages"]
    
    # Global Loop Guardrail:
    # Detect duplicate back-to-back tool calls
    if len(messages) >= 2:
        prev_ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        if len(prev_ai_msgs) >= 2:
            last_call = prev_ai_msgs[-1].tool_calls[0]
            penultimate_call = prev_ai_msgs[-2].tool_calls[0]
            
            # If the last two tool calls have identical names and arguments, inject a loop-break message
            if last_call["name"] == penultimate_call["name"] and last_call["args"] == penultimate_call["args"]:
                logger.warning(f"Loop detected on tool '{last_call['name']}'. Injecting loop guardrail warning.")
                loop_break_prompt = (
                    f"System Warning: You have already executed the tool '{last_call['name']}' with args {last_call['args']}. "
                    "Do NOT call this tool again. Synthesize your final answer immediately based on the results already retrieved."
                )
                messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + messages + [SystemMessage(content=loop_break_prompt)]
                response = llm_with_tools.invoke(messages_with_warning)
                return {"messages": [response]}
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}

tool_node = ToolNode(tools)

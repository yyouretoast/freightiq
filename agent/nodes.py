import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, APIStatusError
from agent.state import AgentState
from agent.tools import tools
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FreightIQ, a senior logistics coordinator and agentic assistant. Your task is to resolve user queries about carriers, shipping options, freight classes, and real-time market trends.

CRITICAL — DATA INTEGRITY RULE:
Never fabricate carrier names, DOT numbers, MC numbers, safety ratings, or any carrier data. You do NOT have carrier knowledge in your training data. You MUST call a tool every time carrier information is requested. If no results are returned, state so honestly.

Tool Selection Guidelines:
- If the query mentions ANY US state (e.g. FL, OH, TX) or region (e.g. Midwest, Southwest), ALWAYS use `carrier_sql_query`. For regions: EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest'). For state: hq_state = 'FL'.
- If the query mentions equipment (flatbed, reefer, dry van) or cargo type (hazmat, produce, pharmaceuticals), use `carrier_sql_query` with json_each() on equipment_types and cargo_specializations.
- If exact attributes (safety ratings, DOT/MC numbers, years operating) are requested, use `carrier_sql_query`.
- Use `carrier_semantic_search` ONLY for qualitative queries with no geographic or structured filters.
- For current market rate trends or news, use `web_search`.
- For NMFC density and freight class calculations, use `freight_class_calculator`.

Formatting & Synthesis:
- List actual carrier names and details returned by tools.
- Synthesize all parts of multi-part queries in your final response.
- When querying carriers, always select carrier_name or * so names are available to present.
"""

llm = ChatGroq(
    model=config.AGENT_MODEL,
    groq_api_key=config.GROQ_API_KEY,
    temperature=0.0,
    streaming=True
)

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True
)
def _invoke_with_retry(model_obj, messages):
    return model_obj.invoke(messages)

def agent_node(state: AgentState):
    logger.info(f"Agent invoked with {len(state['messages'])} messages in context.")
    messages = state["messages"]
    
    if len(messages) >= 2:
        prev_ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        if len(prev_ai_msgs) >= 2:
            last_calls = prev_ai_msgs[-1].tool_calls
            penultimate_calls = prev_ai_msgs[-2].tool_calls
            
            if last_calls and penultimate_calls:
                last_call = last_calls[0]
                penultimate_call = penultimate_calls[0]
                
                if last_call["name"] == penultimate_call["name"] and last_call["args"] == penultimate_call["args"]:
                    logger.warning(f"Loop detected on tool '{last_call['name']}'. Injecting loop guardrail.")
                    loop_break_prompt = (
                        f"System Warning: You have already executed '{last_call['name']}' with args {last_call['args']}. "
                        "Do NOT call this tool again. Synthesize your final answer immediately based on the results already retrieved in plain text."
                    )
                    messages_with_warning = [SystemMessage(content="You are FreightIQ. Respond directly in plain text. Do not output tool calls or JSON markup.")] + messages + [SystemMessage(content=loop_break_prompt)]
                    response = _invoke_with_retry(llm, messages_with_warning)
                    return {"messages": [response]}

        sql_calls = 0
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.tool_calls:
                if any(tc["name"] == "carrier_sql_query" for tc in m.tool_calls):
                    sql_calls += 1
                else:
                    break
            elif isinstance(m, HumanMessage):
                break
        
        if sql_calls >= 4:
            logger.warning(f"Excessive SQL queries detected ({sql_calls}). Injecting SQL loop guardrail.")
            loop_break_prompt = (
                "System Warning: You have executed carrier_sql_query multiple times. "
                "If no matching records exist, synthesize your final answer now in plain text stating that no matching carriers were found."
            )
            messages_with_warning = [SystemMessage(content="You are FreightIQ. Respond directly in plain text. Do not output tool calls or JSON markup.")] + messages + [SystemMessage(content=loop_break_prompt)]
            response = _invoke_with_retry(llm, messages_with_warning)
            return {"messages": [response]}
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = _invoke_with_retry(llm_with_tools, messages_with_system)
    return {"messages": [response]}

tool_node = ToolNode(tools)

import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, InternalServerError, APIConnectionError, NotFoundError
from agent.state import AgentState
from agent.tools import tools
import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a freight carrier research assistant. Answer logistics, carrier lookup, freight class, and market rate queries using the provided tools.

Rules:
1. Grounding: All carrier facts (names, DOT/MC numbers, safety ratings, equipment, locations) must come strictly from tool results. If no records match, state that directly. Never invent carrier data.
2. Tool Routing:
   - Structured carrier queries (by state, region, equipment, cargo, safety rating, DOT/MC): use `carrier_sql_query`.
   - Qualitative carrier descriptions (reputation, service quality, specialized handling): use `carrier_semantic_search`. Do not follow up with SQL queries unless structured filtering is explicitly requested.
   - Live market rates and industry news: use `web_search`.
   - NMFC density and freight class lookups: use `freight_class_calculator`.
3. Single Tool Principle: Select the single most appropriate tool for the inquiry. Synthesize and present the final answer immediately once results are returned from that tool; do not chain or invoke secondary tools unless the user explicitly requested multiple distinct lookups.
4. Presentation: Format carrier results cleanly using markdown tables or bullet points with key attributes (Name, DOT/MC, HQ, Equipment, Safety). For multi-part queries, address every component directly.
"""

llm = ChatGroq(
    model=config.AGENT_MODEL,
    groq_api_key=config.GROQ_API_KEY,
    temperature=0.0,
    max_tokens=config.MAX_OUTPUT_TOKENS,
    streaming=True
)

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

@retry(
    retry=retry_if_exception_type((RateLimitError, InternalServerError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True
)
def _invoke_with_retry(model_obj, messages):
    try:
        return model_obj.invoke(messages)
    except NotFoundError as e:
        logger.warning(f"Configured model failed with 404 ({e}). Falling back to 'qwen/qwen3.8-27b'.")
        fallback_llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            groq_api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            streaming=True
        )
        is_tool_bound = hasattr(model_obj, "tools") or "bind_tools" in str(type(model_obj)) or hasattr(model_obj, "bound")
        fallback_target = fallback_llm.bind_tools(tools, parallel_tool_calls=False) if is_tool_bound else fallback_llm
        return fallback_target.invoke(messages)

def agent_node(state: AgentState):
    logger.info(f"Agent invoked with {len(state['messages'])} messages in context.")
    messages = state["messages"]
    
    # Scope loop detection to the current user turn to prevent false alarms across multi-turn sessions
    last_human_idx = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    current_turn_msgs = messages[last_human_idx + 1:] if last_human_idx >= 0 else messages
    
    if len(current_turn_msgs) >= 2:
        prev_ai_msgs = [m for m in current_turn_msgs if isinstance(m, AIMessage) and m.tool_calls]
        if len(prev_ai_msgs) >= 2:
            last_calls = prev_ai_msgs[-1].tool_calls
            penultimate_calls = prev_ai_msgs[-2].tool_calls
            
            if last_calls and penultimate_calls:
                last_call = last_calls[0]
                penultimate_call = penultimate_calls[0]
                
                if last_call["name"] == penultimate_call["name"] and last_call["args"] == penultimate_call["args"]:
                    logger.warning(f"Loop detected on tool '{last_call['name']}'. Injecting loop guardrail.")
                    loop_break_directive = (
                        f"Repeat tool call detected for '{last_call['name']}'. "
                        "Do not invoke this tool again. Synthesize your final answer directly in plain text using the results already retrieved."
                    )
                    messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + messages + [HumanMessage(content=loop_break_directive)]
                    response = _invoke_with_retry(llm, messages_with_warning)
                    return {"messages": [response]}

        # Check for excessive consecutive calls to ANY single tool (e.g. 3 consecutive calls)
        consecutive_tool_count = 0
        last_tool_name = None
        for m in reversed(current_turn_msgs):
            if isinstance(m, AIMessage) and m.tool_calls:
                tname = m.tool_calls[0]["name"]
                if last_tool_name is None:
                    last_tool_name = tname
                    consecutive_tool_count = 1
                elif last_tool_name == tname:
                    consecutive_tool_count += 1
                else:
                    break
            elif isinstance(m, HumanMessage):
                break
        
        if consecutive_tool_count >= 3:
            logger.warning(f"Excessive repeated calls detected for tool '{last_tool_name}' ({consecutive_tool_count}). Injecting loop guardrail.")
            loop_break_directive = (
                f"Multiple repeated calls executed for tool '{last_tool_name}'. "
                "Do not invoke any tools again. Synthesize your final answer now in plain text using the results retrieved so far, or state that data is unavailable."
            )
            messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + messages + [HumanMessage(content=loop_break_directive)]
            response = _invoke_with_retry(llm, messages_with_warning)
            return {"messages": [response]}
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = _invoke_with_retry(llm_with_tools, messages_with_system)
    return {"messages": [response]}

tool_node = ToolNode(tools)

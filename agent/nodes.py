import time
import logging
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError, InternalServerError, APIConnectionError, NotFoundError, APIStatusError
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
   - USDOT safety compliance, operating authority, and FMCSA insurance checks: use `check_fmcsa_authority`.
3. Single Tool Principle: Select the single most appropriate tool for the inquiry. Synthesize and present the final answer immediately once results are returned from that tool; do not chain or invoke secondary tools unless the user explicitly requested multiple distinct lookups.
4. Presentation: Format carrier results cleanly using markdown tables or bullet points with key attributes (Name, DOT/MC, HQ, Equipment, Safety). For multi-part queries, address every component directly.
"""

_active_llm = None
_active_llm_with_tools = None

def get_active_models():
    global _active_llm, _active_llm_with_tools
    if _active_llm is None:
        _active_llm = ChatGroq(
            model=config.AGENT_MODEL,
            groq_api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            streaming=True
        )
        _active_llm_with_tools = _active_llm.bind_tools(tools, parallel_tool_calls=False)
    return _active_llm, _active_llm_with_tools

def switch_to_sibling():
    global _active_llm, _active_llm_with_tools
    current = getattr(_active_llm, "model_name", "") or getattr(config, "AGENT_MODEL", "")
    alt_model = "qwen/qwen3.6-27b" if "3.8" in current else "qwen/qwen3.8-27b"
    logger.warning(f"Switching active inference model to '{alt_model}'.")
    _active_llm = ChatGroq(
        model=alt_model,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        streaming=True
    )
    _active_llm_with_tools = _active_llm.bind_tools(tools, parallel_tool_calls=False)
    return _active_llm, _active_llm_with_tools

llm, llm_with_tools = get_active_models()

def _is_retryable_error(exc):
    if isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) in (413, 429, 500, 502, 503, 504):
        return True
    return False

@retry(
    retry=_is_retryable_error,
    wait=wait_exponential(multiplier=2, min=3, max=45),
    stop=stop_after_attempt(4),
    reraise=True
)
def _invoke_with_retry(is_tool_bound, messages):
    base_llm, tool_llm = get_active_models()
    target = tool_llm if is_tool_bound else base_llm
    try:
        return target.invoke(messages)
    except (NotFoundError, RateLimitError) as e:
        logger.warning(f"Model failed with {type(e).__name__} ({e}). Switching to sibling model.")
        base_alt, tool_alt = switch_to_sibling()
        target_alt = tool_alt if is_tool_bound else base_alt
        return target_alt.invoke(messages)



def _prepare_context_messages(messages):
    """
    Enforces context truncation to the last 8 messages and bounds individual tool outputs
    to stay within model token quotas and avoid 413 Payload Too Large errors.
    """
    truncated = messages[-8:] if len(messages) > 8 else list(messages)
    cleaned = []
    for m in truncated:
        if isinstance(m, ToolMessage) and len(str(m.content)) > 2000:
            bounded_text = str(m.content)[:2000] + "\n\n... [Output truncated to stay within model token quota]"
            cleaned.append(ToolMessage(content=bounded_text, tool_call_id=m.tool_call_id, name=m.name))
        else:
            cleaned.append(m)
    return cleaned


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
                    messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages) + [HumanMessage(content=loop_break_directive)]
                    response = _invoke_with_retry(False, messages_with_warning)
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
            messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages) + [HumanMessage(content=loop_break_directive)]
            response = _invoke_with_retry(False, messages_with_warning)
            return {"messages": [response]}
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages)
    response = _invoke_with_retry(True, messages_with_system)
    return {"messages": [response]}



tool_node = ToolNode(tools)

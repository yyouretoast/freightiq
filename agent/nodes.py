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
4. Presentation: Format carrier results cleanly using markdown tables or bullet points with key attributes (Name, DOT/MC, HQ, Equipment, Safety). For multi-part queries, address every component directly and concisely without repeating raw tool dumps verbatim so answers complete cleanly within token limits.
"""
import threading
from typing import Optional
from langchain_core.runnables import RunnableConfig
from agent.models import create_model_instance

_model_lock = threading.Lock()
_active_llm = None
_active_llm_with_tools = None

# Session-scoped cache for multi-user web sessions: (provider, model, key) -> (base_llm, tool_llm)
_session_cache = {}
_session_cache_lock = threading.Lock()

def reset_active_models():
    global _active_llm, _active_llm_with_tools
    with _model_lock:
        _active_llm = None
        _active_llm_with_tools = None
    with _session_cache_lock:
        _session_cache.clear()

def get_active_models(configurable: Optional[dict] = None):
    global _active_llm, _active_llm_with_tools
    
    # 1. If session-level configuration is passed, use session-scoped instance to prevent crosstalk
    if configurable and any(k in configurable for k in ("provider", "model", "api_key", "base_url")):
        provider = configurable.get("provider") or getattr(config, "LLM_PROVIDER", "groq")
        model = configurable.get("model") or getattr(config, "AGENT_MODEL", "qwen/qwen3.8-27b")
        api_key = configurable.get("api_key")
        base_url = configurable.get("base_url")
        cache_key = (provider, model, api_key, base_url)
        
        with _session_cache_lock:
            if cache_key in _session_cache:
                return _session_cache[cache_key]
            
            base_m, tool_m = create_model_instance(
                provider=provider,
                model_name=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                streaming=True
            )
            _session_cache[cache_key] = (base_m, tool_m)
            return base_m, tool_m

    # 2. Process-global default instance
    if _active_llm is None:
        with _model_lock:
            if _active_llm is None:
                provider = getattr(config, "LLM_PROVIDER", "groq")
                model = getattr(config, "AGENT_MODEL", "qwen/qwen3.8-27b")
                _active_llm, _active_llm_with_tools = create_model_instance(
                    provider=provider,
                    model_name=model,
                    temperature=0.0,
                    max_tokens=config.MAX_OUTPUT_TOKENS,
                    streaming=True
                )
    return _active_llm, _active_llm_with_tools

def switch_to_sibling(current_model_name: str = ""):
    global _active_llm, _active_llm_with_tools
    with _model_lock:
        provider = getattr(config, "LLM_PROVIDER", "groq")
        if provider == "groq":
            current = current_model_name or getattr(_active_llm, "model_name", "") or getattr(config, "AGENT_MODEL", "")
            alt_model = "qwen/qwen3.6-27b" if "3.8" in current else "qwen/qwen3.8-27b"
            logger.warning(f"Switching active inference model to '{alt_model}'.")
            _active_llm, _active_llm_with_tools = create_model_instance(
                provider="groq",
                model_name=alt_model,
                temperature=0.0,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                streaming=True
            )
        elif provider in ("gemini", "google"):
            current = current_model_name or "gemini-2.5-flash"
            alt_model = "gemini-1.5-flash" if "2.5" in current else "gemini-2.5-flash"
            logger.warning(f"Switching active inference model to '{alt_model}'.")
            _active_llm, _active_llm_with_tools = create_model_instance(
                provider="gemini",
                model_name=alt_model,
                temperature=0.0,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                streaming=True
            )
        return _active_llm, _active_llm_with_tools

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
def _invoke_with_retry(is_tool_bound, messages, configurable=None):
    base_llm, tool_llm = get_active_models(configurable)
    target = tool_llm if is_tool_bound else base_llm
    try:
        return target.invoke(messages)
    except (NotFoundError, RateLimitError) as e:
        logger.warning(f"Model failed with {type(e).__name__} ({e}). Switching to sibling model.")
        base_alt, tool_alt = switch_to_sibling(getattr(target, "model_name", ""))
        target_alt = tool_alt if is_tool_bound else base_alt
        return target_alt.invoke(messages)



def get_windowed_messages(messages, max_messages=None):
    """
    Returns a sliding window of conversation messages aligned to a user turn (HumanMessage)
    without splitting tool calls from their corresponding tool outputs.
    """
    max_msgs = max_messages or getattr(config, "CONVERSATION_WINDOW", 8)
    if len(messages) <= max_msgs:
        return list(messages)
    slice_idx = -max_msgs
    while abs(slice_idx) < len(messages):
        first_msg = messages[slice_idx]
        if isinstance(first_msg, ToolMessage):
            slice_idx -= 1
        elif isinstance(first_msg, AIMessage) and getattr(first_msg, "tool_calls", None):
            slice_idx -= 1
        else:
            break
    while abs(slice_idx) < len(messages) and not isinstance(messages[slice_idx], HumanMessage):
        slice_idx -= 1
    return list(messages[slice_idx:])

def _prepare_context_messages(messages):
    """
    Enforces context truncation to config.CONVERSATION_WINDOW without splitting
    tool calls from tool messages, and bounds individual tool outputs to stay
    within model token quotas.
    """
    truncated = get_windowed_messages(messages, getattr(config, "CONVERSATION_WINDOW", 8))
    truncation_limit = getattr(config, "TOOL_TRUNCATION_LIMIT", 2000)
    cleaned = []
    for m in truncated:
        if isinstance(m, ToolMessage) and len(str(m.content)) > truncation_limit:
            bounded_text = str(m.content)[:truncation_limit] + "\n\n... [Output truncated to stay within model token quota]"
            cleaned.append(ToolMessage(content=bounded_text, tool_call_id=m.tool_call_id, name=m.name))
        else:
            cleaned.append(m)
    return cleaned


def agent_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info(f"Agent invoked with {len(state['messages'])} messages in context.")
    configurable = config.get("configurable", {}) if config else {}
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
                    response = _invoke_with_retry(False, messages_with_warning, configurable=configurable)
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
            response = _invoke_with_retry(False, messages_with_warning, configurable=configurable)
            return {"messages": [response]}
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages)
    response = _invoke_with_retry(True, messages_with_system, configurable=configurable)
    return {"messages": [response]}



tool_node = ToolNode(tools)

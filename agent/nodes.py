import logging
import threading
import time

from groq import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from agent.models import create_model_instance
from agent.state import AgentState
from agent.tools import tools

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
4. Presentation: Format carrier results using markdown tables or bullet points with key attributes (Name, DOT/MC, HQ, Equipment, Safety). For multi-part queries, address each component directly and concisely; do not output raw serialized tool dumps.
"""

_model_lock = threading.Lock()
_active_llm = None
_active_llm_with_tools = None
_last_failover_time = 0.0
FAILOVER_COOLDOWN_SECONDS = 300.0  # 5 minutes cooldown to restore primary model

# Session-scoped cache for multi-user web sessions: (provider, model, key) -> (base_llm, tool_llm)
_session_cache = {}
_session_cache_lock = threading.Lock()

def _create_session_models(provider, model, api_key=None, base_url=None):
    """Create a provider/model instance pair and register it in the session cache."""
    base_m, tool_m = create_model_instance(
        provider=provider,
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        streaming=True
    )
    with _session_cache_lock:
        _session_cache[(provider, model, api_key, base_url)] = (base_m, tool_m)
    return base_m, tool_m

def reset_active_models():
    global _active_llm, _active_llm_with_tools, _last_failover_time
    with _model_lock:
        _active_llm = None
        _active_llm_with_tools = None
        _last_failover_time = 0.0
    with _session_cache_lock:
        _session_cache.clear()

def get_active_models(configurable: dict | None = None):
    global _active_llm, _active_llm_with_tools
    
    # 1. If session-level configuration is passed, use session-scoped instance to prevent crosstalk
    if configurable and any(k in configurable for k in ("provider", "model", "api_key", "base_url")):
        provider = configurable.get("provider") or config.LLM_PROVIDER
        model = configurable.get("model") or config.AGENT_MODEL
        api_key = configurable.get("api_key")
        base_url = configurable.get("base_url")
        cache_key = (provider, model, api_key, base_url)
        
        with _session_cache_lock:
            if cache_key in _session_cache:
                return _session_cache[cache_key]
        
        return _create_session_models(provider, model, api_key=api_key, base_url=base_url)

    # 2. Process-global default instance
    global _active_llm, _active_llm_with_tools, _last_failover_time
    # Circuit breaker: if failover cooldown has elapsed, attempt to restore primary model
    if (_active_llm is not None and _last_failover_time > 0.0
            and time.time() - _last_failover_time > FAILOVER_COOLDOWN_SECONDS):
        with _model_lock:
            if _active_llm is not None and time.time() - _last_failover_time > FAILOVER_COOLDOWN_SECONDS:
                logger.info("Failover cooldown elapsed. Attempting to restore primary model.")
                _active_llm = None
                _active_llm_with_tools = None
                _last_failover_time = 0.0

    if _active_llm is None:
        with _model_lock:
            if _active_llm is None:
                provider = config.LLM_PROVIDER
                model = config.AGENT_MODEL
                _active_llm, _active_llm_with_tools = create_model_instance(
                    provider=provider,
                    model_name=model,
                    temperature=0.0,
                    max_tokens=config.MAX_OUTPUT_TOKENS,
                    streaming=True
                )
    return _active_llm, _active_llm_with_tools

SIBLING_MODEL_MAP = {
    # qwen/qwen3.6-27b does not exist on Groq (verified via GET /openai/v1/models);
    # the only comparable text sibling is openai/gpt-oss-120b.
    "groq": lambda curr: "openai/gpt-oss-120b" if "qwen" in curr else "qwen/qwen3.8-27b",
    "openai": lambda curr: "gpt-4o" if "mini" in curr else "gpt-4o-mini",
    "ollama": lambda curr: "llama3.2:latest" if "qwen" in curr else "qwen2.5:14b",
    "local": lambda curr: "llama3.2:latest" if "qwen" in curr else "qwen2.5:14b",
    "openai_compatible": lambda curr: "llama3.2:latest" if "qwen" in curr else "qwen2.5:14b",
}

def _extract_model_name(llm_obj) -> str:
    if llm_obj is None:
        return ""
    underlying = getattr(llm_obj, "bound", llm_obj)
    return (
        getattr(underlying, "model_name", None)
        or getattr(underlying, "model", None)
        or getattr(llm_obj, "model_name", None)
        or getattr(llm_obj, "model", None)
        or ""
    )

def switch_to_sibling(current_model_name: str = "", configurable: dict | None = None):
    global _active_llm, _active_llm_with_tools
    
    # 1. Session-scoped failover
    if configurable and any(k in configurable for k in ("provider", "model", "api_key", "base_url")):
        provider = (configurable.get("provider") or config.LLM_PROVIDER).lower()
        current = (
            current_model_name
            or configurable.get("model")
            or config.AGENT_MODEL
        )
        mapper = SIBLING_MODEL_MAP.get(provider, SIBLING_MODEL_MAP["groq"])
        alt_model = mapper(current)
        logger.warning(
            f"Switching session model from '{current}' to sibling '{alt_model}' (provider: {provider})."
        )
        
        configurable["model"] = alt_model
        if "session_state" in configurable and hasattr(configurable["session_state"], "__setitem__"):
            try:
                configurable["session_state"]["session_model"] = alt_model
            except Exception as sess_e:
                logger.debug(f"Session model state update skipped: {sess_e}")
        
        return _create_session_models(
            provider,
            alt_model,
            api_key=configurable.get("api_key"),
            base_url=configurable.get("base_url")
        )

    # 2. Process-global failover
    global _last_failover_time
    with _model_lock:
        provider = (config.LLM_PROVIDER).lower()
        current = (
            current_model_name
            or _extract_model_name(_active_llm)
            or config.AGENT_MODEL
        )
        mapper = SIBLING_MODEL_MAP.get(provider, SIBLING_MODEL_MAP["groq"])
        alt_model = mapper(current)
        logger.warning(
            f"Switching active inference model from '{current}' to sibling '{alt_model}' (provider: {provider})."
        )
        _last_failover_time = time.time()
        _active_llm, _active_llm_with_tools = create_model_instance(
            provider=provider,
            model_name=alt_model,
            temperature=0.0,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            streaming=True
        )
        return _active_llm, _active_llm_with_tools

def _is_rate_limit_or_not_found(exc):
    if isinstance(exc, (NotFoundError, RateLimitError)):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in (404, 429):
        return True
    exc_type_name = type(exc).__name__
    return any(term in exc_type_name for term in ("RateLimit", "NotFound", "ResourceExhausted"))

def _is_retryable_error(exc):
    # Groq-side transient failure: the provider intermittently rejects its own
    # generated tool call ("tool call validation failed ... missing properties")
    # before any tokens stream (verified: error arrives with zero stream chunks,
    # at both the 950 and 2048 token caps). Identical requests succeed on retry.
    if "tool call validation failed" in str(exc):
        return True
    if isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) in (413, 429, 500, 502, 503, 504):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code in (413, 429, 500, 502, 503, 504):
        return True
    exc_type_name = type(exc).__name__
    return any(term in exc_type_name for term in ("RateLimit", "InternalServer", "APIConnection", "ServiceUnavailable"))

@retry(
    retry=_is_retryable_error,
    wait=wait_exponential(multiplier=2, min=3, max=45),
    stop=stop_after_attempt(4),
    reraise=True
)
def _invoke_with_retry(is_tool_bound, messages, configurable=None):
    base_llm, tool_llm = get_active_models(configurable)
    target = tool_llm if is_tool_bound else base_llm
    curr_model = _extract_model_name(target)
    try:
        return target.invoke(messages)
    except Exception as e:
        if _is_rate_limit_or_not_found(e):
            logger.warning(f"Model '{curr_model}' failed with {type(e).__name__} ({e}). Switching to sibling model.")
            base_alt, tool_alt = switch_to_sibling(curr_model, configurable=configurable)
            target_alt = tool_alt if is_tool_bound else base_alt
            alt_model = _extract_model_name(target_alt)
            try:
                return target_alt.invoke(messages)
            except Exception as alt_err:
                logger.error(f"Sibling model failover to '{alt_model}' also failed: {alt_err}")
                raise
        raise



def get_windowed_messages(messages, max_messages=None):
    """
    Returns a sliding window of conversation messages aligned to a user turn (HumanMessage)
    without splitting tool calls from their corresponding tool outputs.
    """
    max_msgs = max_messages or getattr(config, "CONVERSATION_WINDOW", 8)
    if len(messages) <= max_msgs:
        return list(messages)
    
    # Start at -max_msgs cutoff
    start_idx = len(messages) - max_msgs
    
    # 1. Look for the nearest HumanMessage at or after start_idx
    forward_human_idx = None
    for i in range(start_idx, len(messages)):
        if isinstance(messages[i], HumanMessage):
            forward_human_idx = i
            break
            
    # 2. If a forward HumanMessage exists, use it to ensure the window stays <= max_msgs
    if forward_human_idx is not None and forward_human_idx > 0:
        return list(messages[forward_human_idx:])
        
    # 3. Otherwise, walk backward to find the nearest preceding HumanMessage
    slice_idx = start_idx
    while slice_idx > 0 and not isinstance(messages[slice_idx], HumanMessage):
        slice_idx -= 1
        
    return list(messages[slice_idx:])

def _prepare_context_messages(messages):
    """
    Enforces context truncation to config.CONVERSATION_WINDOW without splitting
    tool calls from tool messages, and bounds individual tool outputs to stay
    within model token quotas.
    """
    truncated = get_windowed_messages(messages, getattr(config, "CONVERSATION_WINDOW", 8))
    truncation_limit = getattr(config, "TOOL_TRUNCATION_LIMIT", 12000)
    cleaned = []
    for m in truncated:
        if isinstance(m, ToolMessage) and len(str(m.content)) > truncation_limit:
            bounded_text = str(m.content)[:truncation_limit] + "\n\n... [Output truncated to stay within model token quota]"
            cleaned.append(ToolMessage(content=bounded_text, tool_call_id=m.tool_call_id, name=m.name))
        else:
            cleaned.append(m)
    return cleaned


def _invoke_loop_guardrail(messages, directive: str, configurable=None) -> dict:
    """Terminal guardrail response: re-invoke the *unbound* LLM with a directive forcing
    plain-text synthesis, bypassing further tool routing for the current turn."""
    messages_with_warning = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages) + [HumanMessage(content=directive)]
    response = _invoke_with_retry(False, messages_with_warning, configurable=configurable)
    return {"messages": [response]}


def agent_node(state: AgentState, config: RunnableConfig | None = None):
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
                # Compare full sets of tool calls in case multiple parallel calls were generated
                last_call_tuples = [(c.get("name"), tuple(sorted((k, str(v)) for k, v in c.get("args", {}).items()))) for c in last_calls]
                penultimate_tuples = [(c.get("name"), tuple(sorted((k, str(v)) for k, v in c.get("args", {}).items()))) for c in penultimate_calls]
                if last_call_tuples == penultimate_tuples and last_call_tuples:
                    tool_name_str = last_calls[0].get("name", "tool")
                    logger.warning(f"Loop detected on tool '{tool_name_str}'. Injecting loop guardrail.")
                    loop_break_directive = (
                        f"Repeat tool call detected for '{tool_name_str}'. "
                        "Do not invoke this tool again. Output final answer in plain text using retrieved data."
                    )
                    return _invoke_loop_guardrail(messages, loop_break_directive, configurable=configurable)

            # Check for alternating tool ping-pong loop (e.g., A -> B -> A)
            if len(prev_ai_msgs) >= 3:
                last_names = [c.get("name") for c in prev_ai_msgs[-1].tool_calls if c.get("name")]
                antepenultimate_names = [c.get("name") for c in prev_ai_msgs[-3].tool_calls if c.get("name")]
                if last_names and last_names == antepenultimate_names:
                    loop_name = last_names[0]
                    logger.warning(f"Alternating tool loop detected for '{loop_name}'. Injecting loop guardrail.")
                    loop_break_directive = (
                        f"Alternating tool calls detected for '{loop_name}'. "
                        "Do not invoke further tools. Output final answer in plain text using retrieved data."
                    )
                    return _invoke_loop_guardrail(messages, loop_break_directive, configurable=configurable)

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
                f"Tool invocation limit reached for '{last_tool_name}'. "
                "Do not invoke further tools. Output final answer in plain text using retrieved data."
            )
            return _invoke_loop_guardrail(messages, loop_break_directive, configurable=configurable)
                
    messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + _prepare_context_messages(messages)
    response = _invoke_with_retry(True, messages_with_system, configurable=configurable)
    return {"messages": [response]}



tool_node = ToolNode(tools, handle_tool_errors=True)

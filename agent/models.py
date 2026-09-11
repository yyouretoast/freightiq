import os
import logging
from typing import Tuple, Any, Optional
import config
from agent.tools import tools

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {
    "groq": {
        "name": "Groq",
        "default_model": "qwen/qwen3.8-27b",
        "models": ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b"],
        "env_keys": ["GROQ_API_KEY"]
    },
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "env_keys": ["OPENAI_API_KEY"]
    },
    "ollama": {
        "name": "Ollama / Local (OpenAI-Compatible)",
        "default_model": "qwen2.5:14b",
        "models": ["qwen2.5:14b", "llama3.2:latest"],
        "env_keys": ["OLLAMA_BASE_URL", "OPENAI_BASE_URL"]
    }
}

def resolve_api_key(provider: str, user_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from user input, config, or environment variables."""
    if user_key and user_key.strip():
        return user_key.strip()
    
    provider_lower = provider.lower()
    if provider_lower == "groq":
        return getattr(config, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    elif provider_lower == "openai":
        return os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", None)
    elif provider_lower in ("ollama", "local", "openai_compatible"):
        return os.getenv("OPENAI_API_KEY", "ollama")
    return None

def bind_tools_safely(llm: Any) -> Any:
    """Bind agent domain tools with single-turn single-tool enforcement when supported."""
    try:
        return llm.bind_tools(tools, parallel_tool_calls=False)
    except (TypeError, ValueError):
        try:
            return llm.bind_tools(tools)
        except Exception as e:
            logger.warning(f"Failed to bind tools to model {llm}: {e}")
            return llm

def create_model_instance(
    provider: str = "groq",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    streaming: bool = True
) -> Tuple[Any, Any]:
    """
    Factory function returning (base_llm, tool_bound_llm) for the requested provider.
    """
    provider_lower = provider.lower()
    max_tokens = max_tokens or getattr(config, "MAX_OUTPUT_TOKENS", 1500)

    # 1. Groq Provider
    if provider_lower == "groq":
        from langchain_groq import ChatGroq
        key = resolve_api_key("groq", api_key)
        if not key:
            raise ValueError("GROQ_API_KEY is missing. Provide a valid Groq API key.")
        model = model_name or getattr(config, "AGENT_MODEL", "qwen/qwen3.8-27b")
        # Groq on-demand tier enforces a strict 1,000 Output Tokens Per Minute (OTPM) ceiling.
        # Cap Groq to 950 tokens to prevent 429 rate limit rejections, while alternative providers can use 1,500+.
        groq_tokens = min(max_tokens, 950) if max_tokens else 950
        llm = ChatGroq(
            model=model,
            groq_api_key=key,
            temperature=temperature,
            max_tokens=groq_tokens,
            streaming=streaming
        )
        return llm, bind_tools_safely(llm)

    # 2. Google Gemini Provider
    # 2. OpenAI Provider
    elif provider_lower == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain_openai is required for OpenAI models. Install with: pip install langchain-openai")
        key = resolve_api_key("openai", api_key)
        if not key:
            raise ValueError("OPENAI_API_KEY is missing.")
        model = model_name or "gpt-4o-mini"
        llm = ChatOpenAI(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
        return llm, bind_tools_safely(llm)

    # 3. Local / Ollama / OpenAI-Compatible Provider
    elif provider_lower in ("ollama", "local", "openai_compatible"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI
        endpoint = base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        key = resolve_api_key("ollama", api_key) or "ollama"
        model = model_name or "qwen2.5:14b"
        llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
        return llm, bind_tools_safely(llm)

    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Choose from: {list(SUPPORTED_PROVIDERS.keys())}")

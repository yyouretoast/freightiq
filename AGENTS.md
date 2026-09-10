# AGENTS.md: Developer & AI Coding Agent Guidelines

## 1. Project Overview
FreightIQ is an agentic freight carrier intelligence system. It utilizes LangGraph for stateful agent orchestration, Groq for fast LLM inference (`qwen/qwen3.8-27b`), ChromaDB for dense vector retrieval (`all-MiniLM-L6-v2`), HuggingFace SentenceTransformers CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for neural re-ranking, and SQLite (WAL mode) for structured querying.

## 2. Core Operational Directives
When inspecting, refactoring, or extending FreightIQ:
1. **Never Break Read-Only SQL Safety**: The SQL tool must always enforce `SELECT`-only statements, wrap user queries in a bounded subquery `LIMIT 25`, and open SQLite via `file:DB?mode=ro`.
2. **Preserve Fallbacks**:
   - `rag/reranker.py` must maintain the dense embedding cosine fallback if the CrossEncoder fails to load.
   - `agent/tools.py::web_search` must maintain the DuckDuckGo (`ddgs`) fallback if `TAVILY_API_KEY` is not provided.
   - `agent/nodes.py` must maintain the `qwen/qwen3.6-27b` sibling model fallback if Groq encounters rate limits or errors with `qwen/qwen3.8-27b`.
3. **Guardrails**:
   - Do not remove turn-scoped loop detection in `agent/nodes.py`.
   - Keep context truncation to the last 8 messages (aligned to `HumanMessage`) to stay within LLM token quotas.
4. **Idempotency**: All database population must go through `scripts/seed_db.py`. Do not create non-standard sqlite initialization routines.

## 3. Standard Verification Commands
Always run these commands before committing changes:
```bash
# Integration verification test suite (all 5 tools & LangGraph loop)
python -m tests.verify_system

# Retrieval benchmark evaluation (Recall@1, Recall@3, Recall@5, MRR)
python -m tests.evaluate_retrieval

# Trajectory & guardrail evaluation (20 scenarios, 100% target accuracy)
python -m tests.evaluate_agent_trajectories

# Concurrency & thread safety test
python -m tests.stress_test_concurrency
```

## 4. Key Directory & Code Map
- `agent/graph.py`: StateGraph definition and conditional edge router.
- `agent/nodes.py`: Agent reasoning node, system prompt, loop breaker logic.
- `agent/tools.py`: 5 domain tools (`carrier_sql_query`, `carrier_semantic_search`, `check_fmcsa_authority`, `freight_class_calculator`, `web_search`).
- `rag/reranker.py`: Two-stage neural cross-encoder with cosine fallback.
- `rag/retriever.py`: Hybrid SQLite + ChromaDB query abstraction.
- `scripts/seed_db.py`: Primary database generation and vector ingestion script.
- `app.py`: Streamlit frontend with token streaming and user Groq key override.

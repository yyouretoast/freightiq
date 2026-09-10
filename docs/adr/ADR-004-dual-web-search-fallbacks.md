# ADR-004: Dual-Engine Web Search Integration with Graceful Fallbacks

## Context & Problem Statement
Freight rate trends, port congestion updates, and spot rate benchmarks change rapidly in real-world freight markets. The FreightIQ agent requires access to live web intelligence. However:
1. Premium search APIs like Tavily require paid API credits and user API keys, which may be absent in local testing, development environments, or free-tier deployments.
2. Web search APIs occasionally experience network timeouts, rate limit throttling, or schema changes.

## Decision
We implement a dual-engine tiered web retrieval mechanism in `agent/tools.py::web_search`:
1. **Primary Engine (Tavily Search):** If `TAVILY_API_KEY` is configured in the environment, queries execute via Tavily for high-relevance structured search results.
2. **Autonomous Secondary Fallback (DuckDuckGo Search):** If `TAVILY_API_KEY` is omitted, or if Tavily throws an exception (HTTP 401/429/500, network failure), the tool immediately and silently falls back to DuckDuckGo (`ddgs.text()` / `ddgs.news()`).
3. **Structured Response Synthesis:** Both search engines format output into standardized `Title`, `Link`, and `Content` blocks with source attribution.

## Consequences
### Positive
- Zero runtime configuration blockers: The system operates completely out-of-the-box without requiring a paid Tavily account.
- Resilience: If external search providers face downtime, the user experience does not degrade or throw unhandled 500 errors.

### Trade-offs & Mitigations
- DuckDuckGo web scraping can occasionally be subject to anti-bot IP challenges. Mitigated by setting aggressive retry limits and gracefully returning informative empty notifications rather than unhandled tracebacks.

# CHANGELOG

All notable changes to the FreightIQ codebase are documented in this file.

## [2.1.0] - 2026-09-10

### Added
- **Hybrid Inverted Index + Vector Retrieval:** Integrated SQLite FTS5 BM25 search with ChromaDB dense vector search using Reciprocal Rank Fusion (RRF $k=60$).
- **Two-Stage Neural Cross-Encoder:** Implemented `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking over candidate fusion pools with fallback to dense cosine similarity.
- **Dataset Scaling to 500 Carriers:** Expanded synthetic carrier database from 200 to 500 rich carrier profiles with realistic fleet terminology (Moffett forklifts, TWIC badges, GDP cold chain, Carrier Vector multi-temp chillers).
- **Stratified 60-Query Retrieval Evaluation:** Replaced 20-query suite with 60 stratified test cases (20 Structured, 20 Qualitative, 20 Multi-Constraint Hybrid) reporting broken-down and overall Recall@1, Recall@3, Recall@5, and MRR.
- **Zero-Row SQL Constraint Relaxation Fallback:** Added automatic constraint relaxation to `carrier_sql_query` when strict multi-conditional `WHERE` queries return 0 rows.
- **Indirect Prompt Injection & Boundary Guardrails:** Added adversarial tests in `tests/evaluate_agent_trajectories.py` verifying resistance to system prompt leakage, SQL mutations, and out-of-domain jailbreaks.
- **Architecture Decision Records (ADRs):** Published ADR-001 through ADR-004 in `docs/adr/`.
- **Dataset Provenance Documentation:** Added `DATA.md` specifying database schema, ingestion lifecycle, and MCMIS scaling trade-offs.
- **Modern Packaging:** Added standard `pyproject.toml` configuration.

### Changed
- **Groq Token Protection & Context Truncation:** Enforced 8-message turn truncation and tool output character bounding to stay strictly within Groq Input Tokens Per Minute (ITPM) quotas.
- **Resilient LLM Inference Failover:** Implemented persistent sibling failover between `qwen/qwen3.8-27b` and `qwen/qwen3.6-27b` on rate limit or 404 occurrences.
- **FTS5 Query Sanitization:** Implemented regex tokenization, stop-word stripping, and disjunctive quoting in `sanitize_fts5_query()` to prevent SQLite FTS5 crashes on punctuation.

### Fixed
- Fixed Windows UTF-8 encoding crash in `tests/verify_system.py` by forcing `utf-8` standard output streams.
- Fixed `np.mean` NameError in `tests/evaluate_retrieval.py` by using standard library summation.
- Resolved ChromaDB idempotent re-ingestion checks to compare against active carrier records.

---

## [1.0.0] - 2026-09-08

### Added
- Initial FreightIQ release with LangGraph stateful agent orchestration.
- SQLite carrier database and ChromaDB dense vector store.
- 5 domain tools: `carrier_sql_query`, `carrier_semantic_search`, `check_fmcsa_authority`, `freight_class_calculator`, `web_search`.
- Streamlit interactive dashboard with token streaming and user API key configuration.

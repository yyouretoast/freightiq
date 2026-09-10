# CHANGELOG

All notable changes to the FreightIQ codebase are documented in this file.

## [2.2.0] - 2026-09-10

### Fixed
- **FMCSA Authority False-Positive Elimination:** Fixed string-comparison bug in `check_fmcsa_authority` that caused unverified USDOT numbers to return a clean `PASS`. The tool now strictly reports `UNVERIFIED / RECORD NOT FOUND` when neither local records nor the live registry confirm authority.
- **Orphaned ToolMessage API Crash Prevention:** Hardened `_prepare_context_messages()` in `agent/nodes.py` with turn-aligned walk-back to prevent slicing off parent assistant messages and triggering HTTP 400 errors from LLM providers.
- **Quote-Aware SQL Constraint Splitter:** Replaced naive regex splitting on `\s+AND\s+` in `carrier_sql_query` with a character scanner that ignores `AND` keywords inside single quotes and parentheses, preventing SQL corruption on queries matching multi-word carrier names or notes.

### Changed
- **Dead Code Pruning:** Removed obsolete PyTorch MLP training script (`scripts/train_reranker.py`), redundant middle shim (`scripts/init_db.py`), and non-standard root `setup.py`.
- **Config Cleanliness:** Pruned unused directory constants (`MODELS_DIR`, `SCRIPTS_DIR`) and dead embedding parameters from `config.py`.
- **Context Window Consolidation:** Consolidated `get_windowed_messages()` into `agent/nodes.py` and eliminated duplicate implementation in `app.py`.
- **LangSmith Observability Documentation:** Restored explicit LangSmith environment variable configuration and distributed tracing documentation in `README.md`.

---

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

# ADR-001: Dual-Modality SQL vs. Dense Vector Retrieval Routing

## Context & Problem Statement
Freight logistics inquiries naturally bifurcate into two distinct operational paradigms:
1. **Deterministic Structured Queries:** Queries with discrete, categorical, or numerical constraints (e.g., "Carriers in Ohio with satisfactory safety rating and dry van equipment", "Carriers with >15 years operating").
2. **Qualitative Semantic Queries:** Queries requiring natural language understanding of service capabilities, operational reputation, and specialized handling (e.g., "Carriers handling temperature-sensitive pharmaceutical cold chain", "Experienced drivers with live GPS milestone tracking").

Using vector similarity for structured queries yields poor precision, hallucinated attribute matching, and misses strict constraints (e.g., dense vectors fail to enforce exact state boundaries or discrete safety rating compliance). Conversely, attempting to parse free-form qualitative language into rigid relational schemas results in empty result sets or brittle SQL generation.

## Decision
We enforce a strict two-pronged routing architecture at the agent orchestration layer:
1. **Structured Domain Tool (`carrier_sql_query`):**
   - Routes deterministic queries directly to SQLite.
   - Enforces read-only safety at both the filesystem URI layer (`file:DB?mode=ro`) and SQL AST layer (rejecting non-`SELECT` statements).
   - Wraps all incoming queries in a bounded subquery (`SELECT * FROM (...) AS _bounded_carriers LIMIT 25`) to prevent runaway table scans or unbounded memory allocation.
   - Provides automated constraint relaxation fallback when multi-conditional `WHERE` clauses return 0 rows.
2. **Semantic Domain Tool (`carrier_semantic_search`):**
   - Routes qualitative and unstructured inquiries to a hybrid inverted-index + dense vector pipeline (FTS5 BM25 + ChromaDB Cosine) fused via Reciprocal Rank Fusion ($k=60$) and reranked via a neural Cross-Encoder.

## Consequences
### Positive
- 100% precision on discrete relational constraints (e.g., FMCSA compliance ratings, state jurisdictions).
- Clean separation of concerns between relational execution and neural semantic ranking.
- Sub-millisecond response latency (0.31 ms) on structured queries without invoking vector embeddings or neural inference, as demonstrated in [Figure 2: Latency vs. Accuracy Pareto Trade-Off](../assets/retrieval_latency_tradeoff.png).

### Trade-offs & Mitigations
- Multi-constraint hybrid queries ("Find California flatbed carriers specializing in high-theft electronics") require consensus between relational properties and semantic descriptions. As demonstrated in [Figure 3: Stratified Retrieval Performance](../assets/retrieval_stratified_categories.png), dense and lexical retrieval collapse to 0.100–0.150 Recall@1 when forced to resolve both dimensions simultaneously. This is mitigated by indexing relational metadata into hybrid FTS5 text documents and re-ranking with a neural cross-encoder, restoring performance to 0.550 Recall@1 (0.750 Recall@5) even under pure natural language input.

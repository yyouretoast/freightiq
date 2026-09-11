# ADR-003: SQLite FTS5 BM25 + Dense Vector Reciprocal Rank Fusion (RRF)

## Context & Problem Statement
Dense semantic retrieval alone fails when queries contain exact domain jargon, equipment codes, or identifiers (e.g., "Moffett", "RGN", "TWIC", "DOT 407", "JIT"). Dense embeddings represent terms as fuzzy statistical distributions, often retrieving semantically tangential documents while missing exact keyword hits. Conversely, pure keyword search (BM25) fails when queries use paraphrased natural language ("keeping medicine cold" vs. "GDP-compliant pharmaceutical cold chain").

## Decision
We implement a unified hybrid retrieval layer fusing SQLite FTS5 (BM25) and ChromaDB (dense vectors):
1. **SQLite FTS5 Inverted Index:** An external content virtual table `carriers_fts` indexing `carrier_name`, `service_regions`, `equipment_types`, `cargo_specializations`, and `notes`.
2. **Robust Query Sanitization:** Natural language queries are tokenized via regex, stripped of conversational stop words, and wrapped in double-quoted disjunctions (`"token1" OR "token2"`) to prevent SQLite FTS5 syntax crashes on punctuation (`-`, `/`, `:`, `?`, `()`).
3. **Reciprocal Rank Fusion (RRF):** Results from BM25 ($R_{bm25}$) and ChromaDB ($R_{dense}$) are merged into a single ranked list using the standard RRF formula with smoothing constant $k=60$:
   $$RRF(d) = \sum_{m \in \{bm25, dense\}} \frac{1}{k + rank_m(d)}$$
4. Top $N=15$ fused candidates are passed directly into the neural Cross-Encoder.

## Consequences
### Positive
- **High Recall Across All Query Modalities:** On 60 stratified benchmark queries (Structured, Qualitative, and Multi-Constraint Hybrid), hybrid fusion with Cross-Encoder re-ranking achieved 0.900 Recall@1 (0.942 MRR) on structured queries, 0.650 Recall@1 (0.727 MRR) on qualitative queries, 0.550 Recall@1 (0.625 MRR) on multi-constraint queries, and 0.764 overall MRR.
- **Zero External Search Engine Dependency:** Purely embedded within SQLite and ChromaDB—no ElasticSearch, OpenSearch, or external Java services required.

### Trade-offs & Mitigations
- FTS5 external content tables must stay in sync with the primary `carriers` table. Handled deterministically in `rag/setup_sqlite.py` and `scripts/seed_db.py` during seeding.

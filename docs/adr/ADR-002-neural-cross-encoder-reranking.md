# ADR-002: Neural Cross-Encoder Re-Ranking Architecture

## Context & Problem Statement
Dense embedding models like `all-MiniLM-L6-v2` compute bi-encoder dot products / cosine similarities independently for queries and documents. While computationally efficient for first-stage retrieval ($O(1)$ query vector dotted against indexed vectors), bi-encoders suffer from information loss because the query tokens cannot attend directly to document tokens during representation generation.

In freight logistics, subtle distinctions—such as "temperature monitoring for fresh produce" vs. "cryogenic transport for liquid gases"—often score similarly in generic embedding space, causing low-precision candidate rankings in the top 1 to 3 positions.

## Decision
We implement a two-stage retrieval pipeline:
1. **First-Stage Hybrid Recall:** Retrieve candidate pool of $N=15$ documents using Reciprocal Rank Fusion (RRF) combining SQLite FTS5 BM25 and ChromaDB vector search.
2. **Second-Stage Precision Re-Ranking:** Re-rank the candidate documents using a pre-trained neural Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The cross-encoder performs full cross-attention over `(query, document)` token pairs simultaneously, scoring passage relevance with high discriminatory fidelity.
3. **Resilient Fallback:** If the Cross-Encoder model fails to download or initialize (e.g., HuggingFace Hub rate limits, offline air-gapped environments), the reranker automatically falls back to dense embedding cosine similarity without throwing unhandled exceptions.

## Consequences
### Positive
- **Metric Gains:** Empirical benchmarks across 60 queries demonstrated a leap in Recall@1 from 0.300 (dense baseline) to **0.700** (+133.3%) and Overall MRR from 0.429 to **0.764** (+78.1%), with 0.650 Recall@1 and 0.727 MRR on qualitative domain queries.
- **Bounded Latency:** Scoring 15 candidate pairs adds ~35ms compute latency on CPU (~499ms total pipeline), staying well within interactive conversational turn thresholds.

### Trade-offs & Mitigations
- In-process memory footprint increases by ~80MB for model weights. This easily fits within free-tier container limits (HuggingFace Spaces 16GB RAM limit, local workstation ~80MB RAM).

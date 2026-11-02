# ADR-002: Neural Cross-Encoder Re-Ranking Architecture

## Context & Problem Statement
Dense embedding models like `all-MiniLM-L6-v2` compute bi-encoder dot products / cosine similarities independently for queries and documents. While computationally efficient for first-stage retrieval ($O(1)$ query vector dotted against indexed vectors), bi-encoders suffer from information loss because the query tokens cannot attend directly to document tokens during representation generation.

In freight logistics, subtle distinctions—such as "temperature monitoring for fresh produce" vs. "cryogenic transport for liquid gases"—often score similarly in generic embedding space, causing low-precision candidate rankings in the top 1 to 3 positions.

## Decision
We implement a two-stage retrieval pipeline:
1. **First-Stage Hybrid Recall:** Retrieve candidate pool of $N=15$ documents using Reciprocal Rank Fusion (RRF) combining SQLite FTS5 BM25 and ChromaDB vector search.
2. **Second-Stage Precision Re-Ranking:** Re-rank the candidate documents using a pre-trained neural Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The cross-encoder performs full cross-attention over `(query, document)` token pairs simultaneously, scoring passage relevance directly over token cross-attention.
3. **Resilient Fallback:** If the Cross-Encoder model fails to download or initialize (e.g., HuggingFace Hub rate limits, offline air-gapped environments), the reranker automatically falls back to dense embedding cosine similarity without throwing unhandled exceptions.

## Consequences
### Positive
- **Metric Gains:** Benchmarks across 60 queries showed an increase in Hit@1 from 0.317 (dense baseline) to **0.717** (+126.2%) and Overall MRR from 0.446 to **0.781** (+75.1%), with 0.650 Hit@1 and 0.727 MRR on qualitative domain queries (see [Figure 1: Multi-Strategy Retrieval Benchmark](../assets/retrieval_benchmark.png) and [Figure 3: Stratified Retrieval Performance](../assets/retrieval_stratified_categories.png)).
- **Bounded Latency:** Cross-Encoder neural inference over 15 candidate pairs dominates hybrid retrieval cost — per-query mean ~657 ms for the cosine-fallback hybrid and ~723 ms for the full cross-encoder pipeline vs ~5-6 ms for pure SQLite paths (measured by `tests/evaluate_retrieval.py` on a local CPU workstation; hardware-dependent). Stays within interactive conversational turn thresholds while trading compute for semantic precision (see [Figure 2: Latency vs. Accuracy Pareto Trade-Off](../assets/retrieval_latency_tradeoff.png)).

> *Latency figures cited in this ADR are measured by `tests/evaluate_retrieval.py` (per-query mean over 60 ground-truth queries, after warm-up) and exported to `data/benchmark_results.json`; re-run the benchmark to regenerate them for your hardware.*

### Trade-offs & Mitigations
- In-process memory footprint increases by ~80MB for the cross-encoder model weights — well within free-tier container limits (e.g. HuggingFace Spaces' 16GB RAM cap) and typical workstation RAM.
- Latency penalty (~723 ms measured vs. ~5 ms SQLite, hardware-dependent): Justifies dual routing in ADR-001 so that only qualitative and nuanced requests incur neural scoring.

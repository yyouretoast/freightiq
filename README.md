---
title: FreightIQ
emoji: 🚚
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# FreightIQ: Agentic Carrier Intelligence System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrator-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blue?style=flat-square&logo=analytics)](https://smith.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![FreightIQ Verification CI](https://github.com/yyouretoast/freightiq/actions/workflows/verify.yml/badge.svg)](https://github.com/yyouretoast/freightiq/actions/workflows/verify.yml)

FreightIQ is an agentic research and carrier intelligence assistant. Powered by a **LangGraph ReAct loop** and **Groq (Llama 3.3 70B)**, it routes shipping queries across a **hybrid search engine (ChromaDB + SQLite)**, re-ranks candidate profiles using a custom **PyTorch MLP**, and queries live web search for real-time market freight rates.

> **Live Demo:** [huggingface.co/spaces/yyouretoast/freightiq](https://huggingface.co/spaces/yyouretoast/freightiq)

<video src="https://github.com/user-attachments/assets/dbf58565-39ee-4d17-a434-6a321c8afed4" width="100%" controls></video>

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [System Architecture](#system-architecture)
- [Custom PyTorch Reranker](#custom-pytorch-reranker)
  - [Retrieval Performance Benchmarks](#retrieval-performance-benchmarks)
- [Production Hardening & Guardrails](#production-hardening--guardrails)
- [Execution Trace](#execution-trace)
- [Database Setup & Data Ingestion](#database-setup--data-ingestion)
- [Quickstart & Installation](#quickstart--installation)
- [Usage Examples](#usage-examples)
- [Known Limitations & Trade-offs](#known-limitations--trade-offs)
- [Roadmap & Production Scaling](#roadmap--production-scaling)
- [License](#license)

---

## Overview

Freight brokers and shippers spend significant time manually querying carrier directories, cross-referencing states, safety ratings, and equipment types across static spreadsheets and DOT lookup portals. FreightIQ provides a unified natural-language interface over these datasets: a user submits a query in plain text, and the agent dynamically routes the request to exact SQL queries, vector similarity lookups, NMFC density calculation, or live web search.

---

## Architecture & Tech Stack

* **Agent Orchestration:** LangGraph, LangChain (ReAct loop, conditional routing)
* **LLM Core:** Llama 3.3 70B (`llama-3.3-70b-versatile` via Groq Cloud API)
* **Vector DB & RAG:** ChromaDB (persistent vector storage)
* **Relational Database:** SQLite (structured read-only query engine)
* **Deep Learning Reranking:** PyTorch (`CarrierReRanker` 2-layer MLP classifier)
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Observability:** LangSmith (trace callbacks and execution telemetry)
* **Web APIs:** DuckDuckGo News API (market rate search)
* **Frontend:** Streamlit (streaming tokens, tool execution cards, active feedback logging)

---

## System Architecture

```text
                      +---------------------------------------+
                      |              User Query               |
                      +---------------------------------------+
                                           |
                                           v
                      +-----------------------------------------+
                      |           LangGraph START Node          |
                      +-----------------------------------------+
                                           |
                                           v
                      +-----------------------------------------+
+-------------------> |               Agent Node                |
|                     |        (Llama 3.3 70B + Tools)          |
|                     +-----------------------------------------+
|                                    /           \
|                         (Tool Requested?)   (No Tool / Done)
|                               /                     \
|                              v                       v
|                      +---------------+         +---------------+
|                      |   Tool Node   |         |  LangGraph    |
|                      | (Executes tool|         |   END Node    |
|                      +---------------+         +---------------+
|                              |                         |
|        +---------------------+-----+-----------------+ |
|        |                           |                 | |
|        v                           v                 v v
|    +--------------------+  +---------------+  +------------------+
|    |  carrier_semantic  |  |  carrier_sql  |  |  freight_class   |
|    |      _search       |  |    _query     |  |    calculator    |
|    +--------------------+  +---------------+  +------------------+
|        |                           |                 |
|        | (Retrieves k docs)        | (Runs SELECT)   | (Runs cubic density
|        v                           v                 |  calculations)
|    +--------------------+  +---------------+         v
|    |     ChromaDB       |  |   SQLite DB   |  +------------------+
|    | (Pre-computed embs)|  |  (Read-Only)  |  |   Output Results |
|    +--------------------+  +---------------+  +------------------+
|        |                                               |
|        v (Stored embeddings)                           |
|    +--------------------+                              |
|    |  PyTorch MLP Model |                              |
|    | (Cosine Fallback)  |                              |
|    +--------------------+                              |
|        |                                               |
|        v (Top-k Results)                               |
|        |                                               |
+--------+-----------------------------------------------+
```

---

## Custom PyTorch Reranker

FreightIQ uses a two-stage retrieval pipeline. Candidate documents retrieved from ChromaDB are re-scored by a custom PyTorch `CarrierReRanker` module (2-layer MLP with Xavier initialization).

The system executes one of two scoring paths:
1. **Fine-Tuned MLP Mode**: If trained weights exist on disk (`rag/data/reranker_weights.pt`), candidate document-query embedding pairs are passed through the MLP network to produce relevance logits.
2. **Cosine Fallback**: If weights are not present, the pipeline falls back to computing cosine similarity via `torch.nn.functional.cosine_similarity`.

Pipeline Steps:
1. **Candidate Retrieval**: ChromaDB extracts the top 15 nearest carrier profiles via vector similarity.
2. **Embedding Re-use**: Stored document embeddings are fetched directly from ChromaDB (`include=["embeddings"]`), avoiding re-encoding latency.
3. **Scoring & Ranking**: Tensors are scored by the MLP network or cosine fallback and sorted in descending order to return the top 5 candidates to the LLM agent.

### Retrieval Performance Benchmarks

Retrieved results were benchmarked across 20 ground-truth query scenarios (`tests/evaluate_retrieval.py`):

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **SQLite Exact Query** | **0.900** | **0.900** | **0.900** | **0.900** |
| **ChromaDB Base Vector** | 0.250 | 0.400 | 0.550 | 0.349 |
| **Reranked Search (Cosine)** | 0.250 | 0.400 | 0.550 | 0.349 |
| **Reranked Search (Trained MLP)** | 0.150 | **0.500** | **0.650** | 0.335 |

---

## Production Hardening & Guardrails

- **SQL Safety**: `carrier_sql_query` enforces strict `SELECT`-only validation and automatically appends `LIMIT 25` to prevent unbounded memory spikes and data exfiltration.
- **Rate-Limit Resilience**: LLM invocations are wrapped with `tenacity.retry` configured for exponential backoff, handling Groq `429 Too Many Requests` API rate limits automatically.
- **Loop Guardrails**: Implements runtime `recursion_limit=10` and dual guardrail logic to detect back-to-back duplicate tool calls and limit iterative SQL reformulations.
- **Read-Only SQLite Isolation**: Connects via `file:DB?mode=ro` to enforce connection-level read-only safety.
- **Singleton Model Caching**: Uses double-checked locking singletons to cache `SentenceTransformer` and PyTorch model instances in memory across Streamlit session runs.
- **Context Window Truncation**: Restricts message context to the last 8 messages per turn, keeping token consumption within API limits while preserving UI history.

---

## Execution Trace

<details>
<summary><b>View multi-tool agent routing trace</b></summary>

```text
[User Prompt]:
"Find a carrier located in Florida (FL) that handles fresh produce. 
What are their DOT and MC numbers? Also, what is the freight class for a 220 lbs crate 
measuring 36x36x36 inches?"

[Step 1 - Agent Reasoning]:
Tool Selected: carrier_sql_query
Input: SELECT * FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce') LIMIT 25

[Step 1 - Tool Execution Result]:
Carrier Name: SunState Logistics
DOT Number: 4892019
MC Number: MC-304918
HQ State: FL
Service Regions: Southeast, Mid-Atlantic
Equipment: Reefer, Dry Van
Specializations: Fresh Produce, Temperature-Controlled
Safety Rating: satisfactory
Years Operating: 14 years

[Step 2 - Agent Reasoning]:
Tool Selected: freight_class_calculator
Input: {"weight_lbs": 220, "length_in": 36, "width_in": 36, "height_in": 36, "cargo_description": "fresh produce"}

[Step 2 - Tool Execution Result]:
Shipment Dimensions: 36x36x36 inches
Volume: 27.00 cubic feet
Weight: 220 lbs
Calculated Density: 8.15 lb/ft³
Standard NMFC Freight Class: 110

[Step 3 - Final Response Synthesis]:
"SunState Logistics is headquartered in Florida (FL) and specializes in fresh produce.
- DOT Number: 4892019
- MC Number: MC-304918
- Safety Rating: Satisfactory (14 years operating)

Shipment Freight Class:
For a 220 lbs crate (36x36x36 in, 27.0 cu ft), the density is 8.15 lb/ft³, which maps to Standard NMFC Freight Class 110."
```

</details>

---

## Database Setup & Data Ingestion

Data setup is handled via distinct scripts for maintenance and reproducibility:
* `rag/generate_carriers.py`: Generates 200 synthetic carrier profiles.
* `rag/setup_sqlite.py`: Ingests carrier records into `carriers.db` with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`). Multi-value lists (`service_regions`, `equipment_types`, `cargo_specializations`) are stored as JSON arrays for `json_each()` SQL queries.
* `rag/ingest_chroma.py`: Encodes carrier profiles using `all-MiniLM-L6-v2` embeddings and persists them to ChromaDB.
* `scripts/init_db.py`: Master setup script executing generation, SQLite schema creation, and vector index ingestion.

---

## Quickstart & Installation

### 1. Clone repository & install dependencies
```bash
git clone https://github.com/yyouretoast/freightiq.git
cd freightiq
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here

# Optional: Override agent model for fast testing
# AGENT_MODEL=llama-3.1-8b-instant

# Optional: LangSmith Telemetry
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=FreightIQ-Agent
```

### 3. Initialize data environment
```bash
python scripts/init_db.py
```

### 4. Run application
```bash
streamlit run app.py
```

### 5. Run test suites & benchmarks
```bash
# Integration verification smoke test
python -m tests.verify_system

# Train PyTorch reranker with early stopping
python scripts/train_reranker.py

# Retrieval benchmark evaluation
python -m tests.evaluate_retrieval

# Multi-threaded concurrency stress test
python -m tests.stress_test_concurrency
```

---

## Usage Examples

1. **Structured State & Safety Search:**
   - *Query:* `"Find all carriers based in Ohio (OH) with a satisfactory safety rating."`
   - *Routing:* Triggers `carrier_sql_query` -> runs `SELECT * FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory' LIMIT 25`.

2. **JSON Array Attribute Matching:**
   - *Query:* `"We need flatbed carriers that handle hazardous materials in the Midwest."`
   - *Routing:* Triggers `carrier_sql_query` using `json_each()` on `service_regions`, `equipment_types`, and `cargo_specializations`.

3. **Qualitative Semantic Search:**
   - *Query:* `"Find me carriers known for exceptional handling of temperature-sensitive goods."`
   - *Routing:* Triggers `carrier_semantic_search` -> candidate retrieval via ChromaDB -> re-ranked via PyTorch MLP.

4. **NMFC Freight Class Calculation:**
   - *Query:* `"What is the NMFC freight class for a 1200 lbs pallet measuring 48x48x48 inches?"`
   - *Routing:* Triggers `freight_class_calculator` -> computes volume (64 cu ft) and density (18.75 lb/ft³) -> maps to Class 70.

---

## Known Limitations & Trade-offs

- **Evaluator Self-Preference Bias**: Evaluation of generated agent answers using LLM-as-a-Judge exhibits self-preference bias when the evaluator and generator share the same model family. Production evaluation harnesses should pair cross-provider evaluators (e.g., GPT-4o, Gemini 1.5 Pro) with exact metric benchmarks.
- **Groq API Free-Tier Throttling**: Groq free-tier rate limits enforce strict TPM/RPM quotas. Automated test scripts set `AGENT_MODEL=llama-3.1-8b-instant` to avoid rate limit spikes during batch test runs.
- **Ephemeral Host Filesystem**: Hugging Face Spaces storage is ephemeral. User feedback logged to `rag/data/feedback.json` resets on cold starts. In multi-instance production environments, feedback records should write directly to PostgreSQL or S3.

---

## Roadmap & Production Scaling

- **Database Scale**: Migrate local SQLite storage to PostgreSQL / Amazon RDS to support multi-region ACID transactions and distributed locking.
- **Vector Store Scale**: Transition local ChromaDB storage to managed vector infrastructure (Pgvector / Pinecone) for multi-million document indexes.
- **Async Execution**: Convert tool execution paths to `asyncio` for non-blocking concurrent tool execution under API server loads (FastAPI / Gunicorn).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

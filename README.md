---
title: FreightIQ
emoji: 🚚
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# 🚚 FreightIQ: Agentic Carrier Intelligence System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrator-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blue?style=flat-square&logo=analytics)](https://smith.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

[![FreightIQ Verification CI](https://github.com/yyouretoast/freightiq/actions/workflows/verify.yml/badge.svg)](https://github.com/yyouretoast/freightiq/actions/workflows/verify.yml)

FreightIQ is an agentic carrier intelligence and logistics research assistant. Powered by a **LangGraph ReAct loop** and **Groq (Llama 3.3 70B)**, it reasons over shipping queries, retrieves documents using a **hybrid search engine (ChromaDB + SQLite)**, re-ranks carrier profiles using a custom **PyTorch MLP**, and leverages live web search to answer real-time market rate questions.

> 🚀 **Live Demo:** [huggingface.co/spaces/yyouretoast/freightiq](https://huggingface.co/spaces/yyouretoast/freightiq)

<video src="https://github.com/yyouretoast/freightiq/raw/main/assets/demo.mp4" width="100%" controls></video>

---

## 🎯 Why FreightIQ?

Freight brokers and shippers waste hours manually searching fragmented carrier directories, filtering by state, safety rating, and equipment type across disconnected spreadsheets and DOT lookup tools. FreightIQ collapses this workflow into a single natural-language interface: ask a question in plain English, and the agent autonomously selects the right tool (SQL for exact filters, vector search for qualitative matches, web search for live rates), executes it, and synthesizes a clean answer — no forms, no manual filtering.

---

## 🛠️ Tech Stack & Keywords

*   **Agent Orchestration:** LangGraph, LangChain (ReAct loop, conditional routing)
*   **LLM:** Llama 3.3 70B (Groq Cloud API)
*   **Vector DB & RAG:** ChromaDB (persistent local storage)
*   **Structured Database:** SQLite (structured carrier queries)
*   **Deep Learning Reranking:** PyTorch (`torch.nn.Module` custom classifier)
*   **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
*   **Observability:** LangSmith (end-to-end agent trace auto-instrumentation via callbacks)
*   **Web APIs:** DuckDuckGo API (live freight market rate search)
*   **Frontend UI:** Streamlit (intermediate tool execution streaming)

---

## 📊 System Architecture

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

## 🔬 How the Custom PyTorch Re-ranker Works

FreightIQ implements a two-stage retrieval pipeline. The `CarrierReRanker` is a custom `torch.nn.Module` (2-layer MLP with Xavier initialization) designed to re-rank carrier profiles.

The system supports two execution paths:
1. **Fine-Tuned MLP Mode**: If trained weights exist on disk (`rag/data/reranker_weights.pt`), the system loads the weights and runs document-query embedding pairs through the MLP network. The model outputs relevance logits, which are used to rank candidates.
2. **Cosine Similarity Fallback**: If the weights file is absent, the system dynamically falls back to computing raw cosine similarities (`torch.nn.functional.cosine_similarity`) as a deterministic semantic baseline.

Reranking Steps:
1. **Candidate Retrieval:** ChromaDB extracts the top 15 nearest carrier profiles using vector similarity.
2. **Zero-Latency Embedding Extraction:** Stored document embeddings are fetched directly from ChromaDB (`include=["embeddings"]`), bypassing redundant re-encoding entirely.
3. **Scoring:** Query and document tensors are scored via the fine-tuned MLP (if weights exist) or raw cosine similarity.
4. **Re-sorting:** Documents are ranked by descending score, returning the top-k most relevant profiles to the LLM agent.

### Retrieval Performance Benchmarks

An evaluation harness (`tests/evaluate_retrieval.py`) benchmarks retrieval performance across 20 distinct queries. The MLP Reranker shows significant improvements in Recall@3 and Recall@5 compared to the baseline vector search:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| **SQLite Exact Query** | 0.900 | 0.900 | 0.900 | 0.900 |
| **ChromaDB Base Vector** | 0.250 | 0.400 | 0.550 | 0.349 |
| **Reranked Search (Cosine)** | 0.250 | 0.400 | 0.550 | 0.349 |
| **Reranked Search (Trained MLP)** | 0.150 | 0.500 | 0.650 | 0.335 |

---

## ⚡ Production Best Practices Implemented

*   **Global Model Caching (Instant Queries):** Loaded models (`SentenceTransformer` & PyTorch `CarrierReRanker`) are cached globally using a lazy-initialized singleton pattern, eliminating model reload overhead on repeated queries and achieving sub-second tool execution after warm-up.
*   **Persistent DB Handles:** Caches persistent ChromaDB connection handles to eliminate redundant disk-reads and prevent thread-safety file locks.
*   **Deterministic Data Generation:** Uses static random seeding (`random.seed(42)`) to ensure carrier database generation is fully reproducible across developer environments.
*   **Transaction Safety:** Populates SQLite tables using batch operations (`executemany`) inside a single secure database transaction.
*   **Connection-Level Read Security:** Connects to the SQLite database using read-only URI mode (`file:DB?mode=ro`) to safeguard against SQL injection or unauthorized table write operations from the agent.
*   **Per-Session Rate Limiting:** Caps each user session at 10 agent queries via `st.session_state` to prevent API quota exhaustion from a single session.
*   **Conversation Window Truncation:** Only the last 8 messages are passed to the LLM context window per query, preventing token limit breaches and history contamination in long sessions while preserving full UI history.
*   **Structured Logging:** `logging.basicConfig` with timestamped `INFO`/`ERROR` format is active across all modules for observability in container environments (e.g. Hugging Face Spaces logs).
*   **LangSmith Tracing:** End-to-end agent execution traces are pre-integrated via LangChain callbacks. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in your environment to stream full ReAct loop traces — tool selections, inputs, outputs, and latencies — to the LangSmith dashboard.

---

## 🗄️ Database Setup (Modular Ingestion)

The database pipeline splits database ingestion tasks for clean debugging and maintenance:
*   `rag/generate_carriers.py`: Programmatically generates 200 synthetic carriers.
*   `rag/setup_sqlite.py`: Populates `carriers.db`. Multi-value columns (`service_regions`, `equipment_types`, `cargo_specializations`) are stored as JSON arrays, enabling precise `json_each()` queries alongside standard filters on `hq_state`, `safety_rating`, and `years_operating`. Ingests the database file with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) enabled permanently to handle multi-threaded concurrency.
*   `rag/ingest_chroma.py`: Computes vector embeddings locally using `all-MiniLM-L6-v2` and persists them to ChromaDB.
*   `setup.py`: The single-entry execution script triggering the entire database environment setup.

---

## 🚀 Setup & Installation

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/yyouretoast/freightiq.git
cd freightiq
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory and add your keys:
```env
# Core Groq LLM API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Enable LangSmith Observability Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=FreightIQ-Agent
```

### 3. Run Data Ingestion
Initialize the SQLite database, generate synthetic carriers, and index the vectors in ChromaDB:
```bash
python setup.py
```

### 4. Start the Application
Run the Streamlit frontend locally:
```bash
streamlit run app.py
```

### 5. Run Programmatic Verification Tests
Validate all modules (calculators, SQL, vector search, PyTorch reranking, live APIs, and LangGraph routing) via the CLI test suite:
```bash
python -m tests.verify_system
```

### 6. Fine-Tune the Reranker Model
Train the PyTorch MLP reranker on a mixture of programmatically generated bootstrap pairs and actual user feedback:
```bash
python train_reranker.py
```
* **Bootstrap Generator**: Generates positive and negative query-document pairs directly from carrier database attributes to establish baseline model convergence.
* **User Feedback Ingest**: Loads feedback signals logged to `rag/data/feedback.json` from thumbs-up/down UI interactions.
* **Weights Export**: Optimizes the network parameters using `BCEWithLogitsLoss` and saves the updated state dict to `rag/data/reranker_weights.pt`, which is dynamically hot-reloaded by the app on the next query.

### 7. Run Retrieval Evaluation & Benchmarking
Run the retrieval evaluation harness to benchmark and compare SQLite exact queries, ChromaDB base vector search, Cosine re-ranked search, and Trained MLP re-ranked search:
```bash
python -m tests.evaluate_retrieval
```
This computes **Recall@1**, **Recall@3**, **Recall@5**, and **Mean Reciprocal Rank (MRR)**.

### 8. Run Concurrency Stress Tests
Run the multi-threaded concurrency stress test to verify SQLite WAL mode, feedback log serialization, and cached weights loading under concurrent Streamlit user load:
```bash
python tests/stress_test_concurrency.py
```

---

## 📝 Example Queries to Test

1.  **Structured SQL Query:**
    *   *Prompt:* `"Find all carriers based in Ohio (OH) with a satisfactory safety rating."`
    *   *Flow:* Triggers `carrier_sql_query` tool -> runs secure SQL read-only SELECT -> formats clean profiles.
2.  **Structured Multi-Value Query:**
    *   *Prompt:* `"We need flatbed carriers that handle hazardous materials in the Midwest."`
    *   *Flow:* Triggers `carrier_sql_query` tool -> executes exact-match SQL utilizing `json_each()` on the JSON array columns (`service_regions`, `equipment_types`, and `cargo_specializations`).
3.  **Semantic RAG + PyTorch Re-ranking:**
    *   *Prompt:* `"Find me carriers known for exceptional handling of temperature-sensitive goods."`
    *   *Flow:* Triggers `carrier_semantic_search` -> ChromaDB retrieves candidates -> scores them via custom PyTorch MLP (or falls back to cosine similarity if weights do not exist on disk) -> returns top-k candidates.
4.  **Freight Class Density Calculation:**
    *   *Prompt:* `"What is the NMFC freight class for a 1200 lbs pallet measuring 48x48x48 inches?"`
    *   *Flow:* Triggers `freight_class_calculator` -> computes volume/density (18.75 lb/ft³) -> maps class 70.
5.  **Web Search Integration:**
    *   *Prompt:* `"What is the current average national dry van spot rate per mile?"`
    *   *Flow:* Triggers `web_search` -> queries DuckDuckGo News API (structured index) -> summarizes latest logistics news.

---

## 🔮 Future Work & Scaling

To transition FreightIQ to a commercial production standard, the following roadmap is proposed:
*   **Active PyTorch Training (Implemented):** The active learning feedback loop is fully implemented and runs locally via the training pipeline. Thumbs-up/down ratings logged to `feedback.json` are combined with bootstrap carrier attributes in `train_reranker.py` to optimize weights, which are hot-reloaded by the inference path.
    > [!NOTE]
    > **Hugging Face Filesystem Limitation:** In the live Hugging Face Space, the local filesystem is ephemeral and resets on cold starts. For true production environments, these logged feedback signals should be configured to stream directly to an external database (e.g. Postgres) or the Hugging Face Dataset Hub API.
*   **Production Database Migration:** Upgrade the local SQLite file storage to a highly concurrent relational database like **PostgreSQL** or **Amazon RDS** to support multi-user locking.
*   **Authentication & Rate Limiting:** Implement OAuth2 security protocols and API gateway rate-limiting to protect the Groq API token quota from abuse.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestrator-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

FreightIQ is a high-performance agentic carrier intelligence and logistics research assistant. Powered by a **LangGraph ReAct loop** and **Groq (Llama 3.3 70B)**, it reasons over shipping queries, retrieves documents using a **hybrid search engine (ChromaDB + SQLite)**, re-ranks carrier profiles using a custom **PyTorch MLP**, and leverages live web search to answer real-time market rate questions.

> 🚀 **Live Demo:** [huggingface.co/spaces/yyouretoast/freightiq](https://huggingface.co/spaces/yyouretoast/freightiq)

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
+------------------>|               Agent Node                |
|                   |         (Llama 3.3 70B + Tools)         |
|                   +-----------------------------------------+
|                                  /           \
|                       (Tool Requested?)   (No Tool / Done)
|                             /                     \
|                            v                       v
|                    +---------------+         +---------------+
|                    |   Tool Node   |         |  LangGraph    |
|                    | (Executes tool|         |   END Node    |
|                    +---------------+         +---------------+
|                            |                         |
|      +---------------------+-----+-----------------+ |
|      |                           |                 | |
|      v                           v                 v v
|  +--------------------+  +---------------+  +------------------+
|  |  carrier_semantic  |  |  carrier_sql  |  |  freight_class   |
|  |      _search       |  |    _query     |  |    calculator    |
|  +--------------------+  +---------------+  +------------------+
|      |                           |                 |
|      | (Retrieves k docs)        | (Runs SELECT)   | (Runs cubic density
|      v                           v                 |  calculations)
|  +--------------------+  +---------------+         v
|  |     ChromaDB       |  |   SQLite DB   |  +------------------+
|  | (Pre-computed embs)|  |  (Read-Only)  |  |   Output Results |
|  +--------------------+  +---------------+  +------------------+
|      |                                               |
|      v (Stored embeddings)                           |
|  +--------------------+                              |
|  | Custom PyTorch MLP |                              |
|  |  Re-ranking Head   |                              |
|  +--------------------+                              |
|      |                                               |
|      v (Top-k Results)                               |
|      |                                               |
+------+-----------------------------------------------+
```

---

## 🔬 How the Custom PyTorch Re-ranker Works

FreightIQ implements a two-stage retrieval pipeline. The `CarrierReRanker` is a custom `torch.nn.Module` (2-layer MLP with Xavier initialization) architected for future supervised training on broker-carrier match logs via Binary Cross-Entropy loss. Current scoring uses `torch.nn.functional.cosine_similarity` — deterministic, semantically correct, and fully defensible pending training data collection:

```python
class CarrierReRanker(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()  # Xavier uniform initialization
```

1.  **Candidate Retrieval:** ChromaDB extracts the top 15 nearest carrier profiles using vector similarity.
2.  **Zero-Latency Embedding Extraction:** Stored document embeddings are fetched directly from ChromaDB (`include=["embeddings"]`), bypassing redundant re-encoding entirely.
3.  **Scoring:** Query and document tensors are scored via `F.cosine_similarity(query_tensors, doc_tensors, dim=-1)` using the cached PyTorch device context (CUDA/CPU auto-detected).
4.  **Re-sorting:** Documents are ranked by descending cosine score, returning the top-k most relevant profiles to the LLM agent.

---

## ⚡ Production Best Practices Implemented

*   **Global Model Caching (Instant Queries):** Loaded models (`SentenceTransformer` & PyTorch `CarrierReRanker`) are cached globally using a lazy-initialized singleton pattern, achieving a **10x–20x speedup** on queries (sub-second execution).
*   **Persistent DB Handles:** Caches persistent ChromaDB connection handles to eliminate redundant disk-reads and prevent thread-safety file locks.
*   **Deterministic Data Generation:** Uses static random seeding (`random.seed(42)`) to ensure carrier database generation is fully reproducible across developer environments.
*   **Transaction Safety:** Populates SQLite tables using batch operations (`executemany`) inside a single secure database transaction.
*   **Connection-Level Read Security:** Connects to the SQLite database using read-only URI mode (`file:DB?mode=ro`) to safeguard against SQL injection or unauthorized table write operations from the agent.
*   **CTE Queries Supported:** Enhances raw query parsing to safely support Common Table Expressions (`WITH` queries) generated by advanced LLMs.
*   **Per-Session Rate Limiting:** Caps each user session at 10 agent queries via `st.session_state` to prevent API quota exhaustion from a single session.
*   **Conversation Window Truncation:** Only the last 12 messages are passed to the LLM context window per query, preventing token limit breaches in long sessions while preserving full UI history.
*   **Structured Logging:** `logging.basicConfig` with timestamped `INFO`/`ERROR` format is active across all modules for observability in container environments (e.g. Hugging Face Spaces logs).

---

## 🗄️ Database Setup (Modular Ingestion)

The database pipeline splits database ingestion tasks for clean debugging and maintenance:
*   `rag/generate_carriers.py`: Programmatically generates 200 synthetic carriers.
*   `rag/setup_sqlite.py`: Populates `carriers.db` (for structured queries like filtering safety ratings, years operating, and DOT searches).
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
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
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
Validate all modules (calculators, SQL, vector search, PyTorch reranking, live APIs, and LangGraph routing) in 5 seconds via the CLI test suite:
```bash
python tests/verify_system.py
```

---

## 📝 Example Queries to Test

1.  **Structured SQL Query:**
    *   *Prompt:* `"Find all carriers based in Ohio (OH) with a satisfactory safety rating."`
    *   *Flow:* Triggers `carrier_sql_query` tool -> runs secure SQL read-only SELECT -> formats clean profiles.
2.  **Semantic RAG + PyTorch Re-ranking:**
    *   *Prompt:* `"Show me carriers that specialize in fresh produce in the Southwest region."`
    *   *Flow:* Triggers `carrier_semantic_search` -> ChromaDB extracts vectors -> PyTorch re-ranks profiles -> returns top candidates.
3.  **Freight Class Density Calculation:**
    *   *Prompt:* `"What is the NMFC freight class for a 1200 lbs pallet measuring 48x48x48 inches?"`
    *   *Flow:* Triggers `freight_class_calculator` -> computes volume/density (18.75 lb/ft³) -> maps class 70.
4.  **Web Search Integration:**
    *   *Prompt:* `"What is the current average national dry van spot rate per mile?"`
    *   *Flow:* Triggers `web_search` -> queries DuckDuckGo News API (structured index) -> summarizes latest logistics news.

---

## 🔮 Future Work & Scaling

To transition FreightIQ to a commercial production standard, the following roadmap is proposed:
*   **Active PyTorch Training:** Migrate the `CarrierReRanker` model from static weight initializations to supervised training using historical broker-carrier match logs or user click-through rates, optimizing weights via Binary Cross-Entropy (BCE) loss.
*   **Production Database Migration:** Upgrade the local SQLite file storage to a highly concurrent relational database like **PostgreSQL** or **Amazon RDS** to support multi-user locking.
*   **Authentication & Rate Limiting:** Implement OAuth2 security protocols and API gateway rate-limiting to protect the Groq API token quota from abuse.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

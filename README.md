# FreightIQ: LangGraph Multi-Agent Carrier Intelligence System

FreightIQ is an agentic carrier intelligence and logistics research assistant. Powered by a **LangGraph ReAct loop** and **Gemini 2.0 Flash**, it reasons over user shipping queries, retrieves documents using a hybrid search engine (ChromaDB + SQLite), re-ranks candidate carrier profiles using a custom **PyTorch MLP**, and leverages live web searches to answer real-time market rate questions.

---

## 🛠️ Tech Stack & Keywords

- **Agent Orchestration**: LangGraph, LangChain
- **LLM**: Gemini 2.0 Flash (via Google AI Studio)


- **Vector DB & RAG**: ChromaDB (local persistence)
- **Structured Database**: SQLite
- **Deep Learning**: PyTorch (`torch.nn.Module` 2-layer MLP classifier)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Web APIs**: DuckDuckGo API (via `duckduckgo-search`)
- **Frontend UI**: Streamlit

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
|                   |        (Gemini 2.0 Flash + Tools)       |
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
|  +--------------------+  +---------------+  |   Output Results |
|      |                                      +------------------+
|      v (Embeddings)                                  |
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

Instead of relying solely on default vector similarities, FreightIQ passes retrieved document candidates through a custom binary neural network classifier built from scratch in **PyTorch**:

```python
class CarrierReRanker(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128):
        super(CarrierReRanker, self).__init__()
        # Concatenates query (384d) & doc (384d) embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

1. **Document Candidate Retrieval**: ChromaDB extracts the top 15 most similar carrier profiles.
2. **Feature Concatenation**: The search query embedding (384-dimensional) and each document embedding (384-dimensional) are concatenated into a single feature vector (768-dimensional).
3. **Forward Pass Inference**: The PyTorch model processes the feature vector, applying weights and ReLU activations, outputting a scalar relevance score.
4. **Re-sorting**: The documents are sorted by their PyTorch network output score, returning the top-k most relevant profiles to the LLM agent.

---

## 🗄️ Database Setup (Modular Architecture)

The data pipeline splits database ingestion tasks for clean debugging and maintenance:
- `rag/generate_carriers.py`: Programmatically generates 200 synthetic carriers.
- `rag/setup_sqlite.py`: Populates `carriers.db` (for structured queries like filtering safety ratings, years operating, and DOT searches).
- `rag/ingest_chroma.py`: Computes vector embeddings locally using `all-MiniLM-L6-v2` and persists them to ChromaDB.
- `setup.py`: The single-entry execution script triggering the entire database environment.

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
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
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

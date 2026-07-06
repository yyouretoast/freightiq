import os

# Database & Storage Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rag", "data", "carriers.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CARRIERS_JSON_PATH = os.path.join(BASE_DIR, "rag", "data", "carriers.json")
CHROMA_COLLECTION_NAME = "freight_carriers"

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANKER_HIDDEN_DIM = 128

# Retrieval Parameters
SEMANTIC_POOL_SIZE = 15
SEMANTIC_RETRIEVAL_K = 5

# UI & Agent Session Configuration
MAX_QUERIES_PER_SESSION = 10
# Increased from 5 to 8 to restore context memory without hitting Groq TPM limits
CONVERSATION_WINDOW = 8
TOOL_TRUNCATION_LIMIT = 1200

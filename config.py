import os
import threading

# Database & Storage Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rag", "data", "carriers.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANKER_HIDDEN_DIM = 128

# Retrieval Parameters
SEMANTIC_POOL_SIZE = 15
SEMANTIC_RETRIEVAL_K = 5

# UI & Agent Session Configuration
MAX_QUERIES_PER_SESSION = 10
CONVERSATION_WINDOW = 12
TOOL_TRUNCATION_LIMIT = 400

# Global Thread Lock
# Defined here because config.py is imported normally, meaning this lock is cached
# in sys.modules and shared globally across all Streamlit exec namespaces.
setup_lock = threading.Lock()

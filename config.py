import os

# Base & Root Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Auto-create runtime directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# Database & Storage Paths
DB_PATH = os.path.join(DATA_DIR, "carriers.db")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
CARRIERS_JSON_PATH = os.path.join(DATA_DIR, "carriers.json")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")
CHROMA_COLLECTION_NAME = "freight_carriers"

# Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM & Search Configuration (centralized env var reads)
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "mock_key_for_ci"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))

# Retrieval Parameters
SEMANTIC_POOL_SIZE = 15
SEMANTIC_RETRIEVAL_K = 5

# UI & Agent Session Configuration
MAX_QUERIES_PER_SESSION = 10
CONVERSATION_WINDOW = 8
TOOL_TRUNCATION_LIMIT = 1200

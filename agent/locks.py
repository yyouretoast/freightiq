import threading

# Global locks shared across all Streamlit sessions via the python module cache
setup_lock = threading.Lock()
feedback_lock = threading.Lock()

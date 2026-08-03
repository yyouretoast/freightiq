import threading

# Thread-safe synchronization locks for initialization and feedback logging
setup_lock = threading.Lock()
feedback_lock = threading.Lock()

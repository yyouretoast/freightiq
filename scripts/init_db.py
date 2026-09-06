import sys
import os

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.seed_db import main

if __name__ == "__main__":
    main()

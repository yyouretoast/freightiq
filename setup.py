import os
import sys
import config

def main():
    print("=== Starting FreightIQ Data Environment Setup ===")
    
    json_path = config.CARRIERS_JSON_PATH
    if not os.path.exists(json_path):
        print("carriers.json not found. Generating synthetic data...")
        from rag.generate_carriers import main as generate_data
        generate_data()
        
    try:
        from rag.setup_sqlite import setup_sqlite
        setup_sqlite()
    except Exception as e:
        print(f"Error setting up SQLite: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        from rag.ingest_chroma import ingest_chroma
        ingest_chroma()
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("=== Setup Completed Successfully ===")

if __name__ == "__main__":
    main()

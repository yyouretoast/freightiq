import os
import sys
import time

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def main():
    start_time = time.time()
    force = "--force" in sys.argv
    print("=== FREIGHTIQ DATABASE & VECTOR INDEX SEEDER ===")
    if force:
        print("[INFO] Force re-seeding enabled via --force.")
    
    json_path = config.CARRIERS_JSON_PATH
    if force or not os.path.exists(json_path):
        print(f"Generating synthetic carrier dataset at {json_path}...")
        from rag.generate_carriers import main as generate_data
        generate_data()
    else:
        print(f"[OK] Found carrier dataset: {json_path}")
        
    print("\n[1/2] Seeding SQLite database (carriers.db)...")
    try:
        from rag.setup_sqlite import setup_sqlite
        setup_sqlite(force=force)
        print("[OK] SQLite database seeded successfully.")
    except Exception as e:
        print(f"[ERROR] SQLite setup failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n[2/2] Ingesting dense vector embeddings into ChromaDB...")
    try:
        from rag.ingest_chroma import ingest_chroma
        ingest_chroma(force=force)
        print("[OK] ChromaDB vector index seeded successfully.")
    except Exception as e:
        print(f"[ERROR] ChromaDB ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Write init sentinel
    sentinel_path = os.path.join(config.DATA_DIR, ".init_complete")
    with open(sentinel_path, "w", encoding="utf-8") as f:
        f.write("OK")
        
    elapsed = time.time() - start_time
    print(f"\n=== Database Seeding Complete in {elapsed:.2f}s ===")

if __name__ == "__main__":
    main()

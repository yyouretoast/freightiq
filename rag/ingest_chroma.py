import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import config
from rag.utils import format_carrier_document

def ingest_chroma():
    json_path = config.CARRIERS_JSON_PATH
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Source carriers.json not found at {json_path}. Run generate_carriers.py first.")

    # Initialize a single client instance reused for both the idempotency check and the upsert
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # Idempotency check: skip re-encoding if collection already populated
    try:
        collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION_NAME)
        if collection.count() > 0:
            print("Chroma collection already populated. Skipping embedding ingestion...")
            return
    except Exception:
        pass

    print("Ingesting carrier profiles into ChromaDB...")

    with open(json_path, "r") as f:
        carriers = json.load(f)

    documents = [format_carrier_document(c) for c in carriers]
    ids = [c['dot_number'] for c in carriers]
    metadatas = [{
        "dot_number": c["dot_number"],
        "carrier_name": c["carrier_name"],
        "hq_state": c["hq_state"],
        "safety_rating": c["safety_rating"]
    } for c in carriers]

    print(f"Loading SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Generating vector embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True).tolist()

    collection = chroma_client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    print("Upserting documents to ChromaDB...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Successfully ingested {len(carriers)} documents into ChromaDB.")

if __name__ == "__main__":
    ingest_chroma()

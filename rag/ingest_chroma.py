import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import config

def format_carrier_document(c):
    return (
        f"Carrier Name: {c['carrier_name']}\n"
        f"DOT Number: {c['dot_number']}\n"
        f"MC Number: {c['mc_number']}\n"
        f"HQ State: {c['hq_state']}\n"
        f"Service Regions: {', '.join(c['service_regions'])}\n"
        f"Equipment: {', '.join(c['equipment_types'])}\n"
        f"Specializations: {', '.join(c['cargo_specializations'])}\n"
        f"Safety Rating: {c['safety_rating']}\n"
        f"Years Operating: {c['years_operating']} years\n"
        f"Contact: {c['contact_email']}\n"
        f"Notes: {c['notes']}"
    )

def ingest_chroma():
    json_path = os.path.join("rag", "data", "carriers.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Source carriers.json not found at {json_path}. Run generate_carriers.py first.")
        
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
    
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="freight_carriers")
    
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

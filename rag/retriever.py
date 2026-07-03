import os
import sqlite3
import chromadb
from rag.reranker import rerank_documents, get_embed_model

DB_PATH = os.path.join("rag", "data", "carriers.db")
CHROMA_PATH = "./chroma_db"

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_chroma_collection():
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection(name="freight_carriers")
    return _CHROMA_COLLECTION

def retrieve_carriers_semantic(query, k=5):
    if not os.path.exists(CHROMA_PATH):
        return ["Error: Vector database not initialized. Run setup.py first."]
        
    try:
        # Pre-compute query embedding using the cached model to prevent Chroma from loading its own
        embed_model = get_embed_model()
        query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()
        
        collection = get_chroma_collection()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(15, collection.count()),
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return []
            
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        embeddings = results["embeddings"][0]
        
        # Pass pre-computed embeddings to avoid re-encoding in the reranker
        ranked_results = rerank_documents(
            query, docs, metadatas, top_k=k, 
            doc_embeddings=embeddings, query_embedding=query_vector
        )
        
        return [r["document"] for r in ranked_results]
    except Exception as e:
        return [f"Error in semantic retrieval: {str(e)}"]

def query_carriers_sql(sql_query):
    if not os.path.exists(DB_PATH):
        return "Error: SQL database not initialized. Run setup.py first."
        
    cleaned_query = sql_query.strip().lower()
    # Safely allow both SELECT and CTE queries
    if not (cleaned_query.startswith("select") or cleaned_query.startswith("with")):
        return "Error: Only read-only queries (SELECT / WITH) are permitted on the carrier database."
        
    conn = None
    try:
        # Connect strictly in read-only mode to prevent write attacks
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        if not rows:
            return "No matching records found in the SQL database."
            
        results = []
        for row in rows:
            record_dict = dict(row)
            fields = [f"{k.replace('_', ' ').title()}: {v}" for k, v in record_dict.items()]
            results.append("\n".join(fields))
            
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"SQLite Error: {str(e)}\nEnsure you are querying columns from the 'carriers' table: id, carrier_name, dot_number, mc_number, hq_state, service_regions, equipment_types, cargo_specializations, safety_rating, years_operating, contact_email, notes."
    finally:
        if conn:
            conn.close()

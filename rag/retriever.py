import logging
import os
import sqlite3
import chromadb
from rag.reranker import rerank_documents, get_embed_model

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("rag", "data", "carriers.db")
CHROMA_PATH = "./chroma_db"

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_chroma_collection():
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="freight_carriers")
        _CHROMA_CLIENT = client
        _CHROMA_COLLECTION = collection
        logger.info(f"ChromaDB collection loaded: {collection.count()} documents")
    return _CHROMA_COLLECTION

def retrieve_carriers_semantic(query, k=5):
    if not os.path.exists(CHROMA_PATH):
        return ["Error: Vector database not initialized. Run setup.py first."]

    try:
        embed_model = get_embed_model()
        query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()

        collection = get_chroma_collection()
        n_results = min(15, collection.count())
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "embeddings"]
        )

        if not results or not results["documents"] or not results["documents"][0]:
            logger.warning("Semantic search returned no results.")
            return []

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        embeddings = results["embeddings"][0]

        ranked_results = rerank_documents(
            query, docs, metadatas, top_k=k,
            doc_embeddings=embeddings, query_embedding=query_vector
        )

        logger.info(f"Semantic search complete: {len(ranked_results)} results returned for query='{query[:60]}'")
        return [r["document"] for r in ranked_results]
    except Exception as e:
        logger.error(f"Semantic retrieval error: {e}")
        return [f"Error in semantic retrieval: {str(e)}"]

def query_carriers_sql(sql_query):
    if not os.path.exists(DB_PATH):
        return "Error: SQL database not initialized. Run setup.py first."

    cleaned_query = sql_query.strip().lower()
    if not (cleaned_query.startswith("select") or cleaned_query.startswith("with")):
        return "Error: Only read-only queries (SELECT / WITH) are permitted on the carrier database."

    conn = None
    try:
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

        logger.info(f"SQL query returned {len(rows)} rows.")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error(f"SQL query error: {e} | Query: {sql_query}")
        return f"SQLite Error: {str(e)}\nEnsure you are querying columns from the 'carriers' table: id, carrier_name, dot_number, mc_number, hq_state, service_regions, equipment_types, cargo_specializations, safety_rating, years_operating, contact_email, notes."
    finally:
        if conn:
            conn.close()

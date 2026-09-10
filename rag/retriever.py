import json
import logging
import os
import re
import sqlite3
import threading
import chromadb
from rag.reranker import rerank_documents, get_embed_model
from rag.utils import format_carrier_document
import config

logger = logging.getLogger(__name__)

# Conversational stop words stripped from FTS5 queries to isolate domain tokens
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "but", "and", "or", "if", "because", "as", "until", "while",
    "that", "this", "these", "those", "then", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "find", "show", "me", "we",
    "need", "give", "get", "what", "which", "who", "whom", "whose",
    "carrier", "carriers", "company", "companies", "located", "headquartered"
}

# Singletons and Thread Locks
_CHROMA_COLLECTION = None
_chroma_lock = threading.Lock()

def get_chroma_collection():
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        with _chroma_lock:
            if _CHROMA_COLLECTION is None:
                client = chromadb.PersistentClient(path=config.CHROMA_PATH)
                _CHROMA_COLLECTION = client.get_collection(name=config.CHROMA_COLLECTION_NAME)
                logger.info(f"ChromaDB collection loaded: {_CHROMA_COLLECTION.count()} documents")
    return _CHROMA_COLLECTION

def sanitize_fts5_query(raw_query: str) -> str | None:
    """
    Extracts alphanumeric tokens, strips conversational stop words,
    escapes reserved words, and wraps tokens in quotes to prevent FTS5 syntax errors.
    """
    tokens = re.findall(r'[A-Za-z0-9]+', raw_query)
    clean = [t for t in tokens if t.lower() not in STOP_WORDS and len(t) > 1]
    if not clean:
        clean = [t for t in tokens if len(t) > 1]
    if not clean:
        return None
    return " OR ".join(f'"{t}"' for t in clean)

def retrieve_carriers_bm25(query: str, limit: int = 25) -> list[dict]:
    """
    Performs fast lexical BM25 retrieval against the SQLite FTS5 inverted index.
    """
    if not os.path.exists(config.DB_PATH):
        return []
    expr = sanitize_fts5_query(query)
    if not expr:
        return []
    try:
        with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT c.*, f.rank
            FROM carriers c
            JOIN carriers_fts f ON c.id = f.rowid
            WHERE carriers_fts MATCH ?
            ORDER BY f.rank
            LIMIT ?
            """, (expr, limit))
            rows = cursor.fetchall()

            candidates = []
            for r in rows:
                row_dict = dict(r)
                doc_text = format_carrier_document(row_dict)
                dot_num = str(row_dict["dot_number"])
                candidates.append({
                    "dot_number": dot_num,
                    "carrier_name": row_dict["carrier_name"],
                    "hq_state": row_dict["hq_state"],
                    "safety_rating": row_dict["safety_rating"],
                    "document": doc_text,
                    "metadata": {
                        "dot_number": dot_num,
                        "carrier_name": row_dict["carrier_name"],
                        "hq_state": row_dict["hq_state"],
                        "safety_rating": row_dict["safety_rating"]
                    }
                })
            return candidates
    except Exception as e:
        logger.error(f"FTS5 BM25 retrieval error: {e}")
        return []

def reciprocal_rank_fusion(bm25_cands: list[dict], dense_cands: list[dict], k: int = 60, top_n: int = 15) -> list[dict]:
    """
    Combines ranked candidate lists from BM25 and dense retrieval using Reciprocal Rank Fusion.
    """
    scores = {}
    doc_map = {}

    for rank_idx, item in enumerate(bm25_cands):
        dot = item["dot_number"]
        scores[dot] = scores.get(dot, 0.0) + (1.0 / (k + rank_idx + 1))
        if dot not in doc_map:
            doc_map[dot] = item

    for rank_idx, item in enumerate(dense_cands):
        dot = item["dot_number"]
        scores[dot] = scores.get(dot, 0.0) + (1.0 / (k + rank_idx + 1))
        if dot not in doc_map:
            doc_map[dot] = item

    sorted_dots = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [doc_map[d] for d in sorted_dots[:top_n]]

def retrieve_carriers_semantic(query, k=config.SEMANTIC_RETRIEVAL_K):
    """
    Two-stage hybrid retrieval:
    1. Lexical retrieval via SQLite FTS5 (BM25)
    2. Dense semantic retrieval via ChromaDB (all-MiniLM-L6-v2)
    3. Candidate fusion via Reciprocal Rank Fusion (RRF k=60)
    4. Neural re-ranking via Hugging Face Cross-Encoder
    """
    if not os.path.exists(config.CHROMA_PATH) or not os.path.exists(config.DB_PATH):
        return ["Error: Databases not initialized. Run scripts/seed_db.py first."]

    try:
        # 1. Lexical candidate retrieval
        bm25_cands = retrieve_carriers_bm25(query, limit=25)

        # 2. Dense semantic candidate retrieval
        embed_model = get_embed_model()
        query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()

        collection = get_chroma_collection()
        total_docs = collection.count()
        dense_cands = []
        if total_docs > 0:
            n_results = min(config.SEMANTIC_POOL_SIZE * 2, total_docs)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            if results and results["documents"] and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    dense_cands.append({
                        "dot_number": str(meta["dot_number"]),
                        "carrier_name": meta.get("carrier_name", ""),
                        "hq_state": meta.get("hq_state", ""),
                        "safety_rating": meta.get("safety_rating", ""),
                        "document": doc,
                        "metadata": meta
                    })

        # 3. Reciprocal Rank Fusion
        fused_cands = reciprocal_rank_fusion(
            bm25_cands, dense_cands, k=60, top_n=config.SEMANTIC_POOL_SIZE
        )

        if not fused_cands:
            logger.warning("Hybrid retrieval returned no candidates.")
            return []

        docs = [c["document"] for c in fused_cands]
        metadatas = [c["metadata"] for c in fused_cands]

        # 4. Cross-Encoder re-ranking
        ranked_results = rerank_documents(
            query, docs, metadatas, top_k=k, query_embedding=query_vector
        )

        logger.info(f"Hybrid retrieval complete: {len(ranked_results)} results returned for query='{query[:60]}'")
        return [r["document"] for r in ranked_results]
    except Exception as e:
        logger.error(f"Hybrid retrieval error: {e}")
        return [f"Error in hybrid retrieval: {str(e)}"]

def query_carriers_sql(sql_query):
    if not os.path.exists(config.DB_PATH):
        return "Error: SQL database not initialized. Run scripts/seed_db.py first."

    # SQLite read-only connection limits are enforced at the connection level (?mode=ro).
    # This renders manual string/keyword matching redundant, as the engine rejects any writes or mutations.

    conn = None
    try:
        conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        if not rows:
            return "No matching records found in the SQL database."

        results = []
        for row in rows:
            record_dict = dict(row)
            fields = []
            for k, v in record_dict.items():
                # Pretty-print JSON array columns (service_regions, equipment_types, cargo_specializations)
                if isinstance(v, str) and v.startswith("["):
                    try:
                        v = ", ".join(json.loads(v))
                    except (json.JSONDecodeError, TypeError):
                        pass
                fields.append(f"{k.replace('_', ' ').title()}: {v}")
            results.append("\n".join(fields))

        logger.info(f"SQL query returned {len(rows)} rows.")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error(f"SQL query error: {e} | Query: {sql_query}")
        return f"SQLite Error: {str(e)}\nEnsure you are querying columns from the 'carriers' table: id, carrier_name, dot_number, mc_number, hq_state, service_regions, equipment_types, cargo_specializations, safety_rating, years_operating, contact_email, notes."
    finally:
        if conn:
            conn.close()

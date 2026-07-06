import os
import json
import sqlite3
import numpy as np
import torch
import sys

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.retriever import get_chroma_collection, query_carriers_sql, retrieve_carriers_semantic
from rag.reranker import rerank_documents, get_embed_model

# Ground truth test cases: (query, target_dot_numbers, corresponding_sql)
EVAL_CASES = [
    {
        "query": "Find a carrier located in Florida (FL) that handles fresh produce.",
        "targets": ["9088677", "3780770"], # Titan Systems #114, Swift Carriers #120
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "query": "We need flatbed carriers that handle hazardous materials in the Midwest.",
        "targets": ["9858078"], # Falcon Systems #252
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'hazardous materials')"
    },
    {
        "query": "Find a carrier headquartered in Ohio (OH) with a satisfactory safety rating.",
        "targets": ["6725944", "5942938"], # NextGen Haulers 80 #101, Swift Freight 80 #143
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory'"
    },
    {
        "query": "Show me carriers headquartered in Texas (TX) equipped with dry vans.",
        "targets": ["7994650", "4236344"], # Red Line Trucking 69 #116, Voyager Lines 38 #147
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van')"
    },
    {
        "query": "We need LTL carriers that operate in the Mountain region.",
        "targets": ["1793595", "3952690"], # Crossroads Trucking #105, Echo Express 96 #121
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mountain') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL')"
    }
]

def run_sqlite_retrieval(sql):
    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [str(row[0]) for row in rows]
    except Exception as e:
        print(f"SQL execution error in evaluation: {e}")
        return []

def run_chroma_vector_search(query, k=5):
    collection = get_chroma_collection()
    embed_model = get_embed_model()
    query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["metadatas"]
    )
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return []
    return [str(m["dot_number"]) for m in results["metadatas"][0]]

def run_reranked_hybrid_search(query, k=5, use_trained=True):
    collection = get_chroma_collection()
    embed_model = get_embed_model()
    query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()
    
    # Retrieve larger pool for reranking
    pool_size = config.SEMANTIC_POOL_SIZE
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=pool_size,
        include=["documents", "metadatas", "embeddings"]
    )
    if not results or not results["documents"] or not results["documents"][0]:
        return []
        
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    embeddings = results["embeddings"][0]
    
    # Check weights file
    weights_path = os.path.join(config.BASE_DIR, "rag", "data", "reranker_weights.pt")
    has_weights = os.path.exists(weights_path)
    
    # Temporarily hide/show weights path to test both Cosine and MLP
    if not use_trained and has_weights:
        # Temporarily rename to bypass MLP check
        temp_path = weights_path + ".tmp"
        os.rename(weights_path, temp_path)
        try:
            ranked = rerank_documents(query, docs, metadatas, top_k=k, doc_embeddings=embeddings, query_embedding=query_vector)
        finally:
            os.rename(temp_path, weights_path)
    else:
        ranked = rerank_documents(query, docs, metadatas, top_k=k, doc_embeddings=embeddings, query_embedding=query_vector)
        
    return [str(r["metadata"]["dot_number"]) for r in ranked]

def calculate_metrics(retrieved_list, targets):
    """
    Calculate Recall@K (K=1,3,5) and Mean Reciprocal Rank (MRR)
    """
    recall_1 = 1.0 if any(t in retrieved_list[:1] for t in targets) else 0.0
    recall_3 = 1.0 if any(t in retrieved_list[:3] for t in targets) else 0.0
    recall_5 = 1.0 if any(t in retrieved_list[:5] for t in targets) else 0.0
    
    mrr = 0.0
    for idx, item in enumerate(retrieved_list):
        if item in targets:
            mrr = 1.0 / (idx + 1)
            break
            
    return recall_1, recall_3, recall_5, mrr

def main():
    print("=== FREIGHTIQ RETRIEVAL BENCHMARK & EVALUATION HARNESS ===")
    
    strategies = {
        "SQLite Exact Query": lambda case: run_sqlite_retrieval(case["sql"]),
        "ChromaDB Base Vector": lambda case: run_chroma_vector_search(case["query"]),
        "Reranked Search (Cosine)": lambda case: run_reranked_hybrid_search(case["query"], use_trained=False),
        "Reranked Search (Trained MLP)": lambda case: run_reranked_hybrid_search(case["query"], use_trained=True)
    }
    
    results = {}
    for name in strategies:
        results[name] = {"r@1": [], "r@3": [], "r@5": [], "mrr": []}
        
    for case in EVAL_CASES:
        print(f"\nEvaluating Query: '{case['query']}'")
        for name, search_fn in strategies.items():
            # Skip trained MLP if weights don't exist yet
            if name == "Reranked Search (Trained MLP)" and not os.path.exists(os.path.join(config.BASE_DIR, "rag", "data", "reranker_weights.pt")):
                continue
                
            retrieved = search_fn(case)
            r1, r3, r5, mrr = calculate_metrics(retrieved, case["targets"])
            
            results[name]["r@1"].append(r1)
            results[name]["r@3"].append(r3)
            results[name]["r@5"].append(r5)
            results[name]["mrr"].append(mrr)
            
            print(f"  - {name:<30} | Retrieved: {len(retrieved):<2} | Recall@5: {r5:.1f} | MRR: {mrr:.3f}")
            
    # Print Summary Table
    print("\n\n=== OVERALL RETRIEVAL METRICS SUMMARY ===")
    print(f"| {'Strategy':<30} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} |")
    print(f"| {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} |")
    
    for name, metrics in results.items():
        if not metrics["r@1"]: # Skip if empty
            continue
        avg_r1 = np.mean(metrics["r@1"])
        avg_r3 = np.mean(metrics["r@3"])
        avg_r5 = np.mean(metrics["r@5"])
        avg_mrr = np.mean(metrics["mrr"])
        print(f"| {name:<30} | {avg_r1:<10.3f} | {avg_r3:<10.3f} | {avg_r5:<10.3f} | {avg_mrr:<8.3f} |")
        
    print("\n=== Evaluation Harness Complete ===")

if __name__ == "__main__":
    main()

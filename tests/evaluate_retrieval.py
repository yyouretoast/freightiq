import os
import sqlite3
import numpy as np
import sys

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.retriever import get_chroma_collection, query_carriers_sql
from rag.reranker import rerank_documents, get_embed_model

# Ground truth test cases: (query, SQL query used to resolve targets dynamically)
EVAL_CASES = [
    {
        "query": "Find a carrier located in Florida (FL) that handles fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "query": "We need flatbed carriers that handle hazardous materials in the Midwest.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'hazardous materials')"
    },
    {
        "query": "Find a carrier headquartered in Ohio (OH) with a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory'"
    },
    {
        "query": "Show me carriers headquartered in Texas (TX) equipped with dry vans.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van')"
    },
    {
        "query": "We need LTL carriers that operate in the Mountain region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mountain') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL')"
    },
    {
        "query": "Find a carrier located in California (CA) that has a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND safety_rating = 'satisfactory'"
    },
    {
        "query": "We need reefer carriers in the Northeast region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer')"
    },
    {
        "query": "Find a carrier in Georgia (GA) specializing in building materials.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'GA' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'building materials')"
    },
    {
        "query": "Show me dry van carriers operating in the Pacific Northwest specializing in electronics.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'electronics')"
    },
    {
        "query": "We need carriers in Illinois (IL) with over 15 years operating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'IL' AND years_operating > 15"
    },
    {
        "query": "Find LTL carriers headquartered in New York (NY) with a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NY' AND safety_rating = 'satisfactory' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL')"
    },
    {
        "query": "We need flatbed carriers in the Southeast region specializing in fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "query": "Find a carrier in North Carolina (NC) specializing in general freight.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NC' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'general freight')"
    },
    {
        "query": "We need dry van carriers operating in the Southwest region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van')"
    },
    {
        "query": "Show me reefer carriers in Florida (FL) specializing in fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "query": "Find a carrier operating in the Mid-Atlantic region specializing in electronics.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mid-Atlantic') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'electronics')"
    },
    {
        "query": "We need flatbed carriers in Texas (TX) with over 10 years of operations.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND years_operating > 10 AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed')"
    },
    {
        "query": "Find reefer carriers headquartered in Ohio (OH) with satisfactory safety ratings.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer')"
    },
    {
        "query": "Show me carriers in California (CA) equipped with flatbeds.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed')"
    },
    {
        "query": "We need LTL carriers in Georgia (GA) specializing in building materials.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'GA' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'building materials')"
    }
]

def resolve_ground_truth_targets():
    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        print("Warning: carriers.db not found. Run setup.py first.")
        return
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for case in EVAL_CASES:
            try:
                cursor.execute(case["sql"])
                rows = cursor.fetchall()
                case["targets"] = [str(row[0]) for row in rows]
            except Exception as e:
                print(f"Error resolving target for query '{case['query']}': {e}")
                case["targets"] = []

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
    
    # Pass force_cosine flag based on use_trained and whether weights file actually exists
    force_cosine = not use_trained or not has_weights
    ranked = rerank_documents(
        query, docs, metadatas, top_k=k,
        doc_embeddings=embeddings, query_embedding=query_vector,
        force_cosine=force_cosine
    )
        
    return [str(r["metadata"]["dot_number"]) for r in ranked]

def calculate_metrics(retrieved_list, targets):
    if not targets:
        return 0.0, 0.0, 0.0, 0.0
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
    
    # Dynamically resolve ground-truth targets from DB first
    resolve_ground_truth_targets()
    
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
            r1, r3, r5, mrr = calculate_metrics(retrieved, case.get("targets", []))
            
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

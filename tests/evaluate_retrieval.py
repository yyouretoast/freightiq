import os
import sqlite3
import sys

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.retriever import get_chroma_collection, query_carriers_sql, retrieve_carriers_bm25, reciprocal_rank_fusion
from rag.reranker import rerank_documents, get_embed_model

# Ground truth test cases: (category, query, SQL query used to resolve targets dynamically)
EVAL_CASES = [
    # --- Category 1: Structured Queries (20 cases) ---
    {
        "category": "Structured",
        "query": "Find a carrier located in Florida (FL) that handles fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "category": "Structured",
        "query": "We need flatbed carriers that handle hazardous materials in the Midwest.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'hazardous materials')"
    },
    {
        "category": "Structured",
        "query": "Find a carrier headquartered in Ohio (OH) with a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory'"
    },
    {
        "category": "Structured",
        "query": "Show me carriers headquartered in Texas (TX) equipped with dry vans.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van')"
    },
    {
        "category": "Structured",
        "query": "We need LTL carriers that operate in the Mountain region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mountain') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL')"
    },
    {
        "category": "Structured",
        "query": "Find a carrier located in California (CA) that has a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND safety_rating = 'satisfactory'"
    },
    {
        "category": "Structured",
        "query": "We need reefer carriers in the Northeast region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer')"
    },
    {
        "category": "Structured",
        "query": "Find a carrier in Georgia (GA) specializing in building materials.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'GA' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'building materials')"
    },
    {
        "category": "Structured",
        "query": "Show me dry van carriers operating in the Pacific Northwest specializing in electronics.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'electronics')"
    },
    {
        "category": "Structured",
        "query": "We need carriers in Illinois (IL) with over 15 years operating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'IL' AND years_operating > 15"
    },
    {
        "category": "Structured",
        "query": "Find LTL carriers headquartered in New York (NY) with a satisfactory safety rating.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NY' AND safety_rating = 'satisfactory' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'LTL')"
    },
    {
        "category": "Structured",
        "query": "We need flatbed carriers in the Southeast region specializing in fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "category": "Structured",
        "query": "Find a carrier in North Carolina (NC) specializing in general freight.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NC' AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'general freight')"
    },
    {
        "category": "Structured",
        "query": "We need dry van carriers operating in the Southwest region.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southwest') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'dry van')"
    },
    {
        "category": "Structured",
        "query": "Show me reefer carriers in Florida (FL) specializing in fresh produce.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'fresh produce')"
    },
    {
        "category": "Structured",
        "query": "Find a carrier operating in the Northeast region specializing in electronics.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND EXISTS (SELECT 1 FROM json_each(cargo_specializations) WHERE value = 'electronics')"
    },
    {
        "category": "Structured",
        "query": "We need flatbed carriers in Texas (TX) with over 10 years of operations.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND years_operating > 10 AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed')"
    },
    {
        "category": "Structured",
        "query": "Find reefer carriers headquartered in Ohio (OH) with satisfactory safety ratings.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer')"
    },
    {
        "category": "Structured",
        "query": "Show me carriers in California (CA) equipped with flatbeds.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'flatbed')"
    },
    {
        "category": "Structured",
        "query": "We need tanker carriers in Pennsylvania (PA).",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'PA' AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'tanker')"
    },

    # --- Category 2: Qualitative / Jargon Queries (20 cases) ---
    {
        "category": "Qualitative",
        "query": "We need preloaded dry van swapping across large warehouse distribution hubs.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%drop-and-hook%'"
    },
    {
        "category": "Qualitative",
        "query": "Looking for refrigerated haulers that provide continuous sensor tracking of internal produce temperatures.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%pulp temp%'"
    },
    {
        "category": "Qualitative",
        "query": "Need certified pharma transport with backup cooling systems and chain-of-custody security seals.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%GDP-compliant%' OR notes LIKE '%dual reefer%'"
    },
    {
        "category": "Qualitative",
        "query": "Looking for certified chemical drivers outfitted with emergency containment gear for flammable liquids.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%Class 3%' OR notes LIKE '%spill kits%'"
    },
    {
        "category": "Qualitative",
        "query": "Carriers offering expedited team drivers and discreet route-tracking for high-value microchips.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%dual-driver%' OR notes LIKE '%covert GPS%'"
    },
    {
        "category": "Qualitative",
        "query": "Flatbed delivery to remote construction jobsites requiring a mounted forklift for self-unloading.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%Moffett forklift%'"
    },
    {
        "category": "Qualitative",
        "query": "Carriers capable of moving oversized heavy industrial equipment using detachable lowboy trailers.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%RGN lowboy%'"
    },
    {
        "category": "Qualitative",
        "query": "Dedicated logistics carriers supporting assembly plants with scheduled multi-stop round-the-clock rounds.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%milk-runs%' OR notes LIKE '%JIT%'"
    },
    {
        "category": "Qualitative",
        "query": "Final-mile retail carriers with rear power liftgates and electric pallet jacks for stores without loading docks.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%liftgate%' OR notes LIKE '%pallet jacks%'"
    },
    {
        "category": "Qualitative",
        "query": "Specialized heavy-haul carriers equipped with steerable dollies for extra-long structural infrastructure components.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%wind turbine%' OR notes LIKE '%superload%'"
    },
    {
        "category": "Qualitative",
        "query": "Bulk fluid transporters with insulated stainless steel tankers and integrated discharge pumps.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%rear pump-off%' OR notes LIKE '%liquid bulk%'"
    },
    {
        "category": "Qualitative",
        "query": "Harbor drayage truckers with federal security clearance credentials for ocean container haulage.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%TWIC badges%' OR notes LIKE '%drayage%'"
    },
    {
        "category": "Qualitative",
        "query": "Pest-free sanitized trailers with air-ride suspension for sensitive supermarket food distribution.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%food-grade%'"
    },
    {
        "category": "Qualitative",
        "query": "Specialized cold chain carriers for medical research samples requiring active data logger validation.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%clinical trial%' OR notes LIKE '%life sciences%'"
    },
    {
        "category": "Qualitative",
        "query": "Certified chemical haulers for toxic inhalation hazards and acid corrosives connected to 24/7 emergency response.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%Class 8%' OR notes LIKE '%Chemtrec%'"
    },
    {
        "category": "Qualitative",
        "query": "Delicate industrial machinery transport requiring shock-absorbing air ride and weatherized tarps.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%CNC equipment%' OR notes LIKE '%machinery tooling%'"
    },
    {
        "category": "Qualitative",
        "query": "Motor carriers specialized in high-priority assembly plant supply lines with narrow delivery appointments.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%tier-1 automotive%'"
    },
    {
        "category": "Qualitative",
        "query": "Open-deck flatbed drivers equipped to haul moisture-sensitive building materials like drywall and shingles.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%roofing shingles%' OR notes LIKE '%drop tarps%'"
    },
    {
        "category": "Qualitative",
        "query": "Less-than-truckload freight consolidation with real-time waypoint alerts and shipment status pings.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%GPS milestone tracking%'"
    },
    {
        "category": "Qualitative",
        "query": "Flatbed haulers featuring aluminum wide-spread tandem axles and wide winch straps for load securement.",
        "sql": "SELECT dot_number FROM carriers WHERE notes LIKE '%spread-axle%'"
    },

    # --- Category 3: Multi-Constraint Hybrid Queries (20 cases) ---
    {
        "category": "Hybrid",
        "query": "Midwest carriers specializing in sequenced automotive assembly parts.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%automotive%'"
    },
    {
        "category": "Hybrid",
        "query": "Southeast reefer carriers with automated pulp temperature logging.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer') AND notes LIKE '%pulp temp%'"
    },
    {
        "category": "Hybrid",
        "query": "Pacific Northwest carriers handling high-security semiconductor transport.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND notes LIKE '%semiconductor%'"
    },
    {
        "category": "Hybrid",
        "query": "North Carolina carriers equipped for permitted heavy-haul superloads.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NC' AND notes LIKE '%heavy-haul%'"
    },
    {
        "category": "Hybrid",
        "query": "Northeast carriers with hydraulic liftgates for urban retail deliveries.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND notes LIKE '%liftgate%'"
    },
    {
        "category": "Hybrid",
        "query": "Mountain region carriers certified for Class 3 flammable liquids transport.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mountain') AND notes LIKE '%Class 3%'"
    },
    {
        "category": "Hybrid",
        "query": "Ohio carriers with multi-temp reefers capable of sub-zero chilling.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND notes LIKE '%multi-temp%'"
    },
    {
        "category": "Hybrid",
        "query": "California carriers with TWIC credentials for port container drayage.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND notes LIKE '%TWIC%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest flatbed carriers with Moffett forklifts for construction sites.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%Moffett%'"
    },
    {
        "category": "Hybrid",
        "query": "Ohio carriers with GDP-compliant cold chain for pharmaceutical products.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND notes LIKE '%GDP-compliant%'"
    },
    {
        "category": "Hybrid",
        "query": "Southeast carriers with insulated stainless steel tankers for chemical liquids.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND notes LIKE '%stainless steel%'"
    },
    {
        "category": "Hybrid",
        "query": "Southwest dry van carriers operating drop-and-hook retail networks.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southwest') AND notes LIKE '%drop-and-hook%'"
    },
    {
        "category": "Hybrid",
        "query": "Florida carriers handling farm-to-cooler transport for seasonal citrus.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND notes LIKE '%citrus%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest heavy equipment carriers utilizing RGN lowboy trailers.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%RGN lowboy%'"
    },
    {
        "category": "Hybrid",
        "query": "Pennsylvania carriers offering food-grade van transport with satellite tracking.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'PA' AND notes LIKE '%food-grade%'"
    },
    {
        "category": "Hybrid",
        "query": "Pacific Northwest carriers certified for hazardous materials with Chemtrec monitoring.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND notes LIKE '%Chemtrec%'"
    },
    {
        "category": "Hybrid",
        "query": "Georgia carriers offering expedited retail store delivery with pallet jacks.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'GA' AND notes LIKE '%pallet jacks%'"
    },
    {
        "category": "Hybrid",
        "query": "Northeast carriers specializing in high-value electronic server racks.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND notes LIKE '%server racks%'"
    },
    {
        "category": "Hybrid",
        "query": "Texas flatbed haulers with aluminum spread-axles and heavy-duty straps.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND notes LIKE '%spread-axle%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest carriers transporting precision CNC machinery with air-ride.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%CNC%'"
    }
]

def resolve_ground_truth_targets():
    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        print("Warning: carriers.db not found. Run scripts/seed_db.py first.")
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

def run_bm25_search(query, k=5):
    candidates = retrieve_carriers_bm25(query, limit=k)
    return [c["dot_number"] for c in candidates]

def run_reranked_hybrid_search(query, k=5, force_cosine=False):
    bm25_cands = retrieve_carriers_bm25(query, limit=25)
    
    collection = get_chroma_collection()
    embed_model = get_embed_model()
    query_vector = embed_model.encode(query, convert_to_numpy=True).tolist()
    
    total_docs = collection.count()
    dense_cands = []
    if total_docs > 0:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(config.SEMANTIC_POOL_SIZE * 2, total_docs),
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
                
    fused = reciprocal_rank_fusion(bm25_cands, dense_cands, k=60, top_n=config.SEMANTIC_POOL_SIZE)
    if not fused:
        return []
        
    docs = [c["document"] for c in fused]
    metadatas = [c["metadata"] for c in fused]
    
    ranked = rerank_documents(
        query, docs, metadatas, top_k=k,
        query_embedding=query_vector,
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
        "FTS5 Lexical Search (BM25)": lambda case: run_bm25_search(case["query"]),
        "Reranked Search (Cosine)": lambda case: run_reranked_hybrid_search(case["query"], force_cosine=True),
        "Reranked Hybrid (Cross-Encoder)": lambda case: run_reranked_hybrid_search(case["query"], force_cosine=False)
    }
    
    results = {}
    category_results = {}
    for name in strategies:
        results[name] = {"r@1": [], "r@3": [], "r@5": [], "mrr": []}
        
    for case in EVAL_CASES:
        cat = case.get("category", "General")
        if cat not in category_results:
            category_results[cat] = {name: {"r@1": [], "r@3": [], "r@5": [], "mrr": []} for name in strategies}

        print(f"\nEvaluating [{cat}] Query: '{case['query']}'")
        for name, search_fn in strategies.items():
            retrieved = search_fn(case)
            r1, r3, r5, mrr = calculate_metrics(retrieved, case.get("targets", []))
            
            results[name]["r@1"].append(r1)
            results[name]["r@3"].append(r3)
            results[name]["r@5"].append(r5)
            results[name]["mrr"].append(mrr)

            category_results[cat][name]["r@1"].append(r1)
            category_results[cat][name]["r@3"].append(r3)
            category_results[cat][name]["r@5"].append(r5)
            category_results[cat][name]["mrr"].append(mrr)
            
            print(f"  - {name:<32} | Targets: {len(case.get('targets', [])):<3} | Retrieved: {len(retrieved):<2} | Recall@5: {r5:.1f} | MRR: {mrr:.3f}")
            
    # Print Per-Category Breakdown Tables
    for cat, cat_metrics in category_results.items():
        print(f"\n\n=== CATEGORY SUMMARY: {cat.upper()} ({len(cat_metrics[list(strategies.keys())[0]]['r@1'])} queries) ===")
        print(f"| {'Strategy':<32} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} |")
        print(f"| {'-'*32} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} |")
        for name, metrics in cat_metrics.items():
            if not metrics["r@1"]:
                continue
            avg_r1 = sum(metrics["r@1"]) / len(metrics["r@1"])
            avg_r3 = sum(metrics["r@3"]) / len(metrics["r@3"])
            avg_r5 = sum(metrics["r@5"]) / len(metrics["r@5"])
            avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
            print(f"| {name:<32} | {avg_r1:<10.3f} | {avg_r3:<10.3f} | {avg_r5:<10.3f} | {avg_mrr:<8.3f} |")

    # Print Overall Summary Table
    print(f"\n\n=== OVERALL RETRIEVAL METRICS SUMMARY ({len(EVAL_CASES)} queries) ===")
    print(f"| {'Strategy':<32} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} |")
    print(f"| {'-'*32} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} |")
    
    for name, metrics in results.items():
        if not metrics["r@1"]: # Skip if empty
            continue
        avg_r1 = sum(metrics["r@1"]) / len(metrics["r@1"])
        avg_r3 = sum(metrics["r@3"]) / len(metrics["r@3"])
        avg_r5 = sum(metrics["r@5"]) / len(metrics["r@5"])
        avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
        print(f"| {name:<32} | {avg_r1:<10.3f} | {avg_r3:<10.3f} | {avg_r5:<10.3f} | {avg_mrr:<8.3f} |")
        
    print("\n=== Evaluation Harness Complete ===")


if __name__ == "__main__":
    main()

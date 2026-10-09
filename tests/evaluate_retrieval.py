import os
import sqlite3
import sys

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.retriever import get_chroma_collection, retrieve_carriers_bm25, reciprocal_rank_fusion
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
        "query": "Motor carriers covering the Midwest experienced in synchronized parts logistics for automobile assembly plants.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%automotive%'"
    },
    {
        "category": "Hybrid",
        "query": "Refrigerated freight haulers in the Southeast offering continuous produce probe readings and precooling for berry shipments.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND EXISTS (SELECT 1 FROM json_each(equipment_types) WHERE value = 'reefer') AND notes LIKE '%pulp temp%'"
    },
    {
        "category": "Hybrid",
        "query": "Pacific Northwest trucking providers equipped for secure locked-door transit of microchips and delicate processor components.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND notes LIKE '%semiconductor%'"
    },
    {
        "category": "Hybrid",
        "query": "North Carolina trucking companies capable of transporting permitted massive infrastructure pieces with police escorts.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'NC' AND (notes LIKE '%superload%' OR notes LIKE '%police escorts%')"
    },
    {
        "category": "Hybrid",
        "query": "Northeast freight carriers operating straight trucks with power tailgates for tight downtown storefront deliveries without a loading dock.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND notes LIKE '%liftgate%'"
    },
    {
        "category": "Hybrid",
        "query": "Mountain state freight companies authorized to carry placarded combustible and flammable fluids with onboard containment kits.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Mountain') AND notes LIKE '%Class 3%'"
    },
    {
        "category": "Hybrid",
        "query": "Ohio cold storage haulers running dual-compartment trailers capable of sustaining deep-freeze conditions below zero.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND notes LIKE '%multi-temp%'"
    },
    {
        "category": "Hybrid",
        "query": "California container drayage fleets holding maritime terminal identification cards for ocean pier pickup.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'CA' AND notes LIKE '%TWIC%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest flatbed trucking providers equipped with truck-mounted piggyback forklifts for direct offloading at active building sites.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%Moffett%'"
    },
    {
        "category": "Hybrid",
        "query": "Ohio refrigerated carriers maintaining validated temperature records under good distribution practice standards for pharmaceuticals.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'OH' AND notes LIKE '%GDP-compliant%'"
    },
    {
        "category": "Hybrid",
        "query": "Southeast bulk haulers with thermal-jacketed non-corrosive alloy tank trailers for industrial liquid chemicals.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southeast') AND notes LIKE '%stainless steel%'"
    },
    {
        "category": "Hybrid",
        "query": "Southwest dry van logistics operators supporting rapid preloaded trailer swaps at high-throughput distribution hubs.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Southwest') AND notes LIKE '%drop-and-hook%'"
    },
    {
        "category": "Hybrid",
        "query": "Florida refrigerated fleets providing direct orchard-to-packing-house distribution for fresh orange and grapefruit harvests.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'FL' AND notes LIKE '%citrus%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest specialized heavy haulers utilizing detachable gooseneck lowbed trailers for moving yellow-iron excavators.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Midwest') AND notes LIKE '%RGN lowboy%'"
    },
    {
        "category": "Hybrid",
        "query": "Wisconsin carriers providing sanitary dry box trailers monitored via real-time satellite telemetry for grocery distribution.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'WI' AND notes LIKE '%food-grade%'"
    },
    {
        "category": "Hybrid",
        "query": "Pacific Northwest hazmat haulers linked to around-the-clock chemical emergency dispatch and incident management support.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Pacific Northwest') AND notes LIKE '%Chemtrec%'"
    },
    {
        "category": "Hybrid",
        "query": "Georgia freight companies providing rush commercial store delivery equipped with manual pump trucks to move skids inside.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'GA' AND notes LIKE '%pallet jacks%'"
    },
    {
        "category": "Hybrid",
        "query": "Northeast freight lines providing high-security locked transport for enterprise datacenter computing cabinets.",
        "sql": "SELECT dot_number FROM carriers WHERE EXISTS (SELECT 1 FROM json_each(service_regions) WHERE value = 'Northeast') AND notes LIKE '%server racks%'"
    },
    {
        "category": "Hybrid",
        "query": "Texas flatbed trucking companies utilizing lightweight split-tandem trailers and heavy ratchet tie-downs for securing cargo.",
        "sql": "SELECT dot_number FROM carriers WHERE hq_state = 'TX' AND notes LIKE '%spread-axle%'"
    },
    {
        "category": "Hybrid",
        "query": "Midwest air-cushioned freight haulers specialized in shock-sensitive computer numerical control milling equipment.",
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
    """
    Computes Hit@k (Success@k) and MRR (Mean Reciprocal Rank).
    Hit@k evaluates whether at least one qualified carrier appears in the top-k results.
    """
    if not targets:
        return 0.0, 0.0, 0.0, 0.0
    hit_1 = 1.0 if any(t in retrieved_list[:1] for t in targets) else 0.0
    hit_3 = 1.0 if any(t in retrieved_list[:3] for t in targets) else 0.0
    hit_5 = 1.0 if any(t in retrieved_list[:5] for t in targets) else 0.0
    
    mrr = 0.0
    for idx, item in enumerate(retrieved_list):
        if item in targets:
            mrr = 1.0 / (idx + 1)
            break
            
    return hit_1, hit_3, hit_5, mrr

def main():
    import json
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
        results[name] = {"h@1": [], "h@3": [], "h@5": [], "mrr": []}
        
    for case in EVAL_CASES:
        cat = case.get("category", "General")
        if cat not in category_results:
            category_results[cat] = {name: {"h@1": [], "h@3": [], "h@5": [], "mrr": []} for name in strategies}

        print(f"\nEvaluating [{cat}] Query: '{case['query']}'")
        for name, search_fn in strategies.items():
            retrieved = search_fn(case)
            h1, h3, h5, mrr = calculate_metrics(retrieved, case.get("targets", []))
            
            results[name]["h@1"].append(h1)
            results[name]["h@3"].append(h3)
            results[name]["h@5"].append(h5)
            results[name]["mrr"].append(mrr)

            category_results[cat][name]["h@1"].append(h1)
            category_results[cat][name]["h@3"].append(h3)
            category_results[cat][name]["h@5"].append(h5)
            category_results[cat][name]["mrr"].append(mrr)
            
            print(f"  - {name:<32} | Targets: {len(case.get('targets', [])):<3} | Retrieved: {len(retrieved):<2} | Hit@5: {h5:.1f} | MRR: {mrr:.3f}")
            
    # Print Per-Category Breakdown Tables
    category_summary = {}
    for cat, cat_metrics in category_results.items():
        print(f"\n\n=== CATEGORY SUMMARY: {cat.upper()} ({len(cat_metrics[list(strategies.keys())[0]]['h@1'])} queries) ===")
        print(f"| {'Strategy':<32} | {'Hit@1':<10} | {'Hit@3':<10} | {'Hit@5':<10} | {'MRR':<8} |")
        print(f"| {'-'*32} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} |")
        category_summary[cat] = {}
        for name, metrics in cat_metrics.items():
            if not metrics["h@1"]:
                continue
            avg_h1 = sum(metrics["h@1"]) / len(metrics["h@1"])
            avg_h3 = sum(metrics["h@3"]) / len(metrics["h@3"])
            avg_h5 = sum(metrics["h@5"]) / len(metrics["h@5"])
            avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
            category_summary[cat][name] = {
                "hit@1": round(avg_h1, 3), "hit@3": round(avg_h3, 3),
                "hit@5": round(avg_h5, 3), "mrr": round(avg_mrr, 3)
            }
            print(f"| {name:<32} | {avg_h1:<10.3f} | {avg_h3:<10.3f} | {avg_h5:<10.3f} | {avg_mrr:<8.3f} |")

    # Print Overall Summary Table
    print(f"\n\n=== OVERALL RETRIEVAL METRICS SUMMARY ({len(EVAL_CASES)} queries) ===")
    print(f"| {'Strategy':<32} | {'Hit@1':<10} | {'Hit@3':<10} | {'Hit@5':<10} | {'MRR':<8} |")
    print(f"| {'-'*32} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} |")
    
    overall_summary = {}
    for name, metrics in results.items():
        if not metrics["h@1"]:
            continue
        avg_h1 = sum(metrics["h@1"]) / len(metrics["h@1"])
        avg_h3 = sum(metrics["h@3"]) / len(metrics["h@3"])
        avg_h5 = sum(metrics["h@5"]) / len(metrics["h@5"])
        avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"])
        overall_summary[name] = {
            "hit@1": round(avg_h1, 3), "hit@3": round(avg_h3, 3),
            "hit@5": round(avg_h5, 3), "mrr": round(avg_mrr, 3)
        }
        print(f"| {name:<32} | {avg_h1:<10.3f} | {avg_h3:<10.3f} | {avg_h5:<10.3f} | {avg_mrr:<8.3f} |")
        
    print("\nNote: Hit@k (Success@k) measures whether >=1 valid carrier is discovered in top-k.")
    
    # Export dynamic results payload for chart generation and CI verification
    benchmark_payload = {
        "overall": overall_summary,
        "categories": category_summary,
        "total_queries": len(EVAL_CASES)
    }
    benchmark_export_path = os.path.join(config.DATA_DIR, "benchmark_results.json")
    try:
        os.makedirs(os.path.dirname(benchmark_export_path), exist_ok=True)
        with open(benchmark_export_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_payload, f, indent=2)
        print(f"[OK] Exported benchmark results to {benchmark_export_path}")
    except Exception as e:
        print(f"[WARNING] Failed to export benchmark results JSON: {e}")

    # Enforce quality gate assertions
    sqlite_h1 = overall_summary["SQLite Exact Query"]["hit@1"]
    assert sqlite_h1 >= 0.98, f"SQLite Exact Query Hit@1 regression: {sqlite_h1:.3f}"

    hybrid_h1 = overall_summary["Reranked Hybrid (Cross-Encoder)"]["hit@1"]
    assert hybrid_h1 >= 0.65, f"Cross-Encoder Hit@1 regression: {hybrid_h1:.3f}"

    hybrid_h5 = overall_summary["Reranked Hybrid (Cross-Encoder)"]["hit@5"]
    assert hybrid_h5 >= 0.80, f"Cross-Encoder Hit@5 regression: {hybrid_h5:.3f}"

    hybrid_mrr = overall_summary["Reranked Hybrid (Cross-Encoder)"]["mrr"]
    assert hybrid_mrr >= 0.70, f"Cross-Encoder MRR regression: {hybrid_mrr:.3f}"

    print("\n[PASSED] All retrieval benchmark thresholds verified successfully!")
    print("=== Evaluation Harness Complete ===")

if __name__ == "__main__":
    main()

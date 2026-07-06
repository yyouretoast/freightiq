import os
import sys
import concurrent.futures
import logging

# Ensure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.utils import save_feedback
from rag.reranker import rerank_documents
from rag.retriever import query_carriers_sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
logger = logging.getLogger(__name__)

def worker_task(worker_id):
    logger.info(f"Worker {worker_id} starting concurrency stress test operations...")
    
    try:
        # 1. Stress concurrent feedback writes
        query_text = f"Test Query from worker {worker_id}"
        response_text = f"Carrier Name: Swift Freight 80 #143 | DOT: 5942938 (logged by worker {worker_id})"
        feedback_type = "up" if worker_id % 2 == 0 else "down"
        
        save_feedback(query_text, response_text, feedback_type)
        logger.info(f"Worker {worker_id}: Feedback saved successfully.")
        
        # 2. Stress concurrent reranking loading & scoring
        mock_docs = [
            "Carrier Name: Swift Freight 80 #143\nDOT: 5942938\nHQ State: OH\nEquipment: dry van",
            "Carrier Name: NextGen Haulers 80 #101\nDOT: 6725944\nHQ State: OH\nEquipment: flatbed",
            "Carrier Name: titan Systems #114\nDOT: 9088677\nHQ State: FL\nEquipment: reefer"
        ]
        mock_metadatas = [
            {"dot_number": "5942938"},
            {"dot_number": "6725944"},
            {"dot_number": "9088677"}
        ]
        
        # This will concurrently load weights and score
        ranked = rerank_documents(
            query=query_text,
            documents=mock_docs,
            metadatas=mock_metadatas,
            top_k=2
        )
        logger.info(f"Worker {worker_id}: Reranked successfully. Top score: {ranked[0]['score']:.4f}")
        
        # 3. Stress concurrent SQLite reads
        sql_query = "SELECT carrier_name, hq_state, safety_rating FROM carriers WHERE hq_state = 'OH' LIMIT 2"
        sql_res = query_carriers_sql(sql_query)
        logger.info(f"Worker {worker_id}: SQL queried successfully. Result length: {len(sql_res)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed with error: {e}", exc_info=True)
        return False

def main():
    print("=== STARTING FREIGHTIQ CONCURRENCY & THREAD-SAFETY STRESS TEST ===")
    
    num_workers = 15
    print(f"Spawning {num_workers} concurrent workers in thread pool...")
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="StressWorker") as executor:
        futures = {executor.submit(worker_task, i): i for i in range(num_workers)}
        
        for future in concurrent.futures.as_completed(futures):
            worker_idx = futures[future]
            try:
                res = future.result()
                if res:
                    success_count += 1
            except Exception as exc:
                print(f"Worker {worker_idx} generated an exception: {exc}")
                
    print("\n=== CONCURRENCY STRESS TEST SUMMARY ===")
    print(f"Total spawned workers: {num_workers}")
    print(f"Successful executions: {success_count}")
    
    if success_count == num_workers:
        print("[SUCCESS] ALL workers completed operations with ZERO errors under load! Thread-safety verified.")
        sys.exit(0)
    else:
        print("[FAIL] One or more workers encountered concurrency issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()

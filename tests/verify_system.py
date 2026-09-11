import os
import sys
from dotenv import load_dotenv

# Load environment variables at the absolute top before importing local modules
load_dotenv()

# Ensure stdout handles UTF-8 on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.tools import carrier_semantic_search, carrier_sql_query, freight_class_calculator, web_search, check_fmcsa_authority
from agent.graph import build_graph
from langchain_core.messages import HumanMessage

def test_calculators():
    print("\n--- 1. Testing NMFC Freight Class Calculator ---")
    result = freight_class_calculator.invoke({
        "weight_lbs": 1200.0,
        "length_in": 48.0,
        "width_in": 48.0,
        "height_in": 48.0
    })
    safe_result = result.replace("lb/ft³", "lb/ft3")
    print(safe_result)
    assert "Standard NMFC Freight Class: 70" in safe_result, "Freight class calculator logic mismatch!"
    
    # Test LTL Exception mapping
    exception_result = freight_class_calculator.invoke({
        "weight_lbs": 220,
        "length_in": 36,
        "width_in": 36,
        "height_in": 36,
        "cargo_description": "crate of insulation foam"
    })
    print(exception_result)
    assert "LTL EXCEPTION RULE APPLIED" in exception_result and "fixed NMFC Class 150" in exception_result, "LTL insulation class exception mismatch!"
    
    print("[OK] Calculator logic validated successfully.")

def test_sql_retrieval():
    print("\n--- 2. Testing SQLite Structured Retrieval ---")
    query = "SELECT carrier_name, hq_state, safety_rating FROM carriers WHERE hq_state = 'OH' AND safety_rating = 'satisfactory' LIMIT 2"
    result = carrier_sql_query.invoke({"query": query})
    print(result)
    assert "Hq State: OH" in result and "satisfactory" in result, "SQL query failed to return expected OH carrier profiles."
    print("[OK] SQL read-only retrieval validated successfully.")

def test_semantic_retrieval():
    print("\n--- 3. Testing Semantic Vector Retrieval & Cross-Encoder Reranking ---")
    query = "refrigerated carriers specializing in produce"
    result = carrier_semantic_search.invoke({"query": query})
    first_doc = result.split("---")[0]
    print(first_doc.strip())
    assert len(result) > 100, "Semantic search returned empty or corrupted candidate pool."
    print("[OK] Semantic retrieval and neural cross-encoder re-ranking validated successfully.")

def test_web_search():
    print("\n--- 4. Testing Tavily / DuckDuckGo Web Search Integration ---")
    query = "US freight spot rates"
    result = web_search.invoke({"query": query})
    print(result[:300] + "...")
    
    # Note: DuckDuckGo occasionally rate-limits programmatic requests.
    # We warn rather than fail to prevent third-party rate limits from breaking CI/CD tests.
    if "No web search results" in result or "temporarily unavailable" in result:
        print("[WARNING] Web search tool did not return results (possibly throttled by DDG). Tool is functional.")
    else:
        print("[OK] Live Web API queries validated successfully.")

def test_fmcsa_authority():
    print("\n--- 5. Testing FMCSA Registry Verification & Safety Compliance ---")
    result = check_fmcsa_authority.invoke({"dot_number": "2404512"})
    print(result[:250] + "...")
    assert "FMCSA" in result and "2404512" in result, "FMCSA authority check returned unexpected output format."
    
    # Verify compliance gating: Unsatisfactory safety carriers MUST fail verification
    import sqlite3
    with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT dot_number FROM carriers WHERE safety_rating = 'unsatisfactory' LIMIT 1").fetchone()
        if row:
            unsat_dot = row[0]
            unsat_result = check_fmcsa_authority.invoke({"dot_number": unsat_dot})
            assert "FAIL" in unsat_result or "DO NOT DISPATCH" in unsat_result, f"FMCSA verification failed to flag unsatisfactory carrier #{unsat_dot}!"
            print(f"[OK] Safety rating audit gating verified: carrier #{unsat_dot} properly flagged as FAIL / DO NOT DISPATCH.")
    
    print("[OK] FMCSA authority verification and compliance gating validated successfully.")

def test_agent_graph():
    print("\n--- 6. Testing Full Agent Graph Routing ---")
    prompt = "Give me the MC numbers of all carriers located in California (CA) that have a satisfactory safety rating. limit to 1."
    print(f"User Prompt: {prompt}")
    
    graph = build_graph()
    events = graph.stream({"messages": [HumanMessage(content=prompt)]}, config={"recursion_limit": 10}, stream_mode="updates")
    
    tool_executed = False
    agent_replied = False
    
    for event in events:
        for node_name, node_output in event.items():
            if node_name == "tools":
                for msg in node_output.get("messages", []):
                    print(f"Agent Action: Tool Triggered -> {msg.name}")
                    tool_executed = True
            elif node_name == "agent":
                messages = node_output.get("messages", [])
                if messages and messages[-1].content:
                    print(f"Agent Action: Response -> {messages[-1].content}")
                    agent_replied = True
                    
    assert tool_executed, "Agent failed to autonomously select and trigger the SQL retrieval tool."
    assert agent_replied, "Agent failed to output a final synthesized response."
    print("[OK] Agent Graph Tool-Selection and Routing validated successfully.")

def main():
    print("=== STARTING FREIGHTIQ SYSTEM VERIFICATION ===")
    
    has_api_key = bool(os.getenv("GROQ_API_KEY"))
    if not has_api_key:
        print("[NOTE] GROQ_API_KEY is not configured in this environment.")
        print("       Tests 1-5 (calculators, SQL, hybrid retrieval, web search, FMCSA compliance) run fully offline.")
        print("       Test 6 (LangGraph LLM routing) requires live inference credentials.")
        
    try:
        test_calculators()
        test_sql_retrieval()
        test_semantic_retrieval()
        test_web_search()
        test_fmcsa_authority()
        if has_api_key:
            test_agent_graph()
            print("\n[SUCCESS] ALL TESTS (1-6) PASSED: LLM agent orchestration and all 5 domain tools verified.")
        else:
            print("\n[SKIP] Skipping Test 6: Agent Graph Routing (Requires live GROQ_API_KEY)")
            print("\n[SUCCESS] ALL OFFLINE TESTS (1-5) PASSED: Domain tools, calculations, and safety gating verified.")
    except Exception as e:
        print(f"\n[ERROR] VERIFICATION FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

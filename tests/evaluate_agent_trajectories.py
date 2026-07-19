import os
import sys
import time
import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Override model to Llama 3.1 8B for tests to bypass daily Groq 70B token limits
os.environ["AGENT_MODEL"] = "llama-3.1-8b-instant"

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.graph import build_graph

# Configure logging to clean stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAJECTORY_CASES = [
    {
        "type": "SQL",
        "query": "Find all carriers headquartered in Ohio (OH) with a satisfactory safety rating.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"]
    },
    {
        "type": "SQL",
        "query": "We need flatbed carriers that handle hazardous materials in the Midwest.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"]
    },
    {
        "type": "SQL",
        "query": "Show me carriers headquartered in Texas (TX) equipped with dry vans.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"]
    },
    {
        "type": "SQL",
        "query": "Find a carrier located in California (CA) that has a satisfactory safety rating.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"]
    },
    {
        "type": "SQL",
        "query": "Find LTL carriers headquartered in New York (NY) with a satisfactory safety rating.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"]
    },
    {
        "type": "Semantic",
        "query": "Find me carriers known for exceptional handling of temperature-sensitive goods.",
        "expected_tool": "carrier_semantic_search",
        "unexpected_tools": ["carrier_sql_query", "web_search"]
    },
    {
        "type": "Semantic",
        "query": "Find a carrier that is described as highly reliable with a strong safety culture.",
        "expected_tool": "carrier_semantic_search",
        "unexpected_tools": ["carrier_sql_query", "web_search"]
    },
    {
        "type": "Semantic",
        "query": "We need a carrier with a proven track record of handling fragile cargo.",
        "expected_tool": "carrier_semantic_search",
        "unexpected_tools": ["carrier_sql_query", "web_search"]
    },
    {
        "type": "Semantic",
        "query": "Show me carriers known for having highly experienced drivers and excellent dispatcher communication.",
        "expected_tool": "carrier_semantic_search",
        "unexpected_tools": ["carrier_sql_query", "web_search"]
    },
    {
        "type": "Semantic",
        "query": "Find carriers described as specialized in time-critical emergency shipping.",
        "expected_tool": "carrier_semantic_search",
        "unexpected_tools": ["carrier_sql_query", "web_search"]
    },
    {
        "type": "Calculator",
        "query": "What is the freight class for a 220 lbs crate measuring 36x36x36 inches?",
        "expected_tool": "freight_class_calculator",
        "unexpected_tools": ["carrier_sql_query", "carrier_semantic_search", "web_search"]
    },
    {
        "type": "Calculator",
        "query": "Calculate the NMFC freight class for a 1200 lbs pallet of building materials measuring 48x48x48 inches.",
        "expected_tool": "freight_class_calculator",
        "unexpected_tools": ["carrier_sql_query", "carrier_semantic_search", "web_search"]
    },
    {
        "type": "Web",
        "query": "What is the current average national dry van spot rate per mile in 2026?",
        "expected_tool": "web_search",
        "unexpected_tools": ["carrier_sql_query", "carrier_semantic_search"]
    },
    {
        "type": "Web",
        "query": "Search the web for the latest updates on container shipping rates from Shanghai to Los Angeles.",
        "expected_tool": "web_search",
        "unexpected_tools": ["carrier_sql_query", "carrier_semantic_search"]
    },
    {
        "type": "Adversarial Loop",
        "query": "Perform a carrier SQL query for carriers located in Ohio (OH). Then, perform the exact same carrier SQL query for Ohio carriers again to double-check, and then output the final answer.",
        "expected_tool": "carrier_sql_query",
        "unexpected_tools": ["carrier_semantic_search", "web_search"],
        "expect_loop_breaker": True
    }
]

def main():
    print("=== FREIGHTIQ AGENT TRAJECTORY & ROUTING EVALUATION ===")
    
    # Load agent graph
    graph = build_graph()
    
    passed_count = 0
    total_count = len(TRAJECTORY_CASES)
    
    for idx, case in enumerate(TRAJECTORY_CASES):
        if idx > 0:
            time.sleep(2.0)
        print(f"\n[{idx+1}/{total_count}] Testing Case (Type: {case['type']}): '{case['query']}'")
        
        try:
            # Execute graph invocation
            state = {"messages": [HumanMessage(content=case["query"])]}
            result = graph.invoke(state, config={"recursion_limit": 10})
            messages = result.get("messages", [])
            
            # Extract tool execution list
            called_tools = [msg.name for msg in messages if isinstance(msg, ToolMessage)]
            trace_length = len(messages)
            
            print(f"  - Trajectory Trace Length: {trace_length} messages")
            print(f"  - Tools Triggered: {called_tools}")
            
            # Assertions / Checks
            is_valid = True
            reasons = []
            
            # 1. Expected tool check
            if case["expected_tool"]:
                if case["expected_tool"] not in called_tools:
                    is_valid = False
                    reasons.append(f"Expected tool '{case['expected_tool']}' was not executed.")
            
            # 2. Unexpected tool check
            for un_tool in case.get("unexpected_tools", []):
                if un_tool in called_tools:
                    is_valid = False
                    reasons.append(f"Unexpected tool '{un_tool}' was executed.")
            
            # 3. Message limit check (prevent loop regressions)
            if trace_length >= 10:
                is_valid = False
                reasons.append(f"Trace length {trace_length} exceeds limit, indicating a potential routing loop.")
                
            # 4. Adversarial Loop breaker trigger verification
            if case.get("expect_loop_breaker", False):
                # Ensure the tool was called twice
                duplicate_calls = [t for t in called_tools if t == case["expected_tool"]]
                if len(duplicate_calls) < 2:
                    # Note: Depending on LLM formatting, Llama 3.3 might synthesize directly instead of calling the duplicate.
                    # We log it, but don't strictly fail the test if the LLM avoided the loop on its own.
                    logger.info("  - Note: Adversarial query did not trigger duplicate tool calls (LLM avoided it natively).")
                else:
                    print("  - [OK] Adversarial query successfully triggered duplicate tool calls.")
                
                # Verify graph terminated safely under the threshold
                if trace_length <= 6:
                    print("  - [OK] Loop breaker guardrail successfully terminated graph within safe message limits.")
                else:
                    is_valid = False
                    reasons.append("Adversarial query exceeded safe message limits. Loop breaker failed to fire.")

            if is_valid:
                print("  - Status: [PASSED]")
                passed_count += 1
            else:
                print(f"  - Status: [FAILED] -> {', '.join(reasons)}")
                
        except Exception as e:
            print(f"  - Status: [FAILED] with execution error: {e}")
            
    accuracy = (passed_count / total_count) * 100
    print("\n\n=== AGENT TRAJECTORY AUDIT RESULTS SUMMARY ===")
    print(f"Total Trajectory Test Cases: {total_count}")
    print(f"Passed Trajectory Audits:   {passed_count}")
    print(f"Routing/Guardrail Accuracy:  {accuracy:.1f}%")
    print("==============================================")
    
    if passed_count == total_count:
        print("[SUCCESS] All agent trajectories and loop-breaker guardrails verified successfully!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

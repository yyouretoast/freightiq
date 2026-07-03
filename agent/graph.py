from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from agent.state import AgentState
from agent.nodes import agent_node, tool_node

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

graph = build_graph()

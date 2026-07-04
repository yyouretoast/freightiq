from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import agent_node, tool_node
from langgraph.prebuilt import tools_condition

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set Entry Point
    workflow.set_entry_point("agent")

    # Add Conditional Edges
    workflow.add_conditional_edges("agent", tools_condition)

    # Add Standard Transitions
    workflow.add_edge("tools", "agent")

    # Compile the workflow graph
    return workflow.compile()

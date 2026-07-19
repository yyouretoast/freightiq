from langgraph.graph import StateGraph
from agent.state import AgentState
from agent.nodes import agent_node, tool_node
from langgraph.prebuilt import tools_condition

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

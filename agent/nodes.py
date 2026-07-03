import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.tools import tools

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
else:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

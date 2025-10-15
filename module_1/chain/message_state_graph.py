"""Entry-point for the message state graph.

Ensure the project root is on sys.path so sibling packages like `common`
can be imported when this file is executed directly from the `chain`
directory. This needs to run before any imports that reference those
packages.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]  # two levels up from chain/
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from langgraph.graph import StateGraph, START, END
from message_state import MessagesState
from tools import create_tool, multiply, add, divide
from common.image_display import display_image
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

class MessagesState(MessagesState):
    pass

llm_with_tools = create_tool()

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # it will look at the output of tool_calling_llm, the last message from assistant, and:
    # if the result is a tool call -> routes to tools
    # if the result is not a tool call -> routes to END
    tools_condition
)
builder.add_edge("tools", END)
graph = builder.compile()
    
# display_image(graph)

# messages = graph.invoke({"messages": HumanMessage(content="Help!")})
# print(messages)

messages = graph.invoke({"messages": HumanMessage(content="Multiple 3 and 8.")})
for m in messages["messages"]:
    m.pretty_print()
    
    
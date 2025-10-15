import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]  # two levels up from chain/
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
 
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from message_state import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from tools import multiply, add, divide
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

class MessagesState(MessagesState):
    pass

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge("tools", "assistant")

### TO ADD Memory:
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()
    
messages = [HumanMessage(content="Multiply that by 5.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages["messages"]:
    m.pretty_print()


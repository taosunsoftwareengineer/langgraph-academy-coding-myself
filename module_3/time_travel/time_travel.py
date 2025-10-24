import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]  # two levels up from chain/
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv("../../.env")

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    
        Args:
            a: first int
            b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divides a by b.
    
        Args:
            a: first int
            b: second int
    """
    return a / b

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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
    
initial_input = {"messages": "Multiply 2 and 3"}
thread = {"configurable": {"thread_id": "1"}}
    
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
    
# state = graph.get_state(thread)
# print(state)

print("\nStart reply\n")

all_states = [s for s in graph.get_state_history(thread)]
# print(len(all_states))

to_reply = all_states[-2]
print(to_reply)
print(to_reply.values)
print(to_reply.next)
print(to_reply.config)

print("\nReply Now!\n")

for event in graph.stream(None, to_reply.config, stream_mode="values"):
    event["messages"][-1].pretty_print()
    
print("\nStart fork\n")
to_fork = all_states[-2]

fork_config = graph.update_state(
    to_fork.config, 
    {"messages": [HumanMessage(content="Multiply 3 and 5", id=to_fork.values["messages"][0].id)]}
)

print("\Fork Now!\n")
print(fork_config) # this will produce a new checkpoint id

all_states = [s for s in graph.get_state_history(thread)]

print(all_states[0].values["messages"])

# next continue running
for event in graph.stream(None, fork_config, stream_mode="values"):
    event["messages"][-1].pretty_print()


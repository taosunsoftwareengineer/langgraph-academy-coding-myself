from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from operator import add
from typing import Annotated

class State(TypedDict):
    foo: Annotated[list[int], add]
    
def node_1(state):
    print("---Node 1---")
    # Return a list to match the reducer's expectation
    return {"foo": [state["foo"][0] + 1]}

builder = StateGraph(State)
builder.add_node("node_1", node_1)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()

# Initialize with a list containing the integer
result = graph.invoke({"foo": [1]})
print(result)
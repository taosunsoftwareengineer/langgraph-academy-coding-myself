from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int
    
class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state["foo"] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return{"foo": state["baz"] + 1}

builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()

result = graph.invoke({"foo": 1})
print(result)

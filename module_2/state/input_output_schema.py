from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class InputState(TypedDict):
    question: str
    
class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question:str
    answer:str
    notes:str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

result = graph.invoke({"question":"hi"})
print(result)


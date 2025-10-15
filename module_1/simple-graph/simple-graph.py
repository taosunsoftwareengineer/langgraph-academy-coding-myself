from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from state_node import State, node_1, node_2, node_3
from edge import decide_mood
import os
from common.image_display import display_image

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

#display_image(graph)

result = graph.invoke({"graph_state": "Hi, this is Tao."})
print(result)

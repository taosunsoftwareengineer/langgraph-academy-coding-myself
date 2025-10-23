from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv
load_dotenv("../../.env")

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state["messages"][-1:])} # invoke only the last message

messages = [AIMessage(f"So you said you were searcing ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Lance"))


builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

output = graph.invoke({"messages": messages})

messages.append(output["messages"][-1])
messages.append(HumanMessage(f"Tell me more about Narwhals", name="Lance"))

for m in output["messages"]:
    m.pretty_print()

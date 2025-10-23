from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio

from dotenv import load_dotenv
load_dotenv("../../.env")

class State(MessagesState):
    summary: str

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def call_model(state: State):
    summary = state.get("summary", "")

    # If there is summary, then add it
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"

        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
        
    response = model.invoke(messages)
    return {"messages": response}

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)

workflow.add_edge(START, "conversation")
workflow.add_edge("conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

async def main():
    print("Streming events")
    config = {"configurable": {"thread_id": "5"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        print(f"Node: {event['metadata'].get('langgraph_node','')}. Type:{event['event']}. Name: {event['name']}")
        
    print("***"*25)
    
    print("Streming tokens")
    node_to_stream = "conversation"
    config = {"configurable": {"thread_id": "6"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        # Get chat model tokens from a particular node
        if event["event"] == "on_chat_model_stream" and event["metadata"].get("langgraph_node", "") == node_to_stream:
            print(event["data"])
            
    print("Streming chunks")
    node_to_stream = "conversation"
    config = {"configurable": {"thread_id": "7"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        # Get chat model tokens from a particular node
        if event["event"] == "on_chat_model_stream" and event["metadata"].get("langgraph_node", "") == node_to_stream:
            data = event["data"]
            print(data["chunk"].content, end="|")
        
if __name__ == "__main__":
    asyncio.run(main())
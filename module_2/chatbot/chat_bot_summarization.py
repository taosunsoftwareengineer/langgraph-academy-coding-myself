from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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

def summarize_conversation(state: State):
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Entend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State):
    """Return the next node to execute."""

    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    
    return END

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Next, start the conversation now:

config = {"configurable": {"thread_id": "1"}}

input_message = HumanMessage(content="Hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="What's my name?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()
    
input_message = HumanMessage(content="I like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()
 
 
summary = graph.get_state(config).values.get("summary","")
print(f"\nSummary: {summary}")
  
input_message = HumanMessage(content="I like Nick Bosa, isn't he the highest paid defensive player?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()  
    
summary = graph.get_state(config).values.get("summary","")
print(f"\nSummary: {summary}")
  
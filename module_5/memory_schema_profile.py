from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing import TypedDict, List
from config import llm

from dotenv import load_dotenv
load_dotenv("../.env")

# chatbot instruction
MODEL_SYSTEM_PROMPT = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# create new memory from the chat history and existing memory
CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your response

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
    - Personal details (name, location)
    - Preferences (likes, dislikes)
    - Interests and hobbies
    - Past experiences
    - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

class UserProfile(TypedDict):
    """User profile schema with typed fileds"""
    user_name: str
    interests: list[str]

user_profile: UserProfile = {
    "user_name": "Lance",
    "interests": ["biking", "technology", "coffee"]
}

model_with_structure = llm.with_structured_output(UserProfile)


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response"""

    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    
    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    system_msg = MODEL_SYSTEM_PROMPT.format(memory=existing_memory_content)
    
    response = llm.invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}
        
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and save a memory to the store."""

    user_id = config["configurable"]["user_id"]
    
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)
    
    # extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model_with_structure.invoke([SystemMessage(content=system_msg)] + state["messages"])

    store.put(namespace, key, {"memory": new_memory})

# build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

across_thread_memory = InMemoryStore()

within_thread_memory = MemorySaver()

graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# run it
input_messages = [HumanMessage(content="Hi, my name is Lance and I like to bike around San Francisco.")]

for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

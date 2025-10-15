from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages

# The below two classes are EQUIVALENT!

class CustomMessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    
class ExtendedMessagesState(MessagesState):
    # MessagesState already has built-in messages
    added_key_1: str
    added_key_2: str

initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                    ]

new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

add_messages(initial_messages, new_message)

# Replacing messages:
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id=1),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance",id=2)
                    ]

# Now this HumanMessage will replace the HumanMessage in initial_messages due to "id"
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Model",id=2)

add_messages(initial_messages, new_message)


# Removal:
messages = [AIMessage("Hi.", name="Bot", id=1)]
messages.append(HumanMessage("Hi.", name="Lance", id=2))
messages.append(AIMessage("So you said you were researching ocean animals?", name="Bot", id=3))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id=4))

delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)

print(add_messages(messages, delete_messages))

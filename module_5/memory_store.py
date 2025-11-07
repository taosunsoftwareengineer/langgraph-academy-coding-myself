import uuid
from langgraph.store.memory import InMemoryStore

from dotenv import load_dotenv
load_dotenv("../../.env")

in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")

key = str(uuid.uuid4())

value = {"food_preference": "I like pizza"}

in_memory_store.put(namespace_for_memory, key, value)

# to retrieve:
memories = in_memory_store.search(namespace_for_memory)

# metadata
print("*****", memories[0].dict())

# key and value
print("\n*****", memories[0].key, memories[0].value)

# get the memory by namespace and key
memory = in_memory_store.get(namespace_for_memory, key)
print("\n*****", memory.dict())
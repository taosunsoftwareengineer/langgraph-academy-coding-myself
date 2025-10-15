from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
import os, getpass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.extend([HumanMessage(content=f"Yes, that's right.", name="Tao")])
messages.extend([AIMessage(content=f"Great, what would you like to learn about?", name="Model")])
messages.extend([HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Tao")])

# for m in messages:
#     m.pretty_print()
    
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        
_set_env("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke(messages)
#print(result)
print(result.response_metadata)

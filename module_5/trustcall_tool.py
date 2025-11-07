from config import llm
from trustcall import create_extractor
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [HumanMessage(content="Hi, I'm Lnace"),
                AIMessage(content="Nice to meet you, Lance."),
                HumanMessage(content="I really like biking around San Francisco.")]

class UserProfile(BaseModel):
    """User profile schema with typed fields"""
    user_name: str = Field(description="The user's preferred name")
    interests: list[str] = Field(description="A list of the user's interests")

trustcall_extractor = create_extractor(
    llm,
    tools=[UserProfile],
    tool_choice="UserProfile"
)

system_msg = "Extract the user profile from the following conversation"

result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + conversation})

for m in result["messages"]: # messages here contain tool calls
    m.pretty_print()

print("\n")
schema = result["responses"]
print(schema)

print("\n")
print(schema[0].model_dump())

print("\n")
print(result["response_metadata"])

print("\n")


# update conversation
updated_conversation = [HumanMessage(content="Hi, I'm Lnace"),
                        AIMessage(content="Nice to meet you, Lance."),
                        HumanMessage(content="I really like biking around San Francisco."),
                        AIMessage(content="San Francisco is a great city! Where do you go after biking?"),
                        HumanMessage(content="I really like to go to a bakery after biking.")]

system_msg = f"""Update the memory (JSON doc) in incorporate new information from the following conversation"""

result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + updated_conversation},
                                    {"existing": {"UserProfile": schema[0].model_dump()}})

for m in result["messages"]:
    m.pretty_print()

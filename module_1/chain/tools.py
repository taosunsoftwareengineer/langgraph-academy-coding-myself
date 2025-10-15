from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    
        Args:
            a: first int
            b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divides a by b.
    
        Args:
            a: first int
            b: second int
    """
    return a / b

def create_tool():
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools([add, multiply, divide])
    return llm_with_tools

# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multipled by 3", name="tao")])
# print(tool_call)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv("../.env")

# Single LLM instance shared across all modules
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
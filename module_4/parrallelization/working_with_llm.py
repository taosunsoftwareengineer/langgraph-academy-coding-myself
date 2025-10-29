from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import operator
from typing import Annotated

from dotenv import load_dotenv
load_dotenv("../../.env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]

def search_web(state):
    """Retrieves docs from web search"""
    tavily_search = TavilySearch(max_results=3)
    search_result = tavily_search.invoke(state["question"])
    
    # Extract the search results from the response
    search_docs = search_result.get('results', [])
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["title"]}\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]}

def search_wikipedia(state):
    """Retrieve docs from wikipedia"""

    search_docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]}

def generate_answer(state):
    """Node to answer a questios"""

    context = state["context"]
    question = state["question"]

    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, context=context)

    answer = llm.invoke([SystemMessage(content=answer_instructions)] + [HumanMessage(content=f"Answer the question.")])

    return {"answer": answer}

builder = StateGraph(State)

builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia","generate_answer")
builder.add_edge("search_web","generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

result = graph.invoke({"question": "Based on the current stock price of Intel, what is the Intel Stock price prediction in Q1 2026"})
print(result["answer"].content)
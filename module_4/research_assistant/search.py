from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from interview import InterviewState
from config import llm

tavily_search = TavilySearch(max_results=3)

search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.
                                    
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

Firs, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

def search_web(state: InterviewState):
    """Retrieves docs from web search"""

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state["messages"])
    
    # Extract the search results from the response
    search_result = tavily_search.invoke(search_query.search_query)
    search_docs = search_result.get('results', [])
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc.get("url", "")}">\n{doc.get("title", "")}\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia"""

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state["messages"])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
    
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    
    return {"context": [formatted_search_docs]}
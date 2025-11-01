"""
Goal: build a multi-agent system around LLMs that optimizes and customizes the research process, with the following steps:
1. Source Selection
2. Planning
3. LLM Utilization
4. Research Process
5. Output Format - a final report
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from analyst import GenerateAnalystsState, create_analysts, human_feedback, should_continue
from interview import InterviewState, generate_question, save_interview, route_messages
from langchain_core.messages import HumanMessage
from IPython.display import Markdown
from search import search_web, search_wikipedia
from answer import generate_answer
from write import write_section

from dotenv import load_dotenv
load_dotenv("../../.env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

analyst_builder = StateGraph(GenerateAnalystsState)
analyst_builder.add_node("create_analysts", create_analysts)
analyst_builder.add_node("human_feedback", human_feedback)

analyst_builder.add_edge(START, "create_analysts")
analyst_builder.add_edge("create_analysts", "human_feedback")
analyst_builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

memory = MemorySaver()
analyst_graph = analyst_builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

# test
max_analysts = 3
topic = """The benefits of adopting LangGraph as an agent framework"""
thread = {"configurable": {"thread_id": "1"}}

# run the graph until the first interruption
for event in analyst_graph.stream({"topic":topic, "max_analysts":max_analysts}, thread, stream_mode="values"):
    analysts = event.get("analysts", "")
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-"*50)

# Get state and look at the next node
state = analyst_graph.get_state(thread)
print(state.next) # should be at the human_feedback node

# next, test the interview
print("***"*10 + "\n")
print(f"The analyst picked for testing is: {analysts[0]}")

# Add ndoes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
thread = {"configurable": {"thread_id":  "1"}}
interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns":2}, thread)

# Display the first section
print(interview["sections"][0])

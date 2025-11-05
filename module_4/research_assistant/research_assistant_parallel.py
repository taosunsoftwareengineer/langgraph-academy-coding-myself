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
from analyst import create_analysts, human_feedback
from interview import InterviewState, generate_question, save_interview, route_messages
from search import search_web, search_wikipedia
from answer import generate_answer
from write import write_section
from research import ResearchGraphState, write_report, write_introduction, write_conclusion, finalize_report, initiate_all_interviews
from IPython.display import Markdown

from dotenv import load_dotenv
load_dotenv("../../.env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=MemorySaver())

# test
max_analysts = 3
topic = """The benefits of adopting LangGraph as an agent framework"""
thread = {"configurable": {"thread_id": "1"}}

# run the graph until the first interruption
for event in graph.stream({"topic":topic, "max_analysts":max_analysts}, thread, stream_mode="values"):
    analysts = event.get("analysts", "")
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-"*50)

graph.update_state(thread, {"human_analyst_feedback": "Add in the CEO of gen ai native startup"}, as_node="human_feedback")

for event in graph.stream(None, thread, stream_mode="values"):
    analysts = event.get("analysts", "")
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-"*50)

graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")

for event in graph.stream(None, thread, stream_mode="updates"):
    print("--None--")
    node_name = next(iter(event.keys()))
    print(node_name)
    
final_state = graph.get_state(thread)
report = final_state.values.get("final_report")
print(report)

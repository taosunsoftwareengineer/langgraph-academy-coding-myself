"""
Goal: build a multi-agent system around LLMs that optimizes and customizes the research process, with the following steps:
1. Source Selection
2. Planning
3. LLM Utilization
4. Research Process
5. Output Format - a final report
"""

from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.types import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from analyst import GenerateAnalystsState, Perspectives, analyst_instructions

from dotenv import load_dotenv
load_dotenv("../../.env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # enforce structured feedback
    structured_llm = llm.with_structured_output(Perspectives)

    system_message = analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)
    
    # generate question
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate the set of analysts.")])

    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analysts"

    return END

builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

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

# Get state and look at the next node
state = graph.get_state(thread)
print(state.next) # should be at the human_feedback node



import operator
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from analyst import Analyst
from config import llm
from langchain_core.messages import get_buffer_string

class InterviewState(MessagesState):
    max_num_turns: int # number turns of conversation
    context: Annotated[list, operator.add]
    analyst: Analyst # analyst asking questions
    interview: str # interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

3. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is the topic of focus and set of goals: {goals}

Begin my introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask question to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    analyst = state["analyst"]
    messages = state["messages"]

    # generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {"messages": [question]}

def save_interview(state: InterviewState):
    """ Save interviews """

    messages = state['messages']

    # convert interview to a string
    interview = get_buffer_string(messages)
    
    # save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """ Route between question and answer """

    messages = state['messages']
    max_num_turns = state.get("max_num_turns", 2)
    
    # check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    
    # end if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"
    
    # this router is run after each question - answer pair
    # get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"

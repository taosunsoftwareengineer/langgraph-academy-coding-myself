import operator
from typing import Annotated
from langgraph.graph import MessagesState
from analyst import Analyst
from pydantic import BaseModel, Field

class InterviewState(MessagesState):
    max_num_turns: int # number turns of conversation
    context: Annotated[list, operator.add]
    analyst: Analyst # analyst asking questions
    interview: str # interview transcript
    sections: list # Final key we duplicate in outer state for Send() API
    
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

3. Specific: Insights that avoid generalities and inclue specific examples from the expert.

Here is the topic of focus and set of goals: {goals}

Begin my introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask question to drill down and refine your understanding of the topic.

When ou are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    analyst = state["analyst"]
    messages = state["messages"]

    # generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = 
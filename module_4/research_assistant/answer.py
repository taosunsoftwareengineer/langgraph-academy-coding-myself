from interview import InterviewState
from langchain_core.messages import SystemMessage
from config import llm

answer_instructions = """ You are en expert being interviewed by an analyst.

Here is analyst area of focus: {goals}.

Your goal is to answer a question posed by the interviewer.

To answer question, use the context:

{context}

when answering questions, follow these guidelines:

1. Use only the information provided in the context.

2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 user [1].

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

6. If the source is <Document source="assistant/docs/llama3_1.pdf" page="7"/> then just list:
[1] assistant/docs/llama3_1.pdf, page 7

And skip the addition of the brackets as well as the document source preamble in your citation."""

def generate_answer(state: InterviewState):
    """ Node to answer a question """

    analyst = state['analyst']
    messages = state['messages']
    context = state['context']

    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    # name the message as coming from the expert
    answer.name = "expert"
    
    return {"messages": [answer]}


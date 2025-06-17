from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from utils import Utils
import warnings
warnings.filterwarnings("ignore")
u = Utils()
llm = u.initialize_llm4()


class Generic_State(TypedDict):
    question: str
    generic_response: str

class Generic_Agent:    
    def generic_bot(self, question):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are a friendly and engaging intelligent assistant.  
                If the user greets you with a simple greeting like "hi", "hello", or similar, respond with a warm and cheerful message.
                Otherwise, respond to their question in a helpful, clear, and conversational tone.

                User: {question}  
                Answer:
            """
            )
        llm_chain = LLMChain(llm = llm, prompt = prompt)
        response = llm_chain.invoke({"question": question})
        print(response)
        return response["text"]
    
def generic_agent(state: Generic_State) -> Command[Literal["__end__"]]:


    agent = Generic_Agent()  
    generic_response = agent.generic_bot(state["question"])

    return Command(
        update = {
            "generic_response": generic_response},
        goto = END
    )
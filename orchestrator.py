import json
from typing import Literal, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from utils import Utils
from general_research_agent import *
from academic_research_agent import *
from product_research_agent import *
from generic_bot import *

u = Utils()
llm = u.initialize_llm()
g1 = General()
g2 = Academic()
g3 = Product()

class ResearchState(TypedDict):
    question: str
    category: str
    subgraph1_response: dict
    subgraph2_response: dict
    subgraph3_response: dict
    generic_response : str


class LangGraphRouter:
    def __init__(self):
        self.graph1 = g1.build_graph()
        self.graph2 = g2.build_graph()
        self.graph3 = g3.build_graph()
        self.graph4 = generic_agent
        self.parent_graph = self.build_parent_graph()

    # 🧠 Orchestrator node with LLM-based router
    def router_node(self, state: ResearchState) -> Command[Literal['general_research', 'academic_research', 'product_research', 'generic_bot']]:
        prompt = """
        You are a query router. Categorize the user query into one of the following categories:

        - general_research: For open-ended, non-academic research like "climate change impact on agriculture"
        - academic_research: For scholarly or scientific topics like "GPT fine-tuning techniques"
        - product_research: For consumer-oriented searches like "best headphones under 2000 INR"
        - generic: For greetings, small talk, or anything irrelevant

        Your final response should be dictionary : {{"next_agent" : "<One of the provided category>"}}

        User query: {question}
        """
        llm_chain_prompt = PromptTemplate(input_variables=['question'], template=prompt)
        llm_chain = LLMChain(llm=llm, prompt=llm_chain_prompt)
        llm_chain_resp = llm_chain.invoke({'question': state['question']})
        print(llm_chain_resp)
        response = u.correct_json(llm_chain_resp["text"])
        response = json.loads(response)
        
        if response["next_agent"] == 'general_research':
            goto = 'general_research_agent'
        elif response["next_agent"] == 'academic_research':
            goto = 'academic_research_agent'
        elif response["next_agent"] == 'product_research':
            goto = 'product_research_agent'
        else:
            goto = 'generic_bot'
        
        return Command(
            goto=goto,
            update={"category": response["next_agent"]}
        )
        
    def build_parent_graph(self):
        parent_builder=StateGraph(ResearchState)
        parent_builder.add_node("orchestrator_node",self.router_node)
        parent_builder.add_node("general_research_agent",self.graph1)
        parent_builder.add_node("academic_research_agent",self.graph2)
        parent_builder.add_node("product_research_agent",self.graph3)
        parent_builder.add_node("generic_bot",self.graph4)
        parent_builder.add_edge(START,"orchestrator_node")
        parent_graph = parent_builder.compile()
        return parent_graph
    
    def run_graph(self,question):
        results = []

        graph=self.parent_graph
        for s in graph.stream({'question': question}, debug= True, stream_mode='values'):
            print(s)
            results.append(s)
        final_state = results[-1]
        print(final_state, type(final_state))
        return final_state
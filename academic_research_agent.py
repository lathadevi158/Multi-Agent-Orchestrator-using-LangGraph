import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from serpapi import GoogleSearch


from utils import Utils
u = Utils()

# Initialize LLM globally
llm = u.initialize_llm()

# 🧠 Custom State for the Academic Research Subgraph
class AcademicResearchState(TypedDict):
    question: str
    academic_results: str
    subgraph2_response: dict


# 📄 Prompt for Supervisor Logic
academic_supervisor_prompt = """
You are a Supervisor Agent for an academic research assistant.

Your job is to route a user's  query to:
- 'academic_search_agent': to simulate gathering results from academic sources like journals, papers, and conference proceedings.
- 'research_summary_agent': to summarize and synthesize those academic results.

Process:
    - First call the 'academic_search_agent' to fetch academic data.
    - After getting the response from 'academic_search_agent' as "SUCCESS", then call the 'research_summary_agent' to get the final response.
    - Once both responses are collected, return 'FINISH'.
"""

# 🛣️ Routing output
class AcademicResearchAgents(TypedDict):
    next: Literal["academic_search_agent", "research_summary_agent", "FINISH"]


# 👨‍🏫 Supervisor Node
def academic_supervisor_node(state: AcademicResearchState) -> Command[Literal["academic_search_agent", "research_summary_agent", "__end__"]]:
    question = state["question"]
    academic_results = state.get("academic_results", [])
    subgraph2_response = state.get("subgraph2_response", {})
    
    messages = f"{academic_supervisor_prompt}\nquestion: {question}\n academic_results: {academic_results}\n subgraph2_response: {subgraph2_response}"
    
    response = llm.with_structured_output(AcademicResearchAgents).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


# 🔍 Academic Search Agent Node
def academic_search_agent_node(state: AcademicResearchState) -> Command:
    question = state["question"]
    results = get_academic_results(question)
    print(f"[Academic Search Agent] Results for '{question}':\n{results}")
    return Command(
        update={"academic_results": results}, goto="academic_supervisor_node")


# 📝 Academic Summary Agent Node
def academic_summary_agent_node(state: AcademicResearchState) -> Command:
    search_text = state.get("academic_results", "")
    
    summary_prompt = PromptTemplate(
        input_variables=["academic_results"],
        template="""
        You are an expert academic assistant.

        Your task is to read the following scholarly findings and research insights, and create a comprehensive, clear, and succinct summary suitable for a researcher or decision-maker.

        Focus on key findings, methods, implications, and insights.

        Academic Results:
        -----------------
        {academic_results}

        Summary:
        """
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = chain.invoke({"academic_results": search_text})
    summary = response["text"].strip()
    
    print(f"[Academic Summary Agent] Summary:\n{summary}")
    
    return Command(
        update={"subgraph2_response": summary}, goto="academic_supervisor_node")

class Academic:
    def __init__(self):
        self.graph = self.build_graph()
        
    # 🔧 LangGraph Builder Function
    def build_graph():
        builder = StateGraph(AcademicResearchState)
        #adding nodes
        builder.add_node("academic_supervisor_node", academic_supervisor_node)
        builder.add_node("academic_search_agent", academic_search_agent_node)
        builder.add_node("research_summary_agent", academic_summary_agent_node)

        builder.add_edge(START, "academic_supervisor_node")

        return builder.compile()
    
#helper function
def get_academic_results(query: str, num_results: int = 5) -> str:
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERP_API_KEY"),
        "num": num_results,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    articles = results.get("organic_results", [])
    if not articles:
        return "No academic results found."

    formatted = []
    for i, article in enumerate(articles[:num_results], 1):
        title = article.get("title", "No title")
        snippet = article.get("snippet", "No description")
        link = article.get("link", "")
        formatted.append(f"{i}. **{title}**\n{snippet}\n🔗 {link}\n")

    return "\n\n".join(formatted)
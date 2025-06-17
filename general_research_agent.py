from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils import Utils
import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables from .env
load_dotenv()
SERPAPI_KEY = os.getenv("SERP_API_KEY")

u = Utils()
# Initialize LLM globally
llm = u.initialize_llm()


# 🧠 Custom State for the General Research Subgraph
class GeneralResearchState(TypedDict):
    question: str
    search_results: str
    subgraph1_response: dict


# 📄 Prompt for Supervisor Logic
supervisor_prompt = """
You are a Supervisor Agent for a general research assistant.

Your job is to route a user's question to:
- 'web_search_agent': to simulate gathering search results.
- 'summary_agent': to summarize gathered results.

Process:
    - First call the 'web_search_agent' to get search results.
    - After getting the response from 'web_search_agent' as "SUCCESS", then call the 'summary_agent' to get final response.
    - Finally, If you got both the 'web_search_agent' response and final response, End the process and return as 'FINISH'.

"""

# 🛣️ Routing output
class GeneralResearchAgents(TypedDict):
    next: Literal["web_search_agent", "summary_agent", "FINISH"]


# 👨‍💼 Supervisor Node
def supervisor_node(state: GeneralResearchState) -> Command[Literal["web_search_agent", "summary_agent", "__end__"]]:
    question = state["question"]
    web_search_response = state.get("search_results",[])
    subgraph1_response = state.get("subgraph1_response",{})
    messages =f"{supervisor_prompt}\nquestion: {question}\n search_results: {web_search_response}\n subgraph1_response: {subgraph1_response}"
    response = llm.with_structured_output(GeneralResearchAgents).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


# 🌐 Web Search Agent Node (Mock Search Results)
def web_search_agent_node(state: GeneralResearchState) -> Command:
    question = state["question"]
    results = get_search_results(question)
    print(f"[Web Search Agent] Real Results for '{question}':\n{results}")
    return Command(
        update={"search_results": results}, goto="supervisor_node")



# 📝 Summary Agent Node
def summary_agent_node(state: GeneralResearchState) -> Command:
    search_text = state.get("search_results", "")
    
    summary_prompt = PromptTemplate(
    input_variables=["search_results"],
    template="""
        You are an expert research assistant.

        Your task is to read the following information gathered from various web search results, and generate a concise, informative, and easy-to-understand summary. 
        Focus on extracting key facts, themes, and insights.

        Web Search Results:
        --------------------
        {search_results}

        Summary:
        """
        )

    chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = chain.invoke({"search_results": search_text})
    summary = response["text"].strip()
    
    print(f"[Summary Agent] Summary:\n{summary}")
    
    return Command(
        update={"subgraph1_response": summary}, goto="supervisor_node")


class General:
    def __init__(self):
        self.graph = self.build_graph()
        
    # 🔧 LangGraph Builder Function
    def build_graph():
        builder = StateGraph(GeneralResearchState)
        #adding nodes to the graph
        builder.add_node("supervisor_node", supervisor_node)
        builder.add_node("web_search_agent", web_search_agent_node)
        builder.add_node("summary_agent", summary_agent_node)
        builder.add_edge(START,"supervisor_node")
        return builder.compile()


#helper functions
def get_search_results(query: str) -> str:
    if not SERPAPI_KEY:
        return "Error: SERPAPI_KEY not found. Please set it in your .env file."

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = []
    for result in results.get("organic_results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        if snippet:
            snippets.append(f"🔹 **{title}**\n{snippet}\n🔗 {link}\n")

    return "\n".join(snippets) if snippets else "No relevant results found."

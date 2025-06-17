import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from serpapi import GoogleSearch
from utils import Utils
u = Utils()

# Initialize LLM globally
llm = u.initialize_llm()

# 🧠 Custom State for the Product Research Subgraph
class ProductResearchState(TypedDict):
    question: str
    trend_data: str
    subgraph3_response: dict


# 📄 Prompt for Supervisor Logic
product_supervisor_prompt = """
You are a Supervisor Agent for a product/market research assistant.

Your job is to route a user's query to:
- 'market_trend_agent': to simulate gathering market and product trend insights.
- 'product_summary_agent': to summarize and interpret these insights.

Process:
    - First call the 'market_trend_agent' to get product or market data.
    - After getting the response from 'market_trend_agent' as "SUCCESS", then call the 'product_summary_agent' to get the final summary.
    - Once both responses are collected, return 'FINISH'.
"""

# 🛣️ Routing output
class ProductResearchAgents(TypedDict):
    next: Literal["market_trend_agent", "product_summary_agent", "FINISH"]


# 👨‍💼 Supervisor Node
def product_supervisor_node(state: ProductResearchState) -> Command[Literal["market_trend_agent", "product_summary_agent", "__end__"]]:
    question = state["question"]
    trend_data = state.get("trend_data", [])
    subgraph3_response = state.get("subgraph3_response", {})

    messages = f"{product_supervisor_prompt}\nquestion: {question}\n trend_data: {trend_data}\n subgraph3_response: {subgraph3_response}"
    
    response = llm.with_structured_output(ProductResearchAgents).invoke(messages)
    goto = response["next"]
    print(f"Next Worker: {goto}")
    
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


def market_trend_agent_node(state: ProductResearchState) -> Command:
    question = state["question"]
    results = get_product_trends(question)
    print(f"[Market Trend Agent] Real trend data for '{question}':\n{results}")
    return Command(update={"trend_data": results}, goto="product_supervisor_node")

# 🧾 Product Summary Agent Node
def product_summary_agent_node(state: ProductResearchState) -> Command:
    trend_text = state.get("trend_data", "")

    summary_prompt = PromptTemplate(
        input_variables=["trend_data"],
        template="""
        You are an expert product research analyst.

        Your task is to analyze the following trend data gathered from various sources online and generate a clear, actionable summary of the product trends.

        Focus on emerging patterns, popular products, customer interests, and potential opportunities.

        Product Trend Data:
        -------------------
        {trend_data}

        Summary:
        """
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt)
    response = chain.invoke({"trend_data": trend_text})
    summary = response["text"].strip()

    print(f"[Product Summary Agent] Summary:\n{summary}")

    return Command(update={"subgraph3_response": summary}, goto="product_supervisor_node")


class Product:
    def __init__(self):
        self.graph = self.build_graph()
        
    # 🔧 LangGraph Builder Function
    def build_graph():
        builder = StateGraph(ProductResearchState)
        #adding nodes
        builder.add_node("product_supervisor_node", product_supervisor_node)
        builder.add_node("market_trend_agent", market_trend_agent_node)
        builder.add_node("product_summary_agent", product_summary_agent_node)

        builder.add_edge(START, "product_supervisor_node")

        return builder.compile()

#helper function
def get_product_trends(query: str, num_results: int = 5) -> str:
    params = {
        "engine": "google",
        "q": query + " product trends",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic = results.get("organic_results", [])

    if not organic:
        return "No product trend data found."

    formatted = []
    for i, item in enumerate(organic[:num_results], 1):
        title = item.get("title", "No title")
        snippet = item.get("snippet", "No description")
        link = item.get("link", "")
        formatted.append(f"{i}. **{title}**\n{snippet}\n🔗 {link}\n")

    return "\n\n".join(formatted)
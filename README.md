# 🧠 Multi-Agent Research Assistant with LangGraph

An intelligent, agentic workflow built using **LangGraph**, featuring dynamic query routing and specialized subgraphs for handling general, academic, and product research. This system demonstrates how LLMs and modular agents can collaborate to deliver smart, routed, and insightful responses.

---

## 🚀 Features

- 🧭 **LLM-based Query Routing**: Automatically categorizes user queries into general, academic, product, or generic intent.  
- 🧪 **Subgraph Architecture**: Each category has its own LangGraph subgraph pipeline for task-specific reasoning.  
- 🔍 **Web Trend Simulation**: Uses SerpAPI to simulate product trend searches for market research.  
- 🧠 **LLM Summarization**: Final insights are crafted using large language models with context.  
- 💬 **Friendly Generic Bot**: Engages users with casual greetings or fallback responses.

---

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — Agentic workflows with graph-based state transitions  
- [LangChain](https://github.com/langchain-ai/langchain) — Framework for LLM-powered applications  
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview) — Model for generation and summarization  
- [SerpAPI](https://serpapi.com) — For product trend data from search engine results  
- [dotenv](https://pypi.org/project/python-dotenv/) — Environment variable management

---

## 🏁 Getting Started

### 🔧 Prerequisites

- Python 3.8+  
- SerpAPI key  
- Create a `.env` file 

---

📘 **Note**  

This project demonstrates a basic agentic workflow using LangGraph and is intended primarily for learning and showcasing how query routing and modular agent design works in a LangChain ecosystem. It serves as a foundational example that can be easily extended with advanced tools like memory, vector databases (e.g., FAISS, Chroma, Azure Cosmos DB), external APIs, or dynamic tool-based agents. While it is minimal by design, the structure is scalable and serves as a great starting point for building more sophisticated, production-ready AI assistants.

---

📄 **License**  
This project is licensed under the MIT License.

---

🙋‍♀️ **Author**  
**Marpally Latha Devi**  
Prompt Engineer | Generative AI Developer  
GitHub: [lathadevi158](https://github.com/lathadevi158)


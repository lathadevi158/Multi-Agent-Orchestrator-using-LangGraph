from orchestrator import *
l = LangGraphRouter()

question= "What are the current trends in sustainable packaging?"
#running the graph
result = l.run_graph(question)
print(
    result.get("subgraph1_response")
    or result.get("subgraph2_response")
    or result.get("subgraph3_response")
    or result.get("generic_response")
    or "No response found."
)

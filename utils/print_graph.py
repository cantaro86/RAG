"""
From inside utils, run this script with:
>> PYTHONPATH=../src python print_graph.py

Run it inside a GPU node.
"""

from langchain_core.runnables.graph import MermaidDrawMethod

from agentic_rag._load_env import cfg, console
from agentic_rag.agent_factory import build_rag_agent

GRAPH_IMAGE_PATH = "../graph.png"

############################### EDIT PATHS
cfg.index_dir = "." + cfg.index_dir
cfg.dizionario_path = "." + cfg.dizionario_path
###############################

agent = build_rag_agent(cfg)

img = agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
with open(GRAPH_IMAGE_PATH, "wb") as f:
    f.write(img)

console.print(f"Graph image saved to [bold green]{GRAPH_IMAGE_PATH}[/bold green]")
print("")
console.print(f"Mermaid graph:\n{agent.get_graph().draw_mermaid()}")

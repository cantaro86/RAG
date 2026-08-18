"""Render the agent graph without loading runtime models or data stores."""

import os
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_IMAGE_PATH = REPO_ROOT / "graph.png"


def _build_graph_for_rendering(cfg):
    from agentic_rag.graph import RAGContext, build_agent_graph

    dependency = MagicMock(name="unloaded_graph_dependency")
    context = RAGContext(
        topic_continuity_classifier=dependency,
        retriever=dependency,
        rag_chain=dependency,
        sanitizer_chain=dependency,
        cleaner_chain=dependency,
        pre_retrieval_question_rewriter=dependency,
        question_transformer=dependency,
        guardrail_chain=dependency,
        synonyms=dependency,
    )
    return build_agent_graph(context, cfg).get_graph()


def main() -> None:
    os.chdir(REPO_ROOT)

    from langchain_core.runnables.graph import MermaidDrawMethod

    from agentic_rag._load_env import cfg, console

    graph = _build_graph_for_rendering(cfg)
    image = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    GRAPH_IMAGE_PATH.write_bytes(image)

    console.print(f"Graph image saved to [bold green]{GRAPH_IMAGE_PATH}[/bold green]")
    console.print(f"\nMermaid graph:\n{graph.draw_mermaid()}")


if __name__ == "__main__":
    main()

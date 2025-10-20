import os

from langchain_core.output_parsers import StrOutputParser
from rich.pretty import Pretty

import src._load_env as _  # noqa: F401
from src._load_env import Config, cfg, console
from src.agent_prompts import prompt_hallucination, prompt_rag, prompt_rewrite_medical, prompt_usefulness
from src.build_faiss import build_faiss_index, load_vectorstore
from src.graph import RAGContext, build_agent_graph
from src.llm_build import build_llm_pipe
from src.retriever import build_retriever
from src.utils import clean_answer, parse_json


# ------------------------
# Interactive loop
# ------------------------
def interactive_loop(cfg: Config):
    """Interactive loop with the agent"""
    vs = load_vectorstore(cfg.index_dir, cfg.embed_model)
    retriever = build_retriever(
        vs,
        cfg.k,
        cfg.rerank_model if cfg.rerank else None,
        cfg.k_reranked,
    )
    llm = build_llm_pipe(cfg.llm_model, cfg.max_new_tokens, cfg.temperature)

    rag_chain = prompt_rag | llm | clean_answer | StrOutputParser()
    hallucination_grader = prompt_hallucination | llm | clean_answer | parse_json
    answer_grader = prompt_usefulness | llm | clean_answer | parse_json
    question_rewriter = prompt_rewrite_medical | llm | clean_answer | StrOutputParser()

    ctx = RAGContext(
        retriever=retriever,
        rag_chain=rag_chain,
        hallucination_grader=hallucination_grader,
        answer_grader=answer_grader,
        question_rewriter=question_rewriter,
    )

    # Build agent graph
    agent = build_agent_graph(ctx)

    console.print("[bold green]RAG Agent. Type 'exit', 'quit' or 'q' to quit.[/bold green]")
    console.print("[yellow]The agent will decide when to search documents and when to respond directly.[/yellow]")

    thread_id = "default"

    while True:
        try:
            question = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        # Invoke agent
        config = {"configurable": {"thread_id": thread_id}}
        agent_input = {"question": question, "config": config}

        last_output = None
        try:
            # Stream the agent's execution
            for output in agent.stream(agent_input):
                for _key, value in output.items():
                    # Node
                    # pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    # console.print(value, indent=2, width=80, depth=None)
                    console.print(Pretty(value, max_depth=4))
                    last_output = value
                console.print("\n---\n")

            # Final generation
            if last_output and isinstance(last_output, dict) and "generation" in last_output:
                console.print(last_output["generation"])
            else:
                console.print("[yellow]No generation returned from agent.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ------------------------


def main():
    console.print(f"Using HF cache dir: [bold]{cfg.hf_home}[/bold]")

    # Rebuild FAISS index if requested
    if getattr(cfg, "reindex", False):
        build_faiss_index(cfg)

    # Ensure FAISS index exists
    if not os.path.isdir(cfg.index_dir) or not os.listdir(cfg.index_dir):
        console.print(
            f"[red]FAISS index not found or empty at {cfg.index_dir}. "
            "Set 'reindex: true' in config.yaml to build it.[/red]"
        )
        return

    # Run interactive chat
    if getattr(cfg, "chat", False):
        interactive_loop(cfg)
    else:
        console.print("[yellow]Set 'chat: true' in config.yaml to start the agent.[/yellow]")


if __name__ == "__main__":
    main()

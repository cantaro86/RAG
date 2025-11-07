import os
import sys

import src._load_env as _  # noqa: F401  # isort: skip
from src._load_env import Config, cfg, console, ONLINE  # noqa: F401  # isort: skip

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.agent_prompts import prompt_general, prompt_rag, prompt_rewrite_medical, prompt_validate_medical
from src.bilingual_question import BilingualQuestion, init_translators
from src.build_faiss import build_faiss_index, load_vectorstore
from src.graph import RAGContext, build_agent_graph
from src.llm_build import build_llm_pipe
from src.loggers import Logger
from src.retriever import build_retriever
from src.utils import extract_answer_text, extract_last_json

logger = Logger.get_logger(__name__)


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

    llm_runnable = RunnableLambda(lambda text: llm.invoke([{"role": "user", "content": str(text)}])["content"])
    clean_answer = RunnableLambda(extract_answer_text)
    parse_json = RunnableLambda(extract_last_json)

    answer_validation = prompt_validate_medical | llm_runnable | clean_answer | parse_json
    chain_general = prompt_general | llm_runnable | clean_answer | StrOutputParser()
    rag_chain = prompt_rag | llm_runnable | clean_answer | StrOutputParser()
    question_rewriter = prompt_rewrite_medical | llm_runnable | clean_answer | StrOutputParser()

    ctx = RAGContext(
        answer_validation=answer_validation,
        chain_general=chain_general,
        retriever=retriever,
        rag_chain=rag_chain,
        question_rewriter=question_rewriter,
    )

    # Build agent graph
    agent = build_agent_graph(ctx)

    console.print("[bold green]RAG Agent. Type 'esci', 'exit', 'quit' or 'q' to quit.[/bold green]")
    console.print("[yellow]The agent will decide when to search documents and when to respond directly.[/yellow]")

    thread_id = "default"

    while True:
        try:
            print("\nYou: ", end="", flush=True)
            question = sys.stdin.buffer.readline().decode("utf-8", errors="replace").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"exit", "quit", "q", "esci"}:
            logger.info("User exited the program session id=%s", thread_id)
            break

        try:
            quest = BilingualQuestion(question)
            logger.debug(f"Language = {quest.lang}, class = {quest}")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        # Invoke agent
        config = {"configurable": {"thread_id": thread_id}}
        agent_input = {"question": quest.en}

        last_output = None
        try:
            # Stream the agent's execution
            for output in agent.stream(agent_input, config=config):
                for _key, value in output.items():
                    logger.debug(f"Key: {_key}")
                    last_output = value

            # Final generation
            if last_output and isinstance(last_output, dict) and "generation" in last_output:
                console.print(last_output["generation"])
            else:
                console.print("[yellow]No generation returned from agent.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ------------------------


def main():
    logger.info(f"Using HF cache dir: {cfg.hf_home}")
    logger.info(f"The network connectivity is: {'online' if ONLINE else 'offline'}")
    logger.info(f"Online flag is set to: {getattr(cfg, 'online', True)}")

    try:
        logger.info("Warming up translators...")
        init_translators()
        logger.info("Translators ready.")
    except Exception as e:
        logger.error(f"Failed to initialize translators: {e}")
        console.print("[red]Failed to initialize translators. Check configuration or models.[/red]")
        return

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

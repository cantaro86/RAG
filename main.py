import os
import sys
import traceback

import src._load_env as _  # noqa: F401  # isort: skip
from src._load_env import Config, cfg, console, ONLINE  # noqa: F401  # isort: skip


from src.agent_factory import build_rag_agent
from src.bilingual_question import BilingualQuestion, init_translators
from src.build_faiss import build_faiss_index
from src.loggers import Logger
from src.ui_gradio import launch_gradio

logger = Logger.get_logger(__name__)


# ------------------------
# Interactive loop
# ------------------------
def interactive_loop(cfg: Config):
    """Interactive loop with the agent"""
    agent = build_rag_agent(cfg)

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
            logger.error(f"Error processing question: {e}")
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
                if quest.lang == "it":
                    # Translate answer back to Italian
                    answer_it = quest.translate_to_italian(last_output["generation"])
                    console.print(answer_it)
                else:
                    console.print(last_output["generation"])
            else:
                console.print("[yellow]No generation returned from agent.[/yellow]")

        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            console.print(f"[red]Error: {e}[/red]")
            raise e


# ------------------------


def main():
    logger.info("Starting RAG Agent")
    logger.info(f"Python interpreter: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Using HF cache dir: {cfg.hf_home}")
    logger.info(f"The network connectivity is: {'online' if ONLINE else 'offline'}")
    logger.info(f"Online flag is set to: {getattr(cfg, 'online', True)}")

    logger.debug(f"Configuration: {cfg.__dict__}")

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

    # Run interactive chat or Gradio UI
    if getattr(cfg, "chat", False):
        interactive_loop(cfg)
    elif getattr(cfg, "gradio", False):
        launch_gradio(cfg)
    else:
        console.print("[yellow]Set 'chat: true' or 'gradio: true' in config.yaml to start the agent.[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

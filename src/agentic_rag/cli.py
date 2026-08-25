import os
import sys
import traceback
from pathlib import Path

import agentic_rag._load_env as _  # noqa: F401  # isort: skip
from agentic_rag.config_schema import Config
from agentic_rag._load_env import (  # noqa: F401  # isort: skip
    EFFECTIVE_HF_HOME,
    HF_ONLINE,
    NETWORK_AVAILABLE,
    NETWORK_PROBE_SKIPPED,
    cfg,
    console,
)
from rich.markup import escape

from agentic_rag.agent_factory import build_rag_agent
from agentic_rag.build_faiss import build_faiss_index
from agentic_rag.detect_language import DetectLanguage
from agentic_rag.loggers import Logger
from agentic_rag.ui_gradio import launch_gradio

logger = Logger.get_logger(__name__)
_INDEX_FILENAMES = ("index.faiss", "index.pkl")


# ------------------------
# Interactive loop
# ------------------------
def interactive_loop(cfg: Config):
    """Interactive loop with the agent"""
    agent = build_rag_agent(cfg)

    console.print("[bold green]RAG Agent. Type 'esci', 'exit', 'quit' or 'q' to quit.[/bold green]")
    console.print("[yellow]On-topic answers are grounded in the indexed documents.[/yellow]")

    thread_id = "default"

    while True:
        try:
            print("\nYou: ", end="", flush=True)
            raw = sys.stdin.buffer.readline()
            if not raw:
                logger.info("EOF reached, exiting interactive loop")
                print()
                break
            question = raw.decode("utf-8", errors="replace").strip()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, exiting interactive loop")
            print()
            break

        if question.strip().lower() in {"exit", "quit", "q", "esci"}:
            logger.info("User exited the program session id=%s", thread_id)
            break

        try:
            quest = DetectLanguage(question, online=cfg.online)
            logger.info(f"Language = {quest.lang}, class = {quest}")
        except ValueError as e:
            logger.error(f"Error processing question: {e}")
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            continue

        # Invoke agent
        config = {"configurable": {"thread_id": thread_id}}
        agent_input = {"question": quest.text}

        last_output = None
        try:
            # Stream the agent's execution
            for output in agent.stream(agent_input, config=config):
                for _key, value in output.items():
                    logger.debug(f"Key: {_key}")
                    last_output = value

            # Final generation
            if last_output and isinstance(last_output, dict) and "generation" in last_output:
                console.print(str(last_output["generation"]), markup=False)
            else:
                console.print("[yellow]No generation returned from agent.[/yellow]")

        except Exception as e:
            logger.exception("Error during agent execution")
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            continue


def _runtime_mode(config: Config) -> str | None:
    override = os.environ.get("AGENTIC_RAG_MODE")
    if override is not None:
        mode = override.strip().lower()
        if mode not in {"chat", "gradio"}:
            raise ValueError("AGENTIC_RAG_MODE must be 'chat' or 'gradio'")
        return mode

    if config.chat:
        return "chat"
    if config.gradio:
        return "gradio"
    return None


def _validate_source_resources(config: Config) -> None:
    markdown_dir = Path(config.md_dir)
    markdown_files = [path for path in markdown_dir.glob("*.md") if path.is_file()]
    if not markdown_dir.is_dir() or not markdown_files:
        message = f"No top-level Markdown files found in {markdown_dir}."
        console.print(f"[red]{message}[/red]")
        logger.error(message)
        raise FileNotFoundError(message)

    dictionary_path = Path(config.dizionario_path)
    if not dictionary_path.is_file():
        message = f"Dictionary file not found: {dictionary_path}."
        console.print(f"[red]{message}[/red]")
        logger.error(message)
        raise FileNotFoundError(message)


def _validate_index_resources(config: Config) -> None:
    index_dir = Path(config.index_dir)
    missing = [name for name in _INDEX_FILENAMES if not (index_dir / name).is_file()]
    if not index_dir.is_dir() or missing:
        missing_names = ", ".join(missing or _INDEX_FILENAMES)
        message = f"FAISS index is incomplete at {index_dir}; missing: {missing_names}."
        console.print(f"[red]{message}[/red]")
        logger.error(message)
        raise FileNotFoundError(message)


# ------------------------


def main():
    logger.info("Starting RAG Agent")
    logger.info(f"Python interpreter: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Using HF cache dir: {EFFECTIVE_HF_HOME}")
    if NETWORK_PROBE_SKIPPED:
        logger.info("Network connectivity probe skipped because offline mode was requested")
    else:
        logger.info(f"The network connectivity is: {'online' if NETWORK_AVAILABLE else 'offline'}")
    logger.info(f"Effective Hugging Face mode is: {'online' if HF_ONLINE else 'offline'}")
    logger.info(f"Configured online flag is set to: {cfg.online}")

    logger.debug(f"Configuration: {cfg.__dict__}")

    mode = _runtime_mode(cfg)
    if mode is None:
        console.print("[yellow]Set 'chat: true' or 'gradio: true' in config.yaml to start the agent.[/yellow]")
        return

    _validate_source_resources(cfg)

    # Rebuild FAISS index if requested
    if cfg.reindex:
        build_faiss_index(cfg)

    _validate_index_resources(cfg)

    # Run interactive chat or Gradio UI
    if mode == "chat":
        interactive_loop(cfg)
    elif mode == "gradio":
        launch_gradio(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

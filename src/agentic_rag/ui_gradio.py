# ruff: noqa: I001
import os
import uuid

from agentic_rag._load_env import attach_debugger_if_requested

import gradio as gr

from agentic_rag.agent_factory import build_rag_agent
from agentic_rag.config_schema import Config
from agentic_rag.detect_language import DetectLanguage
from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)
STREAM_FAILURE_MESSAGE = "The request could not be completed. Please try again."


def _server_settings(cfg: Config) -> tuple[str, int]:
    host = os.environ.get("GRADIO_SERVER_NAME", os.environ.get("GRADIO_HOST", cfg.gradio_host)).strip()
    if (
        not host
        or len(host) > 253
        or any(character.isspace() for character in host)
        or "://" in host
        or "/" in host
        or "\\" in host
    ):
        raise ValueError(f"Invalid Gradio server host: {host!r}")

    raw_port = os.environ.get("GRADIO_SERVER_PORT", os.environ.get("GRADIO_PORT", str(cfg.gradio_port))).strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"Invalid Gradio server port: {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Gradio server port must be between 1 and 65535, got {port}")
    return host, port


def _normalize_history(history) -> list[dict[str, str]]:
    messages = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content", ""))})
        elif isinstance(item, list | tuple) and len(item) == 2:
            user_message, assistant_message = item
            if user_message is not None:
                messages.append({"role": "user", "content": str(user_message)})
            if assistant_message is not None:
                messages.append({"role": "assistant", "content": str(assistant_message)})
    return messages


def _delete_checkpoint(agent, thread_id: str | None) -> None:
    checkpointer = getattr(agent, "checkpointer", None)
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if thread_id and callable(delete_thread):
        delete_thread(thread_id)


def _response_callback(agent, *, online: bool | None = None):
    def respond(message: str, history, session_state: dict | None):
        visible_history = _normalize_history(history)
        previous_thread_id = session_state.get("thread_id") if isinstance(session_state, dict) else None
        expected_history = session_state.get("history") if isinstance(session_state, dict) else None
        history_changed = expected_history != visible_history

        if not previous_thread_id or history_changed:
            _delete_checkpoint(agent, previous_thread_id)
            thread_id = str(uuid.uuid4())
            agent_input = {"question": message, "history": visible_history}
        else:
            thread_id = previous_thread_id
            agent_input = {"question": message}

        try:
            quest = DetectLanguage(message, online=online)
            logger.debug(f"Language = {quest.lang}, class = {quest}")
        except ValueError as exc:
            logger.error(f"Error processing question: {exc}")
            return f"Error: {exc}", {"thread_id": thread_id, "history": None}

        config = {"configurable": {"thread_id": thread_id}}
        agent_input["question"] = quest.text

        last_output = None
        try:
            for output in agent.stream(agent_input, config=config):
                for _key, value in output.items():
                    last_output = value
        except Exception:
            logger.exception("Unexpected agent stream failure for thread_id=%s", thread_id)
            return STREAM_FAILURE_MESSAGE, {"thread_id": thread_id, "history": None}

        if last_output and isinstance(last_output, dict) and "generation" in last_output:
            answer = str(last_output["generation"])
            updated_history = [
                *visible_history,
                {"role": "user", "content": quest.text},
                {"role": "assistant", "content": answer},
            ]
            return answer, {"thread_id": thread_id, "history": updated_history}

        return "No generation returned from agent.", {"thread_id": thread_id, "history": None}

    return respond


def launch_gradio(cfg: Config):
    server_name, server_port = _server_settings(cfg)
    agent = build_rag_agent(cfg)

    attach_debugger_if_requested()
    respond = _response_callback(agent, online=cfg.online)

    thread_state = gr.State(value=None)

    demo = gr.ChatInterface(
        fn=respond,
        additional_inputs=[thread_state],
        additional_outputs=[thread_state],
        title="RAG Agent",
    )

    demo.queue().launch(
        server_name=server_name,
        server_port=server_port,
        share=cfg.gradio_share,
    )

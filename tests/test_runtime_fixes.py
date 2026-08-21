import importlib
import logging
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import dotenv
import numpy as np
import pytest

import agentic_rag._load_env as runtime_env
import agentic_rag.cli as cli
import agentic_rag.loggers as loggers
import agentic_rag.ui_gradio as ui_gradio

pytestmark = pytest.mark.cpu


def _runtime_config(tmp_path: Path, *, reindex: bool = False) -> SimpleNamespace:
    markdown_dir = tmp_path / "markdown"
    index_dir = tmp_path / "index"
    markdown_dir.mkdir()
    index_dir.mkdir()
    (markdown_dir / "document.md").write_text("# Documento", encoding="utf-8")
    dictionary_path = tmp_path / "dictionary.xlsx"
    dictionary_path.write_bytes(b"dictionary")
    if not reindex:
        (index_dir / "index.faiss").write_bytes(b"faiss")
        (index_dir / "index.pkl").write_bytes(b"metadata")
    return SimpleNamespace(
        md_dir=str(markdown_dir),
        dizionario_path=str(dictionary_path),
        index_dir=str(index_dir),
        reindex=reindex,
        online=False,
        chat=True,
        gradio=False,
    )


def test_effective_hf_mode_honors_config_network_and_offline_flags(monkeypatch):
    """Verify effective hf mode honors config network and offline flags."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert runtime_env.hf_online_enabled(True) is True
    assert runtime_env.hf_online_enabled(False) is False

    monkeypatch.setenv("HF_HUB_OFFLINE", "true")
    assert runtime_env.hf_online_enabled(True) is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "YES")
    assert runtime_env.hf_online_enabled(True) is False


def test_forced_offline_skips_network_probe(monkeypatch):
    """Verify forced offline skips network probe."""
    probe = MagicMock(return_value=True)
    monkeypatch.setattr(runtime_env, "is_online_fast", probe)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    assert runtime_env._network_available(False) is False
    probe.assert_not_called()

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert runtime_env._network_available(True) is False
    probe.assert_not_called()

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert runtime_env._network_available(True) is True
    probe.assert_called_once_with()


def test_network_probe_closes_its_socket(monkeypatch):
    """Verify network probe closes its socket."""
    connection = MagicMock()
    create_connection = MagicMock(return_value=connection)
    monkeypatch.setattr(runtime_env.socket, "create_connection", create_connection)

    assert runtime_env.is_online_fast() is True
    create_connection.assert_called_once_with(("huggingface.co", 443), timeout=0.25)
    connection.close.assert_called_once_with()


def test_config_path_prefers_override_then_working_directory(monkeypatch, tmp_path):
    """Verify config path prefers override then working directory."""
    working_config = tmp_path / "config.yaml"
    working_config.write_text("working: true", encoding="utf-8")
    override = tmp_path / "custom" / "runtime.yaml"
    override.parent.mkdir()
    override.write_text("override: true", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("AGENTIC_RAG_CONFIG", "custom/runtime.yaml")
    assert runtime_env._resolve_config_path() == override

    monkeypatch.delenv("AGENTIC_RAG_CONFIG")
    assert runtime_env._resolve_config_path() == working_config


@pytest.mark.parametrize(
    ("module_name", "dependency_name"),
    [
        ("agentic_rag.llm_build", "transformers"),
        ("agentic_rag.ui_gradio", "gradio"),
    ],
)
def test_runtime_env_loads_before_hf_sensitive_dependencies(module_name, dependency_name):
    """Verify runtime env loads before hf sensitive dependencies."""
    code = textwrap.dedent(f"""
        import builtins
        import socket

        events = []
        original_import = builtins.__import__

        def tracked_import(name, *args, **kwargs):
            if name in {{"agentic_rag._load_env", "{dependency_name}"}}:
                events.append(name)
            return original_import(name, *args, **kwargs)

        def blocked_socket(*args, **kwargs):
            raise AssertionError("forced offline import attempted a socket probe")

        builtins.__import__ = tracked_import
        socket.create_connection = blocked_socket
        __import__("{module_name}")
        assert events.index("agentic_rag._load_env") < events.index("{dependency_name}"), events
    """)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_TOKEN": "test-token",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_import_does_not_patch_numpy_and_is_reload_safe(monkeypatch):
    """Verify runtime import does not patch numpy and is reload safe."""
    native_array = np.array
    create_connection = MagicMock(side_effect=AssertionError("forced offline must not probe"))
    monkeypatch.setattr(runtime_env.socket, "create_connection", create_connection)
    monkeypatch.setattr(dotenv, "load_dotenv", MagicMock())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reloaded = importlib.reload(runtime_env)
    assert np.array is native_array
    assert reloaded.NETWORK_AVAILABLE is False
    assert reloaded.HF_ONLINE is False
    assert reloaded.ONLINE is reloaded.HF_ONLINE
    assert reloaded.os.environ["HF_HUB_OFFLINE"] == "1"
    assert reloaded.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert reloaded.EFFECTIVE_HF_HOME == reloaded.os.environ["HF_HOME"]
    assert reloaded.EFFECTIVE_HF_HUB_CACHE == reloaded.os.environ["HF_HUB_CACHE"]
    create_connection.assert_not_called()


def test_logger_updates_existing_logger_and_deduplicates_factory_handler(monkeypatch):
    """Verify logger updates existing logger and deduplicates factory handler."""
    monkeypatch.setattr(loggers.cfg, "log_level", "CRITICAL")
    monkeypatch.setattr(loggers.cfg, "log_to_file", False)
    monkeypatch.setattr(loggers.Logger, "_console_handler", None)
    target = logging.getLogger("tests.runtime.logger")
    target.handlers.clear()
    unrelated_handler = logging.NullHandler()
    target.addHandler(unrelated_handler)
    target.propagate = True

    assert loggers.Logger.get_logger(target.name) is target
    loggers.Logger.get_logger(target.name)

    factory_handlers = [
        handler for handler in target.handlers if hasattr(handler, loggers.Logger._FACTORY_HANDLER_ATTRIBUTE)
    ]
    assert target.level == logging.CRITICAL
    assert target.propagate is False
    assert unrelated_handler not in target.handlers
    assert len(factory_handlers) == 1
    target.handlers.clear()


def test_logger_respects_real_external_handler(monkeypatch):
    """Verify logger respects real external handler."""
    monkeypatch.setattr(loggers.cfg, "log_level", "INFO")
    monkeypatch.setattr(loggers.cfg, "log_to_file", False)
    target = logging.getLogger("tests.runtime.external_logger")
    target.handlers.clear()
    external_handler = logging.StreamHandler()
    target.addHandler(external_handler)

    loggers.Logger.get_logger(target.name)
    loggers.Logger.get_logger(target.name)

    assert target.handlers == [external_handler]
    assert target.propagate is False
    target.handlers.clear()


def test_logger_initialization_is_thread_safe(monkeypatch):
    """Verify logger initialization is thread safe."""
    monkeypatch.setattr(loggers.cfg, "log_level", "INFO")
    monkeypatch.setattr(loggers.cfg, "log_to_file", False)
    monkeypatch.setattr(loggers.Logger, "_console_handler", None)
    target = logging.getLogger("tests.runtime.threaded_logger")
    target.handlers.clear()

    with ThreadPoolExecutor(max_workers=8) as executor:
        created = list(executor.map(lambda _index: loggers.Logger.get_logger(target.name), range(32)))

    factory_handlers = [
        handler for handler in target.handlers if hasattr(handler, loggers.Logger._FACTORY_HANDLER_ATTRIBUTE)
    ]
    assert created == [target] * 32
    assert len(factory_handlers) == 1
    target.handlers.clear()


def test_file_logger_creates_parent_directories(monkeypatch, tmp_path, enabled_test_logging):
    """Verify file logger creates parent directories."""
    log_path = tmp_path / "nested" / "logs" / "runtime.log"
    monkeypatch.setattr(loggers.cfg, "log_level", "INFO")
    monkeypatch.setattr(loggers.cfg, "log_to_file", True)
    monkeypatch.setattr(loggers.Logger, "LOG_FILE", str(log_path))
    monkeypatch.setattr(loggers.Logger, "_file_handler", None)
    target = logging.getLogger("tests.runtime.file_logger")
    target.handlers.clear()

    loggers.Logger.get_logger(target.name)

    assert log_path.parent.is_dir()
    assert not log_path.exists()
    target.info("create delayed log")
    assert log_path.is_file()
    for handler in target.handlers:
        handler.close()
    target.handlers.clear()


def test_cli_rejects_missing_resources_before_agent_construction(monkeypatch, tmp_path):
    """Verify cli rejects missing resources before agent construction."""
    config = _runtime_config(tmp_path)
    Path(config.dizionario_path).unlink()
    monkeypatch.setattr(cli, "cfg", config)
    build_index = MagicMock()
    chat = MagicMock()
    gradio = MagicMock()
    monkeypatch.setattr(cli, "build_faiss_index", build_index)
    monkeypatch.setattr(cli, "interactive_loop", chat)
    monkeypatch.setattr(cli, "launch_gradio", gradio)

    with pytest.raises(FileNotFoundError, match="Dictionary file not found"):
        cli.main()
    build_index.assert_not_called()
    chat.assert_not_called()
    gradio.assert_not_called()


def test_cli_requires_both_expected_index_files(tmp_path):
    """Verify cli requires both expected index files."""
    config = _runtime_config(tmp_path)
    Path(config.index_dir, "index.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="index.pkl"):
        cli._validate_index_resources(config)


def test_cli_requires_markdown_at_top_level(tmp_path):
    """Verify cli requires markdown at top level."""
    markdown_dir = tmp_path / "markdown"
    nested_dir = markdown_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "document.md").write_text("# Nested", encoding="utf-8")
    dictionary_path = tmp_path / "dictionary.xlsx"
    dictionary_path.write_bytes(b"dictionary")
    config = SimpleNamespace(md_dir=str(markdown_dir), dizionario_path=str(dictionary_path))

    with pytest.raises(FileNotFoundError, match="No top-level Markdown files"):
        cli._validate_source_resources(config)


def test_cli_mode_override_does_not_mutate_config(monkeypatch, tmp_path):
    """Verify cli mode override does not mutate config."""
    config = _runtime_config(tmp_path)
    monkeypatch.setattr(cli, "cfg", config)
    monkeypatch.setenv("AGENTIC_RAG_MODE", "gradio")
    chat = MagicMock()
    gradio = MagicMock()
    monkeypatch.setattr(cli, "interactive_loop", chat)
    monkeypatch.setattr(cli, "launch_gradio", gradio)

    cli.main()

    chat.assert_not_called()
    gradio.assert_called_once_with(config)
    assert config.chat is True
    assert config.gradio is False


def test_cli_validates_rebuilt_index_before_starting_agent(monkeypatch, tmp_path):
    """Verify cli validates rebuilt index before starting agent."""
    config = _runtime_config(tmp_path, reindex=True)
    monkeypatch.setattr(cli, "cfg", config)

    def build_index(_config):
        Path(config.index_dir, "index.faiss").write_bytes(b"faiss")
        Path(config.index_dir, "index.pkl").write_bytes(b"metadata")

    build = MagicMock(side_effect=build_index)
    chat = MagicMock()
    monkeypatch.setattr(cli, "build_faiss_index", build)
    monkeypatch.setattr(cli, "interactive_loop", chat)

    cli.main()

    build.assert_called_once_with(config)
    chat.assert_called_once_with(config)


def test_cli_passes_runtime_online_policy_to_language_detector(monkeypatch):
    """Verify cli passes runtime online policy to language detector."""
    config = SimpleNamespace(online=False)
    agent = SimpleNamespace(stream=MagicMock(return_value=[{"answer": {"generation": "risposta"}}]))
    question = SimpleNamespace(lang="it", text="ciao")
    detector = MagicMock(return_value=question)
    stdin = SimpleNamespace(buffer=SimpleNamespace(readline=MagicMock(side_effect=[b"ciao\n", b"q\n"])))
    monkeypatch.setattr(cli, "build_rag_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(cli, "DetectLanguage", detector)
    monkeypatch.setattr(cli.sys, "stdin", stdin)

    cli.interactive_loop(config)

    detector.assert_called_once_with("ciao", online=False)


def test_cli_continues_after_agent_stream_failure(monkeypatch):
    """Verify cli continues after agent stream failure."""
    config = SimpleNamespace(online=False)
    agent = SimpleNamespace(stream=MagicMock(side_effect=RuntimeError("stream failed")))
    question = SimpleNamespace(lang="it", text="ciao")
    stdin = SimpleNamespace(buffer=SimpleNamespace(readline=MagicMock(side_effect=[b"ciao\n", b"q\n"])))
    monkeypatch.setattr(cli, "build_rag_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(cli, "DetectLanguage", MagicMock(return_value=question))
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    log_exception = MagicMock()
    monkeypatch.setattr(cli.logger, "exception", log_exception)

    cli.interactive_loop(config)

    log_exception.assert_called_once_with("Error during agent execution")


def test_cli_without_runtime_mode_returns_before_resource_validation(monkeypatch):
    """Verify cli without runtime mode returns before resource validation."""
    config = SimpleNamespace(online=False, chat=False, gradio=False)
    monkeypatch.setattr(cli, "cfg", config)
    validate = MagicMock()
    monkeypatch.setattr(cli, "_validate_source_resources", validate)

    cli.main()

    validate.assert_not_called()


def test_gradio_environment_overrides_and_validates_yaml(monkeypatch):
    """Verify gradio environment overrides and validates yaml."""
    config = SimpleNamespace(gradio_host="0.0.0.0", gradio_port=7860)
    monkeypatch.setenv("GRADIO_SERVER_NAME", "127.0.0.1")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8123")
    assert ui_gradio._server_settings(config) == ("127.0.0.1", 8123)

    monkeypatch.setenv("GRADIO_SERVER_PORT", "70000")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        ui_gradio._server_settings(config)
    monkeypatch.setenv("GRADIO_SERVER_PORT", "not-a-port")
    with pytest.raises(ValueError, match="Invalid Gradio server port"):
        ui_gradio._server_settings(config)
    monkeypatch.setenv("GRADIO_SERVER_PORT", "8123")
    monkeypatch.setenv("GRADIO_SERVER_NAME", "https://invalid.example")
    with pytest.raises(ValueError, match="Invalid Gradio server host"):
        ui_gradio._server_settings(config)


def test_gradio_state_is_initialized_lazily_per_session(monkeypatch):
    """Verify gradio state is initialized lazily per session."""
    config = SimpleNamespace(
        online=False,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        gradio_share=False,
    )
    state = object()
    state_factory = MagicMock(return_value=state)
    demo = MagicMock()
    demo.queue.return_value = demo
    chat_interface = MagicMock(return_value=demo)
    monkeypatch.setattr(ui_gradio, "build_rag_agent", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ui_gradio, "attach_debugger_if_requested", MagicMock())
    monkeypatch.setattr(ui_gradio.gr, "State", state_factory)
    monkeypatch.setattr(ui_gradio.gr, "ChatInterface", chat_interface)

    ui_gradio.launch_gradio(config)

    state_factory.assert_called_once_with(value=None)
    callback = chat_interface.call_args.kwargs["fn"]
    assert callback.__name__ == "respond"
    demo.launch.assert_called_once_with(server_name="127.0.0.1", server_port=7860, share=False)


def test_gradio_callback_logs_stream_failure_and_preserves_thread(monkeypatch):
    """Verify gradio callback logs stream failure and preserves thread."""
    quest = SimpleNamespace(lang="it", text="domanda")
    monkeypatch.setattr(ui_gradio, "DetectLanguage", MagicMock(return_value=quest))
    stream_error = RuntimeError("stream failed")
    agent = SimpleNamespace(stream=MagicMock(side_effect=stream_error))
    log_exception = MagicMock()
    monkeypatch.setattr(ui_gradio.logger, "exception", log_exception)
    respond = ui_gradio._response_callback(agent, online=False)

    visible_history = [
        {"role": "user", "content": "precedente"},
        {"role": "assistant", "content": "risposta"},
    ]
    session = {"thread_id": "stable-thread", "history": visible_history}
    message, returned_session = respond("domanda", [["precedente", "risposta"]], session)

    assert message == ui_gradio.STREAM_FAILURE_MESSAGE
    assert returned_session == {"thread_id": "stable-thread", "history": None}
    log_exception.assert_called_once_with("Unexpected agent stream failure for thread_id=%s", "stable-thread")
    ui_gradio.DetectLanguage.assert_called_once_with("domanda", online=False)


def test_gradio_thread_ids_are_per_session_and_rotate_after_clear(monkeypatch):
    """Verify gradio thread ids are per session and rotate after clear."""
    monkeypatch.setattr(
        ui_gradio,
        "DetectLanguage",
        MagicMock(side_effect=lambda text, **_kwargs: SimpleNamespace(lang="it", text=text)),
    )
    agent = SimpleNamespace(stream=MagicMock(return_value=[{"answer": {"generation": "risposta"}}]))
    generated_ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000003",
    ]
    uuid4 = MagicMock(side_effect=generated_ids)
    monkeypatch.setattr(ui_gradio.uuid, "uuid4", uuid4)
    respond = ui_gradio._response_callback(agent, online=False)

    _message, first_session = respond("prima", [], None)
    _message, second_session = respond("seconda", [], None)
    _message, continued_session = respond("continua", [["prima", "risposta"]], first_session)
    _message, cleared_session = respond("nuova", [], first_session)

    assert first_session["thread_id"] == generated_ids[0]
    assert second_session["thread_id"] == generated_ids[1]
    assert first_session["thread_id"] != second_session["thread_id"]
    assert continued_session["thread_id"] == first_session["thread_id"]
    assert cleared_session["thread_id"] == generated_ids[2]
    assert cleared_session["thread_id"] != first_session["thread_id"]
    assert uuid4.call_count == 3


def test_gradio_shortened_history_forks_and_reseeds_checkpoint(monkeypatch):
    """Verify gradio shortened history forks and reseeds checkpoint."""
    monkeypatch.setattr(
        ui_gradio,
        "DetectLanguage",
        MagicMock(side_effect=lambda text, **_kwargs: SimpleNamespace(lang="it", text=text)),
    )
    delete_thread = MagicMock()
    agent = SimpleNamespace(
        stream=MagicMock(return_value=[{"answer": {"generation": "nuova risposta"}}]),
        checkpointer=SimpleNamespace(delete_thread=delete_thread),
    )
    monkeypatch.setattr(ui_gradio.uuid, "uuid4", MagicMock(return_value="new-thread"))
    respond = ui_gradio._response_callback(agent, online=False)
    old_history = [
        {"role": "user", "content": "prima"},
        {"role": "assistant", "content": "risposta uno"},
        {"role": "user", "content": "seconda"},
        {"role": "assistant", "content": "risposta due"},
    ]

    _message, session = respond(
        "riprova",
        [["prima", "risposta uno"]],
        {"thread_id": "old-thread", "history": old_history},
    )

    delete_thread.assert_called_once_with("old-thread")
    agent.stream.assert_called_once_with(
        {
            "question": "riprova",
            "history": [
                {"role": "user", "content": "prima"},
                {"role": "assistant", "content": "risposta uno"},
            ],
        },
        config={"configurable": {"thread_id": "new-thread"}},
    )
    assert session["thread_id"] == "new-thread"


@pytest.mark.parametrize("terminal_event", ["eof", "interrupt"])
def test_cli_exits_cleanly_before_detection_on_terminal_end(monkeypatch, terminal_event):
    """Verify EOF and keyboard interruption end chat without invoking the agent."""
    config = SimpleNamespace(online=False)
    agent = SimpleNamespace(stream=MagicMock())
    readline = MagicMock(return_value=b"")
    if terminal_event == "interrupt":
        readline.side_effect = KeyboardInterrupt
    stdin = SimpleNamespace(buffer=SimpleNamespace(readline=readline))
    detector = MagicMock()
    monkeypatch.setattr(cli, "build_rag_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(cli, "DetectLanguage", detector)
    monkeypatch.setattr(cli.sys, "stdin", stdin)

    cli.interactive_loop(config)

    detector.assert_not_called()
    agent.stream.assert_not_called()


def test_cli_reports_language_rejection_and_continues_to_exit(monkeypatch):
    """Verify invalid-language input is reported without invoking the graph."""
    config = SimpleNamespace(online=False)
    agent = SimpleNamespace(stream=MagicMock())
    stdin = SimpleNamespace(buffer=SimpleNamespace(readline=MagicMock(side_effect=[b"hello\n", b"q\n"])))
    detector = MagicMock(side_effect=ValueError("Italian only"))
    console = MagicMock()
    monkeypatch.setattr(cli, "build_rag_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(cli, "DetectLanguage", detector)
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli, "console", console)

    cli.interactive_loop(config)

    detector.assert_called_once_with("hello", online=False)
    console.print.assert_any_call("[red]Error: Italian only[/red]")
    agent.stream.assert_not_called()
    assert stdin.buffer.readline.call_count == 2


def test_cli_reports_when_agent_stream_has_no_generation(monkeypatch):
    """Verify a completed stream without generation produces the fallback notice."""
    config = SimpleNamespace(online=False)
    agent = SimpleNamespace(stream=MagicMock(return_value=[{"node": {"documents": []}}]))
    quest = SimpleNamespace(lang="it", text="domanda")
    stdin = SimpleNamespace(buffer=SimpleNamespace(readline=MagicMock(side_effect=[b"domanda\n", b"q\n"])))
    console = MagicMock()
    monkeypatch.setattr(cli, "build_rag_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(cli, "DetectLanguage", MagicMock(return_value=quest))
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli, "console", console)

    cli.interactive_loop(config)

    console.print.assert_any_call("[yellow]No generation returned from agent.[/yellow]")


def test_cli_rejects_invalid_runtime_mode_override(monkeypatch):
    """Verify unsupported runtime mode overrides fail before startup dispatch."""
    config = SimpleNamespace(chat=True, gradio=False)
    monkeypatch.setenv("AGENTIC_RAG_MODE", "tui")

    with pytest.raises(ValueError, match="AGENTIC_RAG_MODE"):
        cli._runtime_mode(config)


def test_gradio_callback_reports_language_rejection_without_streaming(monkeypatch):
    """Verify Gradio returns validation errors without invoking the graph stream."""
    detector = MagicMock(side_effect=ValueError("Italian only"))
    agent = SimpleNamespace(stream=MagicMock())
    monkeypatch.setattr(ui_gradio, "DetectLanguage", detector)
    monkeypatch.setattr(ui_gradio.uuid, "uuid4", MagicMock(return_value="new-thread"))
    respond = ui_gradio._response_callback(agent, online=False)

    answer, session = respond("hello", [], None)

    assert answer == "Error: Italian only"
    assert session == {"thread_id": "new-thread", "history": None}
    detector.assert_called_once_with("hello", online=False)
    agent.stream.assert_not_called()


def test_gradio_callback_reports_missing_generation(monkeypatch):
    """Verify Gradio returns its fallback when the stream has no generated answer."""
    quest = SimpleNamespace(lang="it", text="domanda")
    agent = SimpleNamespace(stream=MagicMock(return_value=[{"node": {"documents": []}}]))
    monkeypatch.setattr(ui_gradio, "DetectLanguage", MagicMock(return_value=quest))
    monkeypatch.setattr(ui_gradio.uuid, "uuid4", MagicMock(return_value="new-thread"))
    respond = ui_gradio._response_callback(agent, online=False)

    answer, session = respond("domanda", [["prima", "risposta"]], None)

    assert answer == "No generation returned from agent."
    assert session == {"thread_id": "new-thread", "history": None}
    agent.stream.assert_called_once_with(
        {
            "question": "domanda",
            "history": [
                {"role": "user", "content": "prima"},
                {"role": "assistant", "content": "risposta"},
            ],
        },
        config={"configurable": {"thread_id": "new-thread"}},
    )


def test_gradio_history_normalization_accepts_messages_and_ignores_malformed_entries():
    """Verify Gradio history normalization handles supported forms and skips malformed data."""
    history = [
        {"role": "user", "content": 1},
        {"role": "system", "content": "ignored"},
        ["question", "answer"],
        (None, "assistant only"),
        ["invalid length"],
        object(),
    ]

    assert ui_gradio._normalize_history(history) == [
        {"role": "user", "content": "1"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": "assistant only"},
    ]

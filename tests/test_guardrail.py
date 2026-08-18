import pytest
from langchain_core.documents import Document

from agentic_rag.detect_language import DetectLanguage
from agentic_rag.graph import build_agent_graph

pytestmark = pytest.mark.cpu  # Mark ALL tests in this module as CPU


def test_language_detection_relaxation():
    # Verify that Italian inputs work as well
    q_it = DetectLanguage("Buongiorno, vorrei informazioni sulla preparazione.")
    assert q_it.lang == "it"
    assert q_it.text == "Buongiorno, vorrei informazioni sulla preparazione."

    # Verify that totally unsupported languages still raise ValueError (e.g. English, French, German)
    with pytest.raises(ValueError, match="Only italian is supported."):
        DetectLanguage("Hello. What is the preparation for Colon-TC?")

    with pytest.raises(ValueError, match="Only italian is supported."):
        DetectLanguage("Bonjour! Comment se préparer pour le Colon-TC?")

    with pytest.raises(ValueError, match="Only italian is supported."):
        DetectLanguage("Guten Tag, ich möchte Informationen über die Vorbereitung haben.")


def test_graph_routing_thanks(mock_ctx, config_data):
    mock_ctx.guardrail_chain.invoke.return_value = "GRAZIE"

    question = "Grazie per le informazioni!"

    agent = build_agent_graph(mock_ctx, config_data)
    state = {"question": question}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "GRAZIE"
    assert "generation" in result
    assert "Di nulla, sono qui per aiutarti." in result["generation"]

    mock_ctx.sanitizer_chain.invoke.assert_called_once()
    mock_ctx.sanitizer_chain.invoke.assert_called_with(state)
    mock_ctx.rag_chain.invoke.assert_not_called()
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_hello(mock_ctx, config_data):
    mock_ctx.guardrail_chain.invoke.return_value = "SALUTO"

    question = "Ciao, vorrei fare una domanda"

    agent = build_agent_graph(mock_ctx, config_data)
    state = {"question": question}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "SALUTO"
    assert "generation" in result
    assert "Ciao, sono un agente IA." in result["generation"]

    mock_ctx.sanitizer_chain.invoke.assert_called_once()
    mock_ctx.sanitizer_chain.invoke.assert_called_with(state)
    mock_ctx.rag_chain.invoke.assert_not_called()
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_off_topic(mock_ctx, config_data):
    mock_ctx.guardrail_chain.invoke.return_value = "OFF_TOPIC"

    question = "Come si prepara la carbonara?"

    agent = build_agent_graph(mock_ctx, config_data)
    state = {"question": question}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "OFF_TOPIC"
    assert "generation" in result
    assert "Spiacente, non posso aiutarti." in result["generation"]

    mock_ctx.sanitizer_chain.invoke.assert_called_once()
    mock_ctx.sanitizer_chain.invoke.assert_called_with(state)
    mock_ctx.rag_chain.invoke.assert_not_called()
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_on_topic(mock_ctx, config_data):
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"

    question = "Qual è la preparazione per la Colon-TC?"

    agent = build_agent_graph(mock_ctx, config_data)
    state = {"question": question}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "ON_TOPIC"
    # It should have run sanitize_question, retriever, and generate_with_docs
    mock_ctx.sanitizer_chain.invoke.assert_called_once()
    mock_ctx.sanitizer_chain.invoke.assert_called_with(state)
    mock_ctx.retriever.invoke.assert_called()
    mock_ctx.rag_chain.invoke.assert_called_once()
    assert result["generation"] == "Mocked RAG response"


def test_guardrail_branch_resets_transient_state_and_preserves_history(mock_ctx, config_data):
    agent = build_agent_graph(mock_ctx, config_data)
    config = {"configurable": {"thread_id": "test-reset-thread"}}
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"

    first = agent.invoke({"question": "Come mi preparo?"}, config=config)
    first_history = first["history"]
    assert first_history

    mock_ctx.guardrail_chain.invoke.return_value = "SALUTO"
    second = agent.invoke({"question": "Ciao"}, config=config)

    assert second["guardrail_status"] == "SALUTO"
    assert second["documents"] == []
    assert second["rewrite_count"] == 0
    assert second["has_docs"] is False
    assert second["history"] == first_history


def test_graph_uses_passed_rerank_cleaning_and_memory_config(mock_ctx, config_data):
    cfg = config_data.model_copy(
        update={
            "rerank": False,
            "threshold": 100.0,
            "clean_answer": True,
            "max_history_turns": 0,
        }
    )
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"
    mock_ctx.retriever.invoke.return_value = [Document(page_content="contenuto", metadata={})]
    mock_ctx.rag_chain.invoke.return_value = "Risposta grezza (Doc 2)"
    mock_ctx.cleaner_chain.invoke.return_value = "Risposta pulita"
    agent = build_agent_graph(mock_ctx, cfg)

    result = agent.invoke(
        {"question": "Come mi preparo?"},
        config={"configurable": {"thread_id": "test-passed-config"}},
    )

    assert result["generation"] == "Risposta pulita"
    assert result["history"] == []
    mock_ctx.cleaner_chain.invoke.assert_called_once_with({"answer": "Risposta grezza (Doc 2)"})


def test_graph_uses_passed_rerank_threshold(mock_ctx, config_data):
    cfg = config_data.model_copy(update={"threshold": 0.95, "rerank": True})
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"
    agent = build_agent_graph(mock_ctx, cfg)

    result = agent.invoke(
        {"question": "Come mi preparo?"},
        config={"configurable": {"thread_id": "test-passed-threshold"}},
    )

    assert result["has_docs"] is False
    assert "Non sono riuscito" in result["generation"]
    mock_ctx.rag_chain.invoke.assert_not_called()

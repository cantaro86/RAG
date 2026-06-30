from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from agentic_rag.detect_language import DetectLanguage
from agentic_rag.graph import RAGContext, build_agent_graph


@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=RAGContext)

    # Mock all chains to return basic mock values
    ctx.guardrail_chain = MagicMock()
    ctx.sanitizer_chain = MagicMock()
    ctx.sanitizer_chain.invoke.return_value = "Sanitized query"
    ctx.topic_continuity_classifier = MagicMock()
    ctx.topic_continuity_classifier.invoke.return_value = "SAME"
    ctx.pre_retrieval_question_rewriter = MagicMock()
    ctx.pre_retrieval_question_rewriter.invoke.return_value = "Rewritten query"
    ctx.question_transformer = MagicMock()
    ctx.question_transformer.invoke.return_value = "Transformed query"
    ctx.rag_chain = MagicMock()
    ctx.rag_chain.invoke.return_value = "Mocked RAG response"
    ctx.cleaner_chain = MagicMock()
    ctx.cleaner_chain.invoke.return_value = "Cleaned mocked RAG response"

    # Mock retriever
    ctx.retriever = MagicMock()
    mock_doc = Document(
        page_content="Istruzioni per pazienti esame Colon-TC.",
        metadata={"rerank_score": 0.9, "source": "Informazioni_per_pazienti.md"},
    )
    ctx.retriever.invoke.return_value = [mock_doc]

    # Mock synonyms
    ctx.synonyms = MagicMock()
    ctx.synonyms.find_matched_terms.return_value = []

    return ctx


def test_language_detection_relaxation():
    # Verify that English inputs do not raise ValueError anymore
    q_en = DetectLanguage("Hello, how are you?")
    assert q_en.lang == "en"
    assert q_en.text == "Hello, how are you?"

    # Verify that Italian inputs work as well
    q_it = DetectLanguage("Buongiorno, vorrei informazioni sulla preparazione.")
    assert q_it.lang == "it"
    assert q_it.text == "Buongiorno, vorrei informazioni sulla preparazione."

    # Verify that totally unsupported languages still raise ValueError (e.g. French, German)
    # Note: simple short words like "Bonjour" or random characters might default to it or en or unknown.
    # We will test a long German sentence.
    with pytest.raises(ValueError, match="Only Italian and English are supported"):
        DetectLanguage("Guten Tag, ich möchte Informationen über die Vorbereitung haben.")


def test_graph_routing_greeting_english(mock_ctx):
    mock_ctx.guardrail_chain.invoke.return_value = "GREETING"

    agent = build_agent_graph(mock_ctx)
    state = {"question": "Hello, I want to ask about Colon-TC"}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "GREETING"
    assert result["language"] == "en"
    assert "generation" in result
    assert "Hello, I am an AI agent." in result["generation"]
    # Check that it did not run RAG or retriever
    mock_ctx.retriever.invoke.assert_not_called()
    mock_ctx.rag_chain.invoke.assert_not_called()


def test_graph_routing_greeting_italian(mock_ctx):
    mock_ctx.guardrail_chain.invoke.return_value = "GREETING"

    agent = build_agent_graph(mock_ctx)
    state = {"question": "Ciao, vorrei fare una domanda"}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "GREETING"
    assert result["language"] == "it"
    assert "generation" in result
    assert "Ciao, sono un agente IA." in result["generation"]
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_off_topic_english(mock_ctx):
    mock_ctx.guardrail_chain.invoke.return_value = "OFF_TOPIC"

    agent = build_agent_graph(mock_ctx)
    state = {"question": "What is the capital of France?"}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "OFF_TOPIC"
    assert result["language"] == "en"
    assert "generation" in result
    assert "Sorry, I cannot help you." in result["generation"]
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_off_topic_italian(mock_ctx):
    mock_ctx.guardrail_chain.invoke.return_value = "OFF_TOPIC"

    agent = build_agent_graph(mock_ctx)
    state = {"question": "Come si prepara la carbonara?"}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "OFF_TOPIC"
    assert result["language"] == "it"
    assert "generation" in result
    assert "Spiacente, non posso aiutarti." in result["generation"]
    mock_ctx.retriever.invoke.assert_not_called()


def test_graph_routing_on_topic_italian(mock_ctx):
    # For ON_TOPIC, it should pass through to sanitize_question and normal RAG
    mock_ctx.guardrail_chain.invoke.return_value = "ON_TOPIC"

    agent = build_agent_graph(mock_ctx)
    state = {"question": "Qual è la preparazione per la Colon-TC?"}
    config = {"configurable": {"thread_id": "test-thread"}}

    result = agent.invoke(state, config=config)

    assert result["guardrail_status"] == "ON_TOPIC"
    assert result["language"] == "it"
    # It should have run sanitize_question, retriever, and generate_with_docs
    mock_ctx.sanitizer_chain.invoke.assert_called_once()
    mock_ctx.retriever.invoke.assert_called()
    mock_ctx.rag_chain.invoke.assert_called_once()
    assert result["generation"] == "Mocked RAG response"

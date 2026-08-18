from copy import deepcopy
from unittest.mock import MagicMock, call

import pytest
from langchain_core.documents import Document

from agentic_rag.graph import (
    clean_answer,
    format_history,
    generate_with_docs,
    pre_retrieval_rewriter,
    push_memory,
    retrieve_and_filter,
    sanitize_question,
    transform_query,
)
from agentic_rag.state import GraphState
from agentic_rag.utils import PATIENT_DOC

pytestmark = pytest.mark.cpu


def test_graph_state_requires_only_question():
    assert GraphState.__required_keys__ == frozenset({"question"})


def test_sanitize_captures_filter_and_resets_transient_state_without_losing_history():
    history = [
        {"role": "user", "content": "Domanda precedente"},
        {"role": "assistant", "content": "Risposta precedente"},
    ]
    stale_doc = Document(page_content="vecchio", metadata={})
    state = {
        "question": "Usa le informazioni per pazienti sulla preparazione",
        "history": history,
        "documents": [stale_doc],
        "rewrite_count": 2,
        "generation": "vecchia risposta",
        "first_question": True,
        "has_docs": True,
        "topic_status": "STESSO",
        "guardrail_status": "ON_TOPIC",
        "source_filter": {"source": "vecchio.md"},
    }
    sanitizer = MagicMock()
    sanitizer.invoke.return_value = "Come devo prepararmi?"

    result = sanitize_question(state, sanitizer)

    assert result["question"] == "Come devo prepararmi?"
    assert result["original_question"] == "Come devo prepararmi?"
    assert result["source_filter"] == {"source": PATIENT_DOC}
    assert result["history"] == history
    assert result["documents"] == []
    assert result["rewrite_count"] == 0
    assert result["generation"] is None
    assert result["first_question"] is False
    assert result["has_docs"] is False
    assert result["topic_status"] is None
    assert result["guardrail_status"] is None


def test_source_filter_survives_query_rewrite_and_retry():
    source_filter = {"source": PATIENT_DOC}
    retriever = MagicMock()
    retriever.invoke.return_value = []
    initial = {"question": "domanda originale", "source_filter": source_filter, "rewrite_count": 0}

    first = retrieve_and_filter(initial, retriever, threshold=0.1, rerank=False)
    transformer = MagicMock()
    transformer.invoke.return_value = "domanda riscritta"
    synonyms = MagicMock()
    synonyms.find_matched_terms.return_value = []
    rewritten = transform_query(first, transformer, synonyms)
    second = retrieve_and_filter(rewritten, retriever, threshold=0.1, rerank=False)

    assert second["source_filter"] == source_filter
    assert retriever.invoke.call_args_list == [
        call("domanda originale", filter=source_filter),
        call("domanda riscritta", filter=source_filter),
    ]


def test_source_filter_is_derived_from_first_rewritten_question_and_persisted_on_retry():
    source_filter = {"source": PATIENT_DOC}
    pre_rewriter = MagicMock()
    pre_rewriter.invoke.return_value = "Cosa dicono le informazioni per pazienti?"
    rewritten = pre_retrieval_rewriter(
        {
            "question": "E per me?",
            "source_filter": None,
            "rewrite_count": 0,
            "history": [],
        },
        pre_rewriter,
        max_history_turns=2,
    )
    retriever = MagicMock()
    retriever.invoke.return_value = []

    first = retrieve_and_filter(rewritten, retriever, threshold=0.1, rerank=False)
    transformer = MagicMock()
    transformer.invoke.return_value = "domanda di recupero senza indicazione del corpus"
    synonyms = MagicMock()
    synonyms.find_matched_terms.return_value = []
    retry = transform_query(first, transformer, synonyms)
    second = retrieve_and_filter(retry, retriever, threshold=0.1, rerank=False)

    assert first["source_filter"] == source_filter
    assert second["source_filter"] == source_filter
    assert retriever.invoke.call_args_list == [
        call("Cosa dicono le informazioni per pazienti?", filter=source_filter),
        call("domanda di recupero senza indicazione del corpus", filter=source_filter),
    ]


def test_rerank_disabled_accepts_faiss_documents_without_scores_or_mutation():
    documents = [Document(page_content="uno", metadata={"source": "a.md"})]
    original_metadata = deepcopy(documents[0].metadata)
    retriever = MagicMock()
    retriever.invoke.return_value = documents

    result = retrieve_and_filter(
        {"question": "domanda", "source_filter": None},
        retriever,
        threshold=0.9,
        rerank=False,
    )

    assert result["has_docs"] is True
    assert result["documents"] == documents
    assert documents[0].metadata == original_metadata
    assert "rerank_score" not in documents[0].metadata


def test_rerank_enabled_requires_score_and_applies_threshold():
    unscored = Document(page_content="senza punteggio", metadata={})
    low = Document(page_content="basso", metadata={"rerank_score": 0.2})
    high = Document(page_content="alto", metadata={"rerank_score": 0.8})
    retriever = MagicMock()
    retriever.invoke.return_value = [unscored, low, high]

    result = retrieve_and_filter(
        {"question": "domanda", "source_filter": None},
        retriever,
        threshold=0.5,
        rerank=True,
    )

    assert result["documents"] == [high]


def test_zero_history_disables_memory_and_push_does_not_mutate_input():
    history = [
        {"role": "user", "content": "prima"},
        {"role": "assistant", "content": "seconda"},
    ]
    state = {"question": "nuova", "history": history}
    original = deepcopy(state)

    assert push_memory(state, "nuova", "risposta", max_history_turns=0) == []
    assert format_history(history, max_history_turns=0) == "No prior conversation."
    assert state == original

    pushed = push_memory(state, "nuova", "risposta", max_history_turns=1)
    assert pushed == [
        {"role": "user", "content": "nuova"},
        {"role": "assistant", "content": "risposta"},
    ]
    assert state == original


def test_history_uses_sanitized_question_and_final_cleaned_answer():
    rag_chain = MagicMock()
    rag_chain.invoke.return_value = "Risposta grezza (Doc 2)"
    generated = generate_with_docs(
        {
            "question": "domanda riscritta per il recupero",
            "original_question": "domanda corretta dell'utente",
            "documents": [],
            "history": [],
        },
        rag_chain,
    )
    cleaner = MagicMock()
    cleaner.invoke.return_value = "  Risposta visibile  "

    result = clean_answer(
        generated,
        cleaner,
        clean_answer_enabled=True,
        max_history_turns=2,
    )

    assert result["generation"] == "Risposta visibile"
    assert result["history"] == [
        {"role": "user", "content": "domanda corretta dell'utente"},
        {"role": "assistant", "content": "Risposta visibile"},
    ]
    cleaner.invoke.assert_called_once_with({"answer": "Risposta grezza (Doc 2)"})

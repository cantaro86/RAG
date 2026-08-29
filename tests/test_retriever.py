from unittest.mock import MagicMock, call

import pytest
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

import agentic_rag.retriever as retriever_module
from agentic_rag.retriever import (
    FilterableRerankerRetriever,
    MPSSentenceCrossEncoder,
    ScoredCrossEncoderReranker,
    build_retriever,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "search_type, expected_kwargs",
    [
        ("similarity", {"k": 5, "fetch_k": 5}),
        ("mmr", {"k": 5, "fetch_k": 5, "lambda_mult": 0.25}),
    ],
)
def test_plain_retriever_uses_safe_fetch_k_for_every_search_type(search_type, expected_kwargs):
    """Verify plain retriever uses safe fetch k for every search type."""
    vectorstore = MagicMock()
    expected = object()
    vectorstore.as_retriever.return_value = expected

    result = build_retriever(
        vectorstore,
        k=5,
        rerank_model=None,
        k_reranked=2,
        search_type=search_type,
        fetch_k=2,
        lambda_mult=0.25,
        online=False,
        cache_folder="/configured/cache",
    )

    assert result is expected
    vectorstore.as_retriever.assert_called_once_with(search_type=search_type, search_kwargs=expected_kwargs)


@pytest.mark.parametrize(
    "search_type, method_name, expected_kwargs",
    [
        ("similarity", "similarity_search", {"k": 6, "fetch_k": 6}),
        ("mmr", "max_marginal_relevance_search", {"k": 6, "fetch_k": 6, "lambda_mult": 0.4}),
    ],
)
def test_reranking_retriever_uses_safe_fetch_k_for_faiss(search_type, method_name, expected_kwargs):
    """Verify reranking retriever uses safe fetch k for faiss."""
    vectorstore = MagicMock()
    document = Document(page_content="contenuto", metadata={})
    getattr(vectorstore, method_name).return_value = [document]
    compressor = MagicMock()
    compressor.compress_documents.return_value = [document]
    retriever = FilterableRerankerRetriever.model_construct(
        vs=vectorstore,
        k=6,
        compressor=compressor,
        search_type=search_type,
        fetch_k=3,
        lambda_mult=0.4,
    )

    result = retriever._get_relevant_documents("domanda")

    assert result == [document]
    getattr(vectorstore, method_name).assert_called_once_with("domanda", **expected_kwargs)
    compressor.compress_documents.assert_called_once_with([document], "domanda")


@pytest.mark.parametrize("online, effective_online", [(True, True), (True, False), (False, False)])
def test_cross_encoder_uses_supported_cache_and_local_only_arguments(monkeypatch, online, effective_online):
    """Verify cross encoder uses supported cache and local only arguments."""
    cross_encoder = MagicMock()
    policy = MagicMock(return_value=effective_online)
    monkeypatch.setattr(retriever_module, "hf_online_enabled", policy)
    monkeypatch.setattr(retriever_module, "CrossEncoder", cross_encoder)

    result = MPSSentenceCrossEncoder(
        "rerank-model",
        online=online,
        cache_folder="/configured/cache",
    )

    assert result.model is cross_encoder.return_value
    cross_encoder.assert_called_once_with(
        "rerank-model",
        device=retriever_module.DEVICE,
        max_length=2048,
        local_files_only=not effective_online,
        model_kwargs={"cache_dir": "/configured/cache"},
        processor_kwargs={"cache_dir": "/configured/cache"},
    )
    policy.assert_called_once_with(online)


def test_table_lookup_accepts_italian_query_and_english_heading():
    """Verify table lookup accepts italian query and english heading."""
    table = Document(
        page_content="| Colonna |",
        metadata={"source": "patient.md", "section": "Table 1 and Table 2", "type": "table"},
    )
    vectorstore = MagicMock()
    vectorstore.docstore._dict = {"table": table}
    compressor = MagicMock()
    compressor.compress_documents.return_value = [table]
    retriever = FilterableRerankerRetriever.model_construct(
        vs=vectorstore,
        k=4,
        compressor=compressor,
        search_type="similarity",
        fetch_k=8,
        lambda_mult=0.3,
    )

    result = retriever._get_relevant_documents("Mostra la tabella 2", filter={"source": "patient.md"})

    assert result == [table]
    compressor.compress_documents.assert_called_once_with([table], "Mostra la tabella 2")
    vectorstore.similarity_search.assert_not_called()


def test_reranker_scores_copies_without_mutating_shared_documents():
    """Verify reranker scores copies without mutating shared documents."""
    original = Document(page_content="contenuto", metadata={"source": "doc.md"})

    class FakeCrossEncoder(BaseCrossEncoder):
        def score(self, text_pairs):
            assert text_pairs == [("domanda", "contenuto")]
            return [0.75]

    model = FakeCrossEncoder()
    reranker = ScoredCrossEncoderReranker(model=model, top_n=1, score_key="rerank_score")

    result = reranker.compress_documents([original], "domanda")

    assert result[0] is not original
    assert result[0].metadata == {"source": "doc.md", "rerank_score": 0.75}
    assert original.metadata == {"source": "doc.md"}


def test_build_retriever_propagates_model_loading_policy(monkeypatch):
    """Verify build retriever propagates model loading policy."""
    cross_encoder = object()
    compressor = object()
    expected = object()
    cross_encoder_builder = MagicMock(return_value=cross_encoder)
    compressor_builder = MagicMock(return_value=compressor)
    retriever_builder = MagicMock(return_value=expected)
    monkeypatch.setattr(retriever_module, "MPSSentenceCrossEncoder", cross_encoder_builder)
    monkeypatch.setattr(retriever_module, "ScoredCrossEncoderReranker", compressor_builder)
    monkeypatch.setattr(retriever_module, "FilterableRerankerRetriever", retriever_builder)

    result = build_retriever(
        object(),
        k=5,
        rerank_model="rerank-model",
        k_reranked=2,
        online=False,
        cache_folder="/configured/cache",
    )

    assert result is expected
    cross_encoder_builder.assert_called_once_with(
        "rerank-model",
        online=False,
        cache_folder="/configured/cache",
    )
    compressor_builder.assert_called_once_with(model=cross_encoder, top_n=2, score_key="rerank_score")


@pytest.mark.parametrize(
    "search_type, method_name, static_kwargs",
    [
        ("similarity", "similarity_search", {"k": 5, "fetch_k": 5}),
        ("mmr", "max_marginal_relevance_search", {"k": 5, "fetch_k": 5, "lambda_mult": 0.25}),
    ],
)
def test_plain_retriever_forwards_each_request_filter_without_retaining_it(
    monkeypatch,
    search_type,
    method_name,
    static_kwargs,
):
    """Verify plain retrieval merges transient source filters into FAISS calls."""
    documents = [Document(page_content="contenuto", metadata={"source": "patient.md"})]
    vectorstore = MagicMock(spec=VectorStore)
    search = getattr(vectorstore, method_name)
    search.return_value = documents

    def make_retriever(**kwargs):
        return VectorStoreRetriever(vectorstore=vectorstore, **kwargs)

    vectorstore.as_retriever.side_effect = make_retriever
    cross_encoder = MagicMock()
    monkeypatch.setattr(retriever_module, "MPSSentenceCrossEncoder", cross_encoder)
    retriever = build_retriever(
        vectorstore,
        k=5,
        rerank_model=None,
        k_reranked=2,
        search_type=search_type,
        fetch_k=2,
        lambda_mult=0.25,
        online=False,
        cache_folder="/configured/cache",
    )
    source_filter = {"source": "patient.md"}

    assert retriever.invoke("solo pazienti", filter=source_filter) == documents
    assert retriever.invoke("tutto") == documents

    assert search.call_args_list == [
        call("solo pazienti", **static_kwargs, filter=source_filter),
        call("tutto", **static_kwargs),
    ]
    cross_encoder.assert_not_called()


def test_cross_encoder_score_returns_plain_floats_with_fixed_batch_size():
    """Verify cross-encoder predictions are converted to floats with the supported batch size."""
    encoder = MPSSentenceCrossEncoder.__new__(MPSSentenceCrossEncoder)
    encoder.model = MagicMock()
    encoder.model.predict.return_value = [1, 0.25]
    pairs = [("q", "one"), ("q", "two")]

    assert encoder.score(pairs) == [1.0, 0.25]
    encoder.model.predict.assert_called_once_with(pairs, batch_size=16)


def test_reranker_orders_truncates_and_rejects_score_count_mismatches():
    """Verify reranking sorts scores, honors top_n, and rejects incomplete scoring."""
    documents = [
        Document(page_content="low", metadata={}, id="low"),
        Document(page_content="high", metadata={}, id="high"),
        Document(page_content="middle", metadata={}, id="middle"),
    ]
    model = MagicMock(spec=BaseCrossEncoder)
    model.score.return_value = [0.1, 0.9, 0.5]
    reranker = ScoredCrossEncoderReranker(model=model, top_n=2)

    result = reranker.compress_documents(documents, "query")

    assert [document.id for document in result] == ["high", "middle"]
    assert [document.metadata["rerank_score"] for document in result] == [0.9, 0.5]

    model.score.return_value = [0.1]
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        reranker.compress_documents(documents, "query")


def test_reranking_retriever_builds_callable_for_negative_metadata_filter():
    """Verify dictionary-valued filters become FAISS-compatible metadata predicates."""
    document = Document(page_content="contenuto", metadata={"source": "scientific.md"})
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [document]
    compressor = MagicMock()
    compressor.compress_documents.return_value = [document]
    retriever = FilterableRerankerRetriever.model_construct(
        vs=vectorstore,
        k=4,
        compressor=compressor,
        search_type="similarity",
        fetch_k=8,
        lambda_mult=0.3,
    )

    result = retriever._get_relevant_documents(
        "domanda",
        filter={"source": {"$ne": "patient.md"}},
    )

    assert result == [document]
    metadata_filter = vectorstore.similarity_search.call_args.kwargs["filter"]
    assert callable(metadata_filter)
    assert metadata_filter({"source": "scientific.md"}) is True
    assert metadata_filter({"source": "patient.md"}) is False
    assert metadata_filter({}) is True

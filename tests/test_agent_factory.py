from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agentic_rag.agent_factory as agent_factory
from agentic_rag.config_schema import Config

from .helpers import make_valid_config

pytestmark = pytest.mark.cpu


def test_build_rag_agent_forwards_passed_config(monkeypatch):
    """Verify build rag agent forwards passed config."""
    data = make_valid_config()
    data.update(
        {
            "online": False,
            "debugger": True,
            "use_gpu_index": True,
            "search_type": "mmr",
            "fetch_k": 4,
            "lambda_mult": 0.65,
        }
    )
    cfg = Config.model_validate(data)
    vectorstore = object()
    retriever = object()
    llm = MagicMock()
    llm.bind.return_value = llm
    graph = object()
    cache_folder = str(Path(cfg.hf_home).resolve() / "hub")

    load_vectorstore = MagicMock(return_value=vectorstore)
    build_retriever = MagicMock(return_value=retriever)
    build_llm_pipe = MagicMock(return_value=llm)
    build_agent_graph = MagicMock(return_value=graph)
    monkeypatch.setattr(agent_factory, "load_vectorstore", load_vectorstore)
    monkeypatch.setattr(agent_factory, "build_retriever", build_retriever)
    monkeypatch.setattr(agent_factory, "build_llm_pipe", build_llm_pipe)
    monkeypatch.setattr(agent_factory, "build_agent_graph", build_agent_graph)
    monkeypatch.setattr(agent_factory, "SynonymStore", MagicMock())

    result = agent_factory.build_rag_agent(cfg)

    assert result is graph
    load_vectorstore.assert_called_once_with(
        cfg.index_dir,
        cfg.embed_model,
        online=False,
        use_gpu_index=True,
        cache_folder=cache_folder,
    )
    build_retriever.assert_called_once_with(
        vectorstore,
        cfg.k,
        cfg.rerank_model,
        cfg.k_reranked,
        score_key="rerank_score",
        search_type="mmr",
        fetch_k=4,
        lambda_mult=0.65,
        online=False,
        cache_folder=cache_folder,
    )
    assert build_llm_pipe.call_args.kwargs == {
        "quantization": cfg.quantization,
        "online": False,
        "debugger": True,
        "cache_folder": cache_folder,
    }
    build_agent_graph.assert_called_once()
    assert build_agent_graph.call_args.args[1] is cfg

    context = build_agent_graph.call_args.args[0]
    llm.invoke.return_value = {"content": "model output"}
    chain_inputs = [
        (context.topic_continuity_classifier, {"question": "q", "history": "h"}),
        (context.rag_chain, {"question": "q", "context": "c"}),
        (context.sanitizer_chain, {"question": "q"}),
        (context.cleaner_chain, {"answer": "a"}),
        (context.pre_retrieval_question_rewriter, {"question": "q", "history": "h"}),
        (context.question_transformer, {"question": "q", "matched_terms": {}}),
        (context.guardrail_chain, {"question": "q"}),
    ]
    assert [chain.invoke(values) for chain, values in chain_inputs] == ["model output"] * len(chain_inputs)

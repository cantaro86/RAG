# src/app_factory.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src._load_env import Config
from src.agent_prompts import prompt_rag, prompt_rewrite_medical, prompt_topic
from src.build_faiss import load_vectorstore
from src.graph import RAGContext, build_agent_graph
from src.llm_build import build_llm_pipe
from src.retriever import build_retriever


def build_rag_agent(cfg: Config):
    """
    Build and return the compiled agent graph.
    Single source of truth used by both CLI and Gradio.
    """
    vs = load_vectorstore(cfg.index_dir, cfg.embed_model)

    retriever_en = build_retriever(
        vs,
        cfg.k,
        cfg.rerank_model if cfg.rerank else None,
        cfg.k_reranked,
        score_key="rerank_score_en",
    )
    retriever_it = build_retriever(
        vs,
        cfg.k,
        cfg.rerank_model if cfg.rerank else None,
        cfg.k_reranked,
        score_key="rerank_score_it",
    )

    llm = build_llm_pipe(
        cfg.llm_model,
        cfg.max_new_tokens,
        cfg.temperature,
        cfg.top_p,
        cfg.top_k,
        cfg.repetition_penalty,
        cfg.no_repeat_ngram_size,
        quantization=cfg.quantization,
    )

    llm_runnable = RunnableLambda(lambda text: llm.invoke([{"role": "user", "content": str(text)}])["content"])

    topic_continuity_classifier = prompt_topic | llm_runnable | StrOutputParser()
    rag_chain = prompt_rag | llm_runnable | StrOutputParser()
    question_rewriter = prompt_rewrite_medical | llm_runnable | StrOutputParser()

    ctx = RAGContext(
        topic_continuity_classifier=topic_continuity_classifier,
        retriever_it=retriever_it,
        retriever_en=retriever_en,
        rag_chain=rag_chain,
        question_rewriter=question_rewriter,
    )

    return build_agent_graph(ctx)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src._load_env import Config
from src.agent_prompts import prompt_clean_chat, prompt_rag, prompt_rewrite_medical, prompt_topic
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

    retriever = build_retriever(
        vs,
        cfg.k,
        cfg.rerank_model if cfg.rerank else None,
        cfg.k_reranked,
        score_key="rerank_score",
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

    llm_cleaner = llm.bind(repetition_penalty=1.0, temperature=1e-5, no_repeat_ngram_size=0, do_sample=False)

    llm_runnable = RunnableLambda(lambda text: llm.invoke([{"role": "user", "content": str(text)}])["content"])
    llm_messages = RunnableLambda(lambda messages: llm_cleaner.invoke(messages)["content"])

    topic_continuity_classifier = prompt_topic | llm_runnable | StrOutputParser()
    rag_chain = prompt_rag | llm_runnable | StrOutputParser()
    cleaner_chain = prompt_clean_chat | llm_messages | StrOutputParser()
    question_rewriter = prompt_rewrite_medical | llm_runnable | StrOutputParser()

    ctx = RAGContext(
        topic_continuity_classifier=topic_continuity_classifier,
        retriever=retriever,
        rag_chain=rag_chain,
        cleaner_chain=cleaner_chain,
        question_rewriter=question_rewriter,
    )

    return build_agent_graph(ctx)

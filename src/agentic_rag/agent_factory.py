from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from agentic_rag.agent_prompts import (
    prompt_clean_chat,
    prompt_guardrail,
    prompt_rag,
    prompt_rewrite_medical,
    prompt_sanitizer,
    prompt_topic,
    prompt_transform_query,
)
from agentic_rag.build_faiss import load_vectorstore
from agentic_rag.config_schema import Config
from agentic_rag.dizionario import SynonymStore
from agentic_rag.graph import RAGContext, build_agent_graph
from agentic_rag.llm_build import build_llm_pipe
from agentic_rag.retriever import build_retriever


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

    llm_cleaner = llm.bind(
        repetition_penalty=1.0, temperature=1.0, no_repeat_ngram_size=0, top_p=1.0, top_k=50, do_sample=False
    )
    llm_sanitizer_model = llm_cleaner

    # with do_sample=False the temperature, top_p and top_k are ignored, but we set them to default values for clarity
    llm_topic_classifier = llm.bind(temperature=1.0, top_p=1.0, top_k=50, do_sample=False, max_new_tokens=5)
    llm_guardrail_classifier = llm.bind(temperature=1.0, top_p=1.0, top_k=50, do_sample=False, max_new_tokens=5)
    llm_rewriter_bound = llm.bind(temperature=0.1, top_p=0.95, top_k=50, do_sample=True)

    def invoke_prompt_value(prompt_value, model):
        messages = prompt_value if isinstance(prompt_value, list) else prompt_value.to_messages()
        return model.invoke(messages)["content"]

    llm_runnable = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm))
    llm_sanitizer = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm_sanitizer_model))
    llm_messages = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm_cleaner))
    llm_rewriter = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm_rewriter_bound))
    llm_topic = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm_topic_classifier))
    llm_guardrail = RunnableLambda(lambda prompt_value: invoke_prompt_value(prompt_value, llm_guardrail_classifier))

    topic_continuity_classifier = prompt_topic | llm_topic | StrOutputParser()
    rag_chain = prompt_rag | llm_runnable | StrOutputParser()
    sanitizer_chain = prompt_sanitizer | llm_sanitizer | StrOutputParser()
    cleaner_chain = prompt_clean_chat | llm_messages | StrOutputParser()
    pre_retrieval_question_rewriter = prompt_rewrite_medical | llm_rewriter | StrOutputParser()
    question_transformer = prompt_transform_query | llm_rewriter | StrOutputParser()
    guardrail_chain = prompt_guardrail | llm_guardrail | StrOutputParser()

    # Load the synonym store
    synonyms = SynonymStore(excel_path=cfg.dizionario_path)

    ctx = RAGContext(
        topic_continuity_classifier=topic_continuity_classifier,
        retriever=retriever,
        rag_chain=rag_chain,
        sanitizer_chain=sanitizer_chain,
        cleaner_chain=cleaner_chain,
        pre_retrieval_question_rewriter=pre_retrieval_question_rewriter,
        question_transformer=question_transformer,
        guardrail_chain=guardrail_chain,
        synonyms=synonyms,
    )

    return build_agent_graph(ctx)

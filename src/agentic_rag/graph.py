import logging
import re
from dataclasses import dataclass, fields
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agentic_rag.config_schema import Config
from agentic_rag.dizionario import SynonymStore
from agentic_rag.loggers import Logger
from agentic_rag.state import GraphState, HistoryMessage
from agentic_rag.utils import extract_source_filter, print_sources, render_context

logger = Logger.get_logger(__name__)


@dataclass
class RAGContext:
    topic_continuity_classifier: object
    retriever: object
    rag_chain: object
    sanitizer_chain: object
    cleaner_chain: object
    pre_retrieval_question_rewriter: object
    question_transformer: object
    social_intent_chain: object
    domain_guardrail_chain: object
    synonyms: SynonymStore

    def __post_init__(self):
        """Ensure all required dependencies are provided."""
        for f in fields(self):
            if getattr(self, f.name) is None:
                raise ValueError(f"Missing required dependency: '{f.name}' in RAGContext")


# ------------------------
# Memory
# ------------------------


def push_memory(
    state: GraphState,
    user_q: str,
    assistant_a: str,
    max_history_turns: int,
) -> list[HistoryMessage]:
    if max_history_turns == 0:
        return []

    msgs = list(state.get("history", []))
    msgs.extend(
        [
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_a},
        ]
    )

    msgs = msgs[-(max_history_turns * 2) :]

    return msgs


def format_history(history: list[HistoryMessage], max_history_turns: int) -> str:
    if not history or max_history_turns == 0:
        return "No prior conversation."
    blocks = []
    # newest last (human likes chronological; model doesn’t care)
    for m in history[-(max_history_turns * 2) :]:
        role = "Q" if m["role"] == "user" else "A"
        blocks.append(f"{role}: {m['content']}")
    return "\n".join(blocks)


# ------------------------
# Agent Nodes
# ------------------------


def sanitize_question(state: GraphState, sanitizer_chain):
    logger.debug("--- SANITIZE QUESTION ---")
    question = state["question"]
    source_filter = extract_source_filter(question)
    sanitized = sanitizer_chain.invoke({"question": question})
    source_filter = source_filter or extract_source_filter(sanitized)
    logger.debug(f"Sanitized question: {sanitized}")
    return {
        **state,
        "question": sanitized,
        "original_question": sanitized,
        "documents": [],
        "rewrite_count": 0,
        "first_question": False,
        "generation": None,
        "topic_status": None,
        "has_docs": False,
        "social_intent": None,
        "guardrail_status": None,
        "source_filter": source_filter,
        "history": state.get("history", []),
    }


def social_intent(state: GraphState, social_intent_chain):
    logger.debug("--- SOCIAL INTENT ---")
    question = state["question"]

    classification = social_intent_chain.invoke({"question": question})
    classification = str(classification).strip().upper()
    logger.debug(f"Social intent classification: {classification}")

    if classification not in ("SALUTO", "GRAZIE", "DOMANDA"):
        logger.warning(f"Unexpected social intent classification '{classification}', defaulting to DOMANDA")
        classification = "DOMANDA"

    return {
        **state,
        "social_intent": classification,
    }


def domain_guardrail(state: GraphState, domain_guardrail_chain):
    logger.debug("--- DOMAIN GUARDRAIL ---")
    question = state["question"]

    classification = domain_guardrail_chain.invoke({"question": question})
    classification = str(classification).strip().upper()
    logger.debug(f"Domain guardrail classification: {classification}")

    if classification not in ("OFF_TOPIC", "ON_TOPIC"):
        logger.warning(f"Unexpected domain classification '{classification}', defaulting to ON_TOPIC")
        classification = "ON_TOPIC"

    return {
        **state,
        "guardrail_status": classification,
    }


def handle_hello(state: GraphState) -> dict:
    logger.debug("--- HANDLE HELLO ---")

    reply = "Ciao, sono un agente IA. Rispondo alle tue domande sull'esame Colon-TC."

    return {
        **state,
        "generation": reply,
    }


def handle_thanks(state: GraphState) -> dict:
    logger.debug("--- HANDLE THANKS ---")

    reply = "Di nulla, sono qui per aiutarti. Rispondo alle tue domande sull'esame Colon-TC."

    return {
        **state,
        "generation": reply,
    }


def handle_off_topic(state: GraphState) -> dict:
    logger.debug("--- HANDLE OFF TOPIC ---")

    reply = "Spiacente, non posso aiutarti. Rispondo alle tue domande sull'esame Colon-TC."

    return {
        **state,
        "generation": reply,
    }


def init_first_question(state: GraphState) -> dict:
    hist = state.get("history", [])
    first = len(hist) == 0
    logger.debug(f"--- INIT FIRST QUESTION ---\nFirst question: {first}")
    return {
        **state,
        "first_question": first,
    }


def topic_detector(state, topic_continuity_classifier, max_history_turns: int):
    """
    Decide whether the new question belongs to the same topic as the recent conversation.
    """

    question = state["question"]
    hist = format_history(state.get("history", []), max_history_turns)

    topic = topic_continuity_classifier.invoke({"question": question, "history": hist})
    topic = str(topic).strip().upper()

    logger.debug("--- TOPIC CONTINUITY CLASSIFIER---")
    logger.debug(f"Topic continuity evaluation: {topic}")

    # topic is either "STESSO" or "NUOVO"
    return {**state, "topic_status": topic}


def pre_retrieval_rewriter(state, pre_retrieval_question_rewriter, max_history_turns: int):
    """
    Resolve ambiguous references in the question before retrieval.
    Runs on followup turns only, using history to make the question self-contained.
    """
    logger.debug("---PRE-RETRIEVAL REWRITER---")
    question = state["question"]
    hist = format_history(state.get("history", []), max_history_turns)

    rewritten = pre_retrieval_question_rewriter.invoke({"question": question, "history": hist})
    logger.debug(f"Pre-retrieval rewritten question: {rewritten}")

    return {**state, "question": rewritten}


def retrieve_and_filter(state, retriever, threshold: float, rerank: bool):
    logger.debug("---RETRIEVE + FILTER---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    source_filter = state.get("source_filter")
    if rewrite_count == 0 and source_filter is None:
        source_filter = extract_source_filter(question)
    logger.debug(f"Source filter: {source_filter}")

    docs = retriever.invoke(question, filter=source_filter)

    if rerank:
        relevant_docs = [
            d for d in docs if "rerank_score" in d.metadata and float(d.metadata["rerank_score"]) > threshold
        ]
    else:
        relevant_docs = list(docs)

    # score_prob = []
    # Debug info
    if logger.level <= logging.DEBUG:
        ### Uncomment if you use a model with scores that need to be converted to probabilities
        # for d in relevant_docs:
        #     _rerank_score = d.metadata["rerank_score"]
        #     score_prob.append(
        #         [round(_rerank_score, 2), float(round(expit(_rerank_score), 2))]
        #     )
        sources_table = print_sources(relevant_docs)
        logger.debug(f"Retrieved {len(docs)} docs, {len(relevant_docs)} accepted (threshold={threshold})")
        # logger.debug(f"Scores and probabilities of all retrieved docs: {score_prob}")
        logger.debug(f"Top sources:\n{sources_table}")

    if relevant_docs:
        logger.info(f"✅ Found {len(relevant_docs)} relevant docs (threshold={threshold})")
        return {
            **state,
            "documents": relevant_docs,
            "rewrite_count": 0,
            "has_docs": True,
            "source_filter": source_filter,
        }
    else:
        logger.info(f"⚠️ No relevant docs found (attempt {rewrite_count + 1})")
        return {
            **state,
            "documents": [],
            "rewrite_count": rewrite_count + 1,
            "has_docs": False,
            "source_filter": source_filter,
        }


def generate_with_docs(state, rag_chain):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    logger.debug("---GENERATE WITH DOCS---")
    q = state["question"]
    docs = state.get("documents", [])
    rendered_docs = render_context(docs)

    answer = rag_chain.invoke({"context": rendered_docs, "question": q})
    logger.info(f"Raw generated answer: {answer}")

    return {**state, "generation": answer, "rewrite_count": 0}


_META_PATTERNS = re.compile(
    # English patterns (keep: LLM may still leak English)
    r"based on (the )?(provided |retrieved )?(sources?|context|documents?)|"
    r"according to (the )?sources?|"
    r"as (mentioned|stated) in (the )?(conversation history|retrieved documents?)|"
    r"the sources? indicate|"
    # Italian — "basato su" family
    r"in base ai (documenti|contesto|fonti)( forniti| recuperati)?|"
    r"sulla base dei (documenti|contesto|fonti)( forniti| recuperati)?|"
    r"basandomi sui? (documenti|contesto|fonti)( forniti| recuperati)?|"
    r"basandosi sui? (documenti|contesto|fonti)( forniti| recuperati)?|"
    # Italian — "secondo" family
    r"secondo (le )?(fonti|i documenti|il contesto)( forniti| recuperati)?|"
    r"stando (alle )?(fonti|ai documenti)( forniti| recuperati)?|"
    # Italian — "come indicato / riportato" family
    r"come (indicato|riportato|menzionato|descritto) "
    r"(nelle? |dai? )?(fonti?|documenti?|contesto)( forniti| recuperati)?|"
    r"come (emerge|risulta) dai (documenti|fonti|testi)( forniti| recuperati)?|"
    r"stando a quanto (indicato|riportato) (nei|dai|nelle) (documenti|fonti)|"
    # Italian — "le fonti indicano / i documenti mostrano"
    r"(le fonti|i documenti) (indicano|mostrano|riportano|suggeriscono|affermano)|"
    r"dal (contesto|materiale) (fornito|recuperato|disponibile)|"
    r"dai (documenti|testi|materiali) (forniti|recuperati|disponibili)|"
    # Citation markers — [1], [doc1], [fonte 2], [sorgente3], (Doc 2)
    r"\[(doc|fonte|source|sorgente)?\s?\d+\]|"
    r"\((doc|fonte|source|sorgente)\s?\d+\)",
    flags=re.IGNORECASE,
)


def clean_answer(
    state: GraphState,
    cleaner_chain,
    clean_answer_enabled: bool,
    max_history_turns: int,
):
    logger.debug("--- CLEAN ANSWER ---")

    raw_answer = state["generation"]
    final_answer = raw_answer

    if clean_answer_enabled:
        if not _META_PATTERNS.search(raw_answer):
            logger.debug("Clean answer skipped (no meta-commentary detected).")
        else:
            cleaned = cleaner_chain.invoke({"answer": raw_answer})
            final_answer = cleaned.strip()
            logger.info(f"Cleaned generated answer: {cleaned}")

    history = push_memory(
        state,
        state.get("original_question", state["question"]),
        final_answer,
        max_history_turns,
    )
    return {**state, "generation": final_answer, "history": history}


def transform_query(state, question_transformer, synonyms):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.debug("---TRANSFORM QUERY---")
    question = state["question"]

    matched_terms = synonyms.find_matched_terms(question)

    # Re-write question
    better_question = question_transformer.invoke({"question": question, "matched_terms": matched_terms})
    logger.debug(f"⚠️ Counter {state['rewrite_count']}. Transformed question: {better_question}")
    return {**state, "question": better_question}


def clear_history(state: GraphState) -> dict:
    # If topic is NEW, wipe history before retrieving
    logger.debug("---CLEAR HISTORY---")
    logger.debug("Clearing conversation history due to new topic.")
    return {
        **state,
        "history": [],
        "first_question": True,
        "rewrite_count": 0,
        "generation": None,
    }


def no_generation(state: GraphState) -> dict:
    logger.debug("--- NO GENERATION NO DOCS---")
    msg = (
        "Non sono riuscito a trovare informazioni pertinenti. Per favore riformula la tua domanda o aggiungi dettagli."
    )
    return {**state, "generation": msg, "has_docs": False}


# ------------------------
# Agent Edges
# ------------------------


def route_first_question(state: GraphState) -> str:
    return "first" if state.get("first_question", False) else "followup"


def decide_relevance(state):
    """
    Decide whether to generate or transform the question.
    """
    has_docs = state.get("has_docs", False)
    rewrite_count = state.get("rewrite_count", 0)

    if has_docs:
        return "generate_with_docs"
    elif has_docs is False and rewrite_count >= 2:
        logger.debug("🚫 Max rewrites reached.")
        return "end"
    else:
        return "transform_query"


def route_on_topic(state: GraphState) -> str:
    """
    Robust topic router with full state logging.
    """

    t = str(state.get("topic_status", "")).strip().upper()
    logger.debug(f"🚦 Extracted topic_status: '{t}' from state")

    if t in ("SAME", "SAME_TOPIC", "STESSO"):
        logger.debug("🚦 → ROUTING 'same'")
        return "same"
    if t in ("NEW", "NEW_TOPIC", "NUOVO"):
        logger.debug("🚦 → ROUTING 'new'")
        return "new"

    # Log failure case
    logger.warning(f"🚦 UNKNOWN topic_status '{t}' → defaulting to 'same'")
    return "same"


def route_social_intent(state: GraphState) -> str:
    status = state.get("social_intent", "DOMANDA")
    if status == "SALUTO":
        return "hello"
    if status == "GRAZIE":
        return "thanks"
    return "content"


def route_domain_guardrail(state: GraphState) -> str:
    if state.get("guardrail_status", "ON_TOPIC") == "OFF_TOPIC":
        return "off_topic"
    if str(state.get("topic_status", "")).strip().upper() in ("NEW", "NEW_TOPIC", "NUOVO"):
        return "new_topic"
    return "on_topic"


# ------------------------
# Build Agent Graph
# ------------------------
def build_agent_graph(ctx: RAGContext, cfg: Config):
    """Build the LangGraph agent"""

    checkpointer = InMemorySaver()

    workflow = StateGraph(GraphState)

    workflow.add_node("sanitize_question", partial(sanitize_question, sanitizer_chain=ctx.sanitizer_chain))

    workflow.add_node("social_intent", partial(social_intent, social_intent_chain=ctx.social_intent_chain))

    workflow.add_node(
        "domain_guardrail",
        partial(domain_guardrail, domain_guardrail_chain=ctx.domain_guardrail_chain),
    )

    workflow.add_node("handle_hello", handle_hello)

    workflow.add_node("handle_thanks", handle_thanks)

    workflow.add_node("handle_off_topic", handle_off_topic)

    workflow.add_node("init_first_question", init_first_question)

    workflow.add_node(
        "retrieve_and_filter",
        partial(
            retrieve_and_filter,
            retriever=ctx.retriever,
            threshold=cfg.threshold,
            rerank=cfg.rerank,
        ),
    )

    workflow.add_node("generate_with_docs", partial(generate_with_docs, rag_chain=ctx.rag_chain))

    workflow.add_node(
        "clean_answer",
        partial(
            clean_answer,
            cleaner_chain=ctx.cleaner_chain,
            clean_answer_enabled=cfg.clean_answer,
            max_history_turns=cfg.max_history_turns,
        ),
    )

    workflow.add_node(
        "transform_query",
        partial(
            transform_query,
            question_transformer=ctx.question_transformer,
            synonyms=ctx.synonyms,
        ),
    )

    workflow.add_node(
        "topic_detector",
        partial(
            topic_detector,
            topic_continuity_classifier=ctx.topic_continuity_classifier,
            max_history_turns=cfg.max_history_turns,
        ),
    )

    workflow.add_node(
        "pre_retrieval_rewriter",
        partial(
            pre_retrieval_rewriter,
            pre_retrieval_question_rewriter=ctx.pre_retrieval_question_rewriter,
            max_history_turns=cfg.max_history_turns,
        ),
    )

    workflow.add_node("clear_history", clear_history)

    workflow.add_node("no_generation", no_generation)

    # EDGES

    workflow.add_edge(START, "sanitize_question")

    workflow.add_edge("sanitize_question", "social_intent")

    workflow.add_conditional_edges(
        "social_intent",
        route_social_intent,
        {
            "hello": "handle_hello",
            "thanks": "handle_thanks",
            "content": "init_first_question",
        },
    )

    workflow.add_edge("handle_hello", END)

    workflow.add_edge("handle_thanks", END)

    workflow.add_edge("handle_off_topic", END)

    workflow.add_conditional_edges(
        "init_first_question",
        route_first_question,
        {
            "first": "domain_guardrail",
            "followup": "topic_detector",
        },
    )

    workflow.add_conditional_edges(
        "topic_detector",
        route_on_topic,
        {
            "same": "pre_retrieval_rewriter",
            "new": "domain_guardrail",
        },
    )

    workflow.add_edge("pre_retrieval_rewriter", "domain_guardrail")

    workflow.add_conditional_edges(
        "domain_guardrail",
        route_domain_guardrail,
        {
            "off_topic": "handle_off_topic",
            "new_topic": "clear_history",
            "on_topic": "retrieve_and_filter",
        },
    )

    workflow.add_edge("clear_history", "retrieve_and_filter")

    workflow.add_conditional_edges(
        "retrieve_and_filter",
        decide_relevance,
        {
            "transform_query": "transform_query",
            "generate_with_docs": "generate_with_docs",
            "end": "no_generation",
        },
    )

    workflow.add_edge("transform_query", "retrieve_and_filter")

    workflow.add_edge("generate_with_docs", "clean_answer")

    workflow.add_edge("clean_answer", END)

    workflow.add_edge("no_generation", END)

    agent = workflow.compile(checkpointer=checkpointer)
    return agent

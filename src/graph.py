import logging
import re
from dataclasses import dataclass, fields

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src._load_env import cfg
from src.bilingual_question import BilingualQuestion
from src.loggers import Logger
from src.state import GraphState
from src.utils import print_sources, render_context

logger = Logger.get_logger(__name__)


@dataclass
class RAGContext:
    topic_continuity_classifier: object
    retriever: object
    rag_chain: object
    cleaner_chain: object
    question_rewriter: object

    def __post_init__(self):
        """Ensure all required dependencies are provided."""
        for f in fields(self):
            if getattr(self, f.name) is None:
                raise ValueError(f"Missing required dependency: '{f.name}' in RAGContext")


# ------------------------
# Memory
# ------------------------


def push_memory(state, user_q, assistant_a):
    msgs = state.get("history", [])

    msgs.append({"role": "user", "content": user_q})
    msgs.append({"role": "assistant", "content": assistant_a})

    msgs = msgs[-(cfg.max_history_turns * 2) :]

    return msgs


def format_history(history: list[dict]):
    if not history:
        return "No prior conversation."
    blocks = []
    # newest last (human likes chronological; model doesn’t care)
    for m in history[-(cfg.max_history_turns * 2) :]:  # 6 turns == 12 entries
        role = "Q" if m["role"] == "user" else "A"
        blocks.append(f"{role}: {m['content']}")
    return "\n".join(blocks)


# ------------------------
# Agent Nodes
# ------------------------


def init_first_question(state: GraphState) -> dict:
    hist = state.get("history", [])
    first = len(hist) == 0
    return {
        **state,
        "first_question": first,
        "rewrite_count": 0,
        "has_docs": False,
        "documents": [],
        "generation": None,
    }


def topic_detector(state, topic_continuity_classifier):
    """
    Decide whether the new question belongs to the same topic as the recent conversation.
    """

    question = state["question"]
    hist = format_history(state.get("history", []))

    topic = topic_continuity_classifier.invoke({"question": question, "history": hist})
    topic = str(topic).strip().upper()

    logger.debug("--- TOPIC CONTINUITY CLASSIFIER---")
    logger.debug(f"Topic continuity evaluation: {topic}")

    # topic is either "SAME" or "NEW"
    return {**state, "topic_status": topic}


def retrieve_and_filter(state, retriever):
    logger.debug("---RETRIEVE + FILTER---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    # Retrieve from both languages if needed
    q = BilingualQuestion(question)
    docs_en = retriever.invoke(q.en)

    relevant_docs = [d for d in docs_en if float(d.metadata.get("rerank_score", 0)) > cfg.threshold]

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

    logger.debug(f"Retrieved {len(docs_en)} docs, {len(relevant_docs)} above threshold {cfg.threshold}")
    # logger.debug(f"Scores and probabilities of all retrieved docs: {score_prob}")
    logger.debug(f"Top sources:\n{sources_table}")

    if relevant_docs:
        logger.info(f"✅ Found {len(relevant_docs)} relevant docs (threshold={cfg.threshold})")
        return {
            **state,
            "documents": relevant_docs,
            "rewrite_count": 0,
            "has_docs": True,
        }
    else:
        logger.info(f"⚠️ No relevant docs found (attempt {rewrite_count + 1})")
        return {
            **state,
            "documents": [],
            "rewrite_count": rewrite_count + 1,
            "has_docs": False,
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
    hist = format_history(state.get("history", []))

    answer = rag_chain.invoke({"history": hist, "context": rendered_docs, "question": q})
    logger.info(f"Raw Generated answer (EN): {answer}")

    msgs = push_memory(state, q, answer)

    return {**state, "generation": answer, "rewrite_count": 0, "history": msgs}


_META_PATTERNS = re.compile(
    r"based on (the )?(provided |retrieved )?(sources?|context|documents?)|"
    r"according to (the )?sources?|"
    r"as (mentioned|stated) in (the )?(conversation history|retrieved documents?)|"
    r"the sources? indicate|"
    r"\[(doc|source)?\s?\d+\]",
    flags=re.IGNORECASE,
)


def clean_answer(state, cleaner_chain):
    logger.debug("--- CLEAN ANSWER ---")

    if cfg.clean_answer is True:
        raw_answer = state["generation"]

        if not _META_PATTERNS.search(raw_answer):
            logger.debug("Clean answer skipped (no meta-commentary detected).")
            return state

        cleaned = cleaner_chain.invoke({"answer": raw_answer})

        logger.info(f"Cleaned Generated answer (EN): {cleaned}")

        return {
            **state,
            "generation": cleaned.strip(),
        }

    return state


def transform_query(state, question_rewriter):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.debug("---TRANSFORM QUERY---")
    question = state["question"]
    hist = format_history(state.get("history", []))

    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "history": hist})
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


def generate_no_docs(state: GraphState) -> dict:
    logger.debug("---GENERATE NO DOCS---")
    msg = "I couldn't find relevant information. Please rephrase your question or add details."
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

    if t in ("SAME", "SAME_TOPIC", "SAMETOPIC"):
        logger.debug("🚦 → ROUTING 'same'")
        return "same"
    if t in ("NEW", "NEW_TOPIC", "NEWTOPIC"):
        logger.debug("🚦 → ROUTING 'new'")
        return "new"

    # Log failure case
    logger.warning(f"🚦 UNKNOWN topic_status '{t}' → defaulting to 'same'")
    return "same"


# ------------------------
# Build Agent Graph
# ------------------------
def build_agent_graph(ctx: RAGContext):
    """Build the LangGraph agent"""

    checkpointer = InMemorySaver()

    workflow = StateGraph(GraphState)

    workflow.add_node("init_first_question", init_first_question)

    workflow.add_node("retrieve_and_filter", lambda state: retrieve_and_filter(state, ctx.retriever))

    workflow.add_node("generate_with_docs", lambda state: generate_with_docs(state, ctx.rag_chain))

    workflow.add_node("clean_answer", lambda state: clean_answer(state, ctx.cleaner_chain))

    workflow.add_node("transform_query", lambda state: transform_query(state, ctx.question_rewriter))

    workflow.add_node("topic_detector", lambda state: topic_detector(state, ctx.topic_continuity_classifier))

    workflow.add_node("clear_history", clear_history)

    workflow.add_node("generate_no_docs", generate_no_docs)

    workflow.add_edge(START, "init_first_question")

    workflow.add_edge("clear_history", "retrieve_and_filter")

    workflow.add_edge("generate_with_docs", "clean_answer")

    workflow.add_conditional_edges(
        "init_first_question",
        route_first_question,
        {
            "first": "retrieve_and_filter",
            "followup": "topic_detector",
        },
    )

    workflow.add_conditional_edges(
        "topic_detector",
        route_on_topic,
        {
            "same": "retrieve_and_filter",
            "new": "clear_history",
        },
    )

    workflow.add_conditional_edges(
        "retrieve_and_filter",
        decide_relevance,
        {
            "transform_query": "transform_query",
            "generate_with_docs": "generate_with_docs",
            "end": "generate_no_docs",
        },
    )

    workflow.add_edge("generate_no_docs", END)

    workflow.add_edge("transform_query", "retrieve_and_filter")

    workflow.add_edge("clean_answer", END)

    agent = workflow.compile(checkpointer=checkpointer)
    return agent

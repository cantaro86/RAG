import logging
from dataclasses import dataclass, fields

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ._load_env import cfg
from .bilingual_question import BilingualQuestion
from .loggers import Logger
from .state import GraphState

logger = Logger.get_logger(__name__)


@dataclass
class RAGContext:
    answer_validation: object
    retriever: object
    rag_chain: object
    chain_general: object
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
    msgs = state.get("messages", [])

    msgs.append({"role": "user", "content": user_q})
    msgs.append({"role": "assistant", "content": assistant_a})

    msgs = msgs[-(cfg.max_history_turns * 2) :]

    return msgs


def format_history(messages: list[dict]):
    if not messages:
        return "No prior conversation."
    blocks = []
    # newest last (human likes chronological; model doesn’t care)
    for m in messages[-(cfg.max_history_turns * 2) :]:  # 6 turns == 12 entries
        role = "Q" if m["role"] == "user" else "A"
        blocks.append(f"{role}: {m['content']}")
    return "\n".join(blocks)


# ------------------------
# Agent Nodes
# ------------------------


def medical_router(state, answer_validation):
    """Decide whether the question is medical or general.
    We use continuity classification to classify follow-up questions.
    """

    logger.info("--- VALIDATE MEDICAL QUESTION ---")
    q = state["question"]
    prev_domain = state.get("last_domain", "general")

    label = answer_validation.invoke({"question": q})["score"].strip().lower()

    if label not in ("medical", "general"):
        logger.debug("Not in medical or general")
        label = "general"

    # *** continuity condition ***
    if label == "general" and prev_domain == "medical":
        logger.debug("Continuity condition triggered")
        label = "medical"

    logger.info(f"Medical evaluation: {label}")

    return {**state, "domain": label, "last_domain": label}


def retrieve_and_filter(state, retriever):
    logger.info("---RETRIEVE + FILTER---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    # Retrieve from both languages if needed
    q = BilingualQuestion(question)
    docs_it = retriever.invoke(q.it)
    docs_en = retriever.invoke(q.en)
    all_docs = docs_it + docs_en

    relevant_docs = [d for d in all_docs if d.metadata.get("rerank_score", 0) > cfg.threshold]

    # Debug info
    if logger.level <= logging.DEBUG:
        from scipy.special import expit

        score_prob = []
        for d in all_docs:
            score_prob.append([d.metadata["rerank_score"], expit(d.metadata["rerank_score"])])
    logger.debug(f"Retrieved {len(all_docs)} docs, {len(relevant_docs)} above threshold {cfg.threshold}")
    logger.debug(f"Scores and probabilities of all retrieved docs: {score_prob}")

    if relevant_docs:
        logger.info(f"✅ Found {len(relevant_docs)} relevant docs (threshold={cfg.threshold})")
        return {
            "documents": relevant_docs,
            "question": question,
            "rewrite_count": rewrite_count,
            "has_docs": True,
        }
    else:
        logger.info(f"⚠️ No relevant docs found (attempt {rewrite_count + 1})")
        return {
            "documents": [],
            "question": question,
            "rewrite_count": rewrite_count + 1,
            "has_docs": False,
        }


def generate(state, chain_general):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    logger.info("---GENERATE---")
    q = state["question"]
    hist = format_history(state.get("messages", []))

    answer = chain_general.invoke({"history": hist, "question": q})

    msgs = push_memory(state, q, answer)

    return {
        **state,
        "generation": answer,
        "rewrite_count": 0,
        "messages": msgs,
        "last_domain": "general",
    }  # "last_domain" here breaks continuity condition


def generate_with_docs(state, rag_chain):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    logger.info("---GENERATE WITH DOCS---")
    q = state["question"]
    docs = state.get("documents", [])
    hist = format_history(state.get("messages", []))

    answer = rag_chain.invoke({"history": hist, "context": docs, "question": q})

    msgs = push_memory(state, q, answer)

    return {**state, "generation": answer, "rewrite_count": 0, "messages": msgs}


def transform_query(state, question_rewriter):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    hist = format_history(state.get("messages", []))

    # Re-write question
    better_question = question_rewriter.invoke({"question": question, "history": hist})
    logger.debug(f"Counter {state['rewrite_count']}. Transformed question: {better_question}")
    return {**state, "question": better_question}


# ------------------------
# Agent Edges
# ------------------------


def decide_relevance(state):
    """
    Decide whether to generate or transform the question.
    """
    has_docs = state.get("has_docs", False)
    rewrite_count = state.get("rewrite_count", 0)

    if has_docs:
        return "generate_with_docs"
    elif has_docs is False and rewrite_count >= 2:
        logger.info("🚫 Max rewrites reached — fallback to generation without context.")
        return "generate"
    else:
        return "transform_query"


# ------------------------
# Build Agent Graph
# ------------------------
def build_agent_graph(ctx: RAGContext):
    """Build the LangGraph agent"""

    checkpointer = InMemorySaver()

    workflow = StateGraph(GraphState)

    workflow.add_node("medical_router", lambda state: medical_router(state, ctx.answer_validation))

    workflow.add_node("retrieve_and_filter", lambda state: retrieve_and_filter(state, ctx.retriever))

    workflow.add_node("generate", lambda state: generate(state, ctx.chain_general))

    workflow.add_node("generate_with_docs", lambda state: generate_with_docs(state, ctx.rag_chain))

    workflow.add_node("transform_query", lambda state: transform_query(state, ctx.question_rewriter))

    workflow.add_edge(START, "medical_router")

    workflow.add_conditional_edges(
        "medical_router",
        lambda state: state["domain"],  # this must return "medical" or "general"
        {
            "medical": "retrieve_and_filter",
            "general": "generate",
        },
    )

    workflow.add_conditional_edges(
        "retrieve_and_filter",
        decide_relevance,
        {"transform_query": "transform_query", "generate": "generate", "generate_with_docs": "generate_with_docs"},
    )

    workflow.add_edge("transform_query", "retrieve_and_filter")

    workflow.add_edge("generate", END)
    workflow.add_edge("generate_with_docs", END)

    agent = workflow.compile(checkpointer=checkpointer)
    return agent

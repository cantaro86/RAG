import re
from dataclasses import dataclass, fields

from langgraph.graph import END, START, StateGraph

from .bilingual_question import BilingualQuestion
from .state import GraphState


@dataclass
class RAGContext:
    retriever: object
    rag_chain: object
    hallucination_grader: object
    answer_grader: object
    question_rewriter: object

    def __post_init__(self):
        """Ensure all required dependencies are provided."""
        for f in fields(self):
            if getattr(self, f.name) is None:
                raise ValueError(f"Missing required dependency: '{f.name}' in RAGContext")


# ------------------------
# Agent Nodes
# ------------------------


def retrieve_and_filter(state, retriever):
    print("---RETRIEVE + FILTER---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    # Retrieve from both languages if needed
    q = BilingualQuestion(question)
    docs_it = retriever.invoke(q.it)
    docs_en = retriever.invoke(q.en)
    all_docs = docs_it + docs_en

    threshold = 0.0
    relevant_docs = [d for d in all_docs if d.metadata.get("rerank_score", 0) > threshold]

    if relevant_docs:
        print(f"✅ Found {len(relevant_docs)} relevant docs (threshold={threshold})")
        return {
            "documents": relevant_docs,
            "question": question,
            "rewrite_count": rewrite_count,
            "has_docs": True,
        }
    else:
        print(f"⚠️ No relevant docs found (attempt {rewrite_count + 1})")
        return {
            "documents": [],
            "question": question,
            "rewrite_count": rewrite_count + 1,
            "has_docs": False,
        }


def generate(state, rag_chain):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    rewrite_count = state.get("rewrite_count", 0)

    if not documents and rewrite_count >= 3:
        print(f"⚠️ No documents found after {rewrite_count} rewrites — generating without context.")
        # context = "No relevant context found. Use general medical knowledge to answer."
        context = (
            "No relevant context found in the knowledge base. "
            "Answer briefly and factually based on general knowledge, "
            "in few sentences."
        )
        generation = rag_chain.invoke({"context": context, "question": question})
    else:
        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "rewrite_count": rewrite_count}


def transform_query(state, question_rewriter):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def validate_question(state):
    """
    Check if the user input is a valid question.
    If not, short-circuit and return a polite response.
    """
    text = state["question"].strip().lower()

    # Simple keyword and structure check
    question_like = bool(re.search(r"\b(what|why|how|when|where|who|which|is|are|can|does|do)\b", text))
    is_question_mark = text.endswith("?")

    if question_like or is_question_mark:
        # Proceed normally
        return {"valid_question": True}
    else:
        print("🚫 Input is not a question. Stopping early.")
        # Optional: short, friendly reply
        return {
            "valid_question": False,
            "generation": "Hello! How can I help you with a question about medical reports or general knowledge?",
        }


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
        return "generate"
    elif rewrite_count >= 3:
        print("🚫 Max rewrites reached — fallback to generation without context.")
        return "generate"
    else:
        return "transform_query"


def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def next_step(state):
    if state["has_docs"]:
        return "generate"
    elif state["rewrite_count"] >= 3:
        print("🚫 Max rewrites reached — fallback to generation without context.")
        return "generate"
    else:
        return "transform_query"


# ------------------------
# Build Agent Graph
# ------------------------
def build_agent_graph(ctx: RAGContext):
    """Build the LangGraph agent"""

    workflow = StateGraph(GraphState)

    workflow.add_node("validate_question", validate_question)

    workflow.add_node("retrieve_and_filter", lambda state: retrieve_and_filter(state, ctx.retriever))

    workflow.add_node("generate", lambda state: generate(state, ctx.rag_chain))

    workflow.add_node("transform_query", lambda state: transform_query(state, ctx.question_rewriter))

    workflow.add_edge(START, "validate_question")

    workflow.add_conditional_edges(
        "validate_question",
        lambda state: "retrieve_and_filter" if state.get("valid_question", False) else END,
        {"retrieve_and_filter": "retrieve_and_filter", END: END},
    )

    workflow.add_conditional_edges(
        "retrieve_and_filter",
        decide_relevance,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve_and_filter")

    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation_v_documents_and_question(state, ctx.hallucination_grader, ctx.answer_grader),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    agent = workflow.compile()
    return agent

from typing import Literal, Required, TypedDict

from langchain_core.documents import Document


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: Required[str]
    original_question: str
    documents: list[Document]
    rewrite_count: int
    history: list[HistoryMessage]
    first_question: bool
    generation: str | None
    source_filter: dict[str, str] | None
    topic_status: Literal["SAME", "NEW", "SAME_TOPIC", "NEW_TOPIC", "STESSO", "NUOVO"] | None
    has_docs: bool
    social_intent: Literal["SALUTO", "GRAZIE", "DOMANDA"] | None
    guardrail_status: Literal["OFF_TOPIC", "ON_TOPIC"] | None

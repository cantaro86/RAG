from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    documents: list[Document]
    rewrite_count: int
    history: list[dict]
    first_question: bool
    generation: NotRequired[str]
    topic_status: NotRequired[Literal["SAME", "NEW", "SAME_TOPIC", "NEW_TOPIC"]]
    has_docs: NotRequired[bool]

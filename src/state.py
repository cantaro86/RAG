from typing import TypedDict

from langchain.schema import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: list[Document]
    rewrite_count: int
    messages: list[dict]
    last_domain: str

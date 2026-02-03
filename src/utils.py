import os

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table as RichTable

from .loggers import Logger

logger = Logger.get_logger(__name__)


PATIENT_DOC = "3.6.25.pdf"


def print_sources(docs: list[Document]) -> str:
    """
    Create a formatted table of source documents and return it as plain text.
    """
    table = RichTable(title="Top Context Chunks")
    table.add_column("#")
    table.add_column("Source")
    table.add_column("Page")
    table.add_column("Chars")
    table.add_column("Rank")
    table.add_column("OCR")

    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page = str(d.metadata.get("page", "?"))
        rank = d.metadata.get("rerank_score", -99)
        ocr = str(d.metadata.get("ocr_used", "NA"))
        table.add_row(str(i), src, page, str(len(d.page_content)), str(rank), ocr)

    # Use a plain console to capture plain text output
    plain_console = Console(width=120, color_system=None, file=None)
    with plain_console.capture() as capture:
        plain_console.print(table)
    return capture.get()


def doc_corpus_label(d: Document) -> str:
    src = os.path.basename(d.metadata.get("source", ""))
    return "Information for patients" if src == PATIENT_DOC else "Colonoscopy literature"


def render_context(docs: list[Document]) -> str:
    # Put patient chunks first so the model sees them earlier (it helps in practice).
    docs_sorted = sorted(
        docs,
        key=lambda d: (
            doc_corpus_label(d) != "Information for patients",  # False first
            d.metadata.get("section") != "Recommendation",
        ),  # False first
    )
    parts = []
    for i, d in enumerate(docs_sorted, 1):
        parts.append(
            f"[{i}] Corpus: {doc_corpus_label(d)}\n"
            f"Section: {d.metadata.get('section', 'main')}\n"
            f"Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, page {d.metadata.get('page', '?')}\n"
            f"Excerpt: {d.page_content.strip()}\n"
        )
    return "\n".join(parts)

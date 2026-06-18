import os
import re
import unicodedata

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table as RichTable

from agentic_rag.loggers import Logger

logger = Logger.get_logger(__name__)


PATIENT_DOC = "Informazioni_per_pazienti.md"

_CORPUS_FILTER_PATTERNS = [
    (
        re.compile(
            r"information[\s_]*for[\s_]*patients?|"
            r"patient[\s_]*information|"
            r"informazioni[\s_]*per[\s_]*pazienti|"
            r"informazioni_per_pazienti|"
            r"informativa[\s_]*per[\s_]*pazienti|"
            r"informativa_per_pazienti",
            re.IGNORECASE,
        ),
        {"source": PATIENT_DOC},
    ),
]


def extract_source_filter(question: str) -> dict | None:
    for pattern, filt in _CORPUS_FILTER_PATTERNS:
        if pattern.search(question):
            return filt
    return None


def print_sources(docs: list[Document]) -> str:
    """
    Create a formatted table of source documents and return it as plain text.
    """
    table = RichTable(title="Top Context Chunks")
    table.add_column("#")
    table.add_column("Source")
    table.add_column("Section")
    table.add_column("Type")
    table.add_column("Language")
    table.add_column("Chars")
    table.add_column("Rank")

    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        section = d.metadata.get("section", "?")
        chunk_type = d.metadata.get("type", "?")
        language = d.metadata.get("language", "?")
        rank = d.metadata.get("rerank_score", -99)

        table.add_row(
            str(i),
            src,
            section,
            chunk_type,
            language,
            str(len(d.page_content)),
            str(rank),
        )

    plain_console = Console(width=160, color_system=None, file=None)
    with plain_console.capture() as capture:
        plain_console.print(table)
    return capture.get()


def doc_corpus_label(d: Document) -> str:
    src = os.path.basename(d.metadata.get("source", ""))
    return "Informazioni per i pazienti" if src == PATIENT_DOC else "Letteratura scientifica"


_RECOMMENDATION_KEYS = {"raccomandazioni", "raccomandazioni principali"}


def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation for robust section matching."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"[^a-z0-9 ]", "", text)


def render_context(docs: list[Document]) -> str:
    def section(d: Document) -> str:
        return d.metadata.get("section", "main")

    def is_recommendation(d: Document) -> bool:
        return _normalize(section(d)) in _RECOMMENDATION_KEYS

    docs_sorted = sorted(
        docs,
        key=lambda d: (
            doc_corpus_label(d) != "Informazioni per i pazienti",  # patient doc first
            not is_recommendation(d),  # recommendations first
        ),
    )

    parts = []
    for i, d in enumerate(docs_sorted, 1):
        chunk_type = d.metadata.get("type", "text")
        parts.append(
            f"[{i}] Corpus: {doc_corpus_label(d)}\n"
            f"Section: {section(d)}\n"
            f"Source: {os.path.basename(d.metadata.get('source', 'unknown'))}\n"
            f"Type: {chunk_type}\n"
            f"Excerpt: {d.page_content.strip()}\n"
        )

    return "\n".join(parts)

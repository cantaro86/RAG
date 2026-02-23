import os

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table as RichTable

from src.loggers import Logger

logger = Logger.get_logger(__name__)


PATIENT_DOC = "3.6.25.pdf"


def print_sources(docs: list[Document]) -> str:
    """
    Create a formatted table of source documents and return it as plain text.
    """
    table = RichTable(title="Top Context Chunks")
    table.add_column("#")
    table.add_column("Source")
    table.add_column("Page start")
    table.add_column("Page end")
    table.add_column("Chars")
    table.add_column("Rank")
    table.add_column("OCR")

    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page_start = str(d.metadata.get("page_start", "?"))
        page_end = str(d.metadata.get("page_end", "?"))
        rank = d.metadata.get("rerank_score", -99)
        ocr = str(d.metadata.get("ocr_used", "NA"))
        table.add_row(str(i), src, page_start, page_end, str(len(d.page_content)), str(rank), ocr)

    # Use a plain console to capture plain text output
    plain_console = Console(width=120, color_system=None, file=None)
    with plain_console.capture() as capture:
        plain_console.print(table)
    return capture.get()


def doc_corpus_label(d: Document) -> str:
    src = os.path.basename(d.metadata.get("source", ""))
    return "Information for patients" if src == PATIENT_DOC else "Colonoscopy literature"


def render_context(docs: list[Document]) -> str:
    def section(d):
        return d.metadata.get("section", "main")

    docs_sorted = sorted(
        docs,
        key=lambda d: (
            doc_corpus_label(d) != "Information for patients",
            section(d) != "Recommendation",
        ),
    )

    parts = []
    for i, d in enumerate(docs_sorted, 1):
        ps, pe = d.metadata.get("page_start"), d.metadata.get("page_end")
        page_str = (
            f"pages {ps}-{pe}"
            if (ps is not None and pe is not None and ps != pe)
            else f"page {d.metadata.get('page', ps if ps is not None else '?')}"
        )
        parts.append(
            f"[{i}] Corpus: {doc_corpus_label(d)}\n"
            f"Section: {section(d)}\n"
            f"Source: {os.path.basename(d.metadata.get('source', 'unknown'))}, {page_str}\n"
            f"Excerpt: {d.page_content.strip()}\n"
        )
    return "\n".join(parts)


def prefer_pagepair_over_page(parents, *, score_key="rerank_score"):
    # Keep best-scored doc per key first (so duplicates collapse deterministically)
    def score(d):
        v = d.metadata.get(score_key, None)
        return float(v) if v is not None else 0.0

    # Index pagepairs by (source, page_start, page_end)
    pagepairs = {}
    for p in parents:
        if p.metadata.get("window") == "pagepair":
            src = p.metadata.get("source")
            ps, pe = p.metadata.get("page_start"), p.metadata.get("page_end")
            if src and ps is not None and pe is not None:
                k = (src, ps, pe)
                if (k not in pagepairs) or (score(p) > score(pagepairs[k])):
                    pagepairs[k] = p

    out = []
    for p in parents:
        if p.metadata.get("window") != "page":
            out.append(p)
            continue

        src = p.metadata.get("source")
        pg = p.metadata.get("page_start")
        if src is None or pg is None:
            out.append(p)
            continue

        # Is this single page contained in an adjacent pagepair?
        contained = ((src, pg - 1, pg) in pagepairs) or ((src, pg, pg + 1) in pagepairs)
        if not contained:
            out.append(p)

    # Finally, dedup exact same (source, window, page_start, page_end) keeping best score
    best = {}
    for p in out:
        k = (
            p.metadata.get("source"),
            p.metadata.get("window"),
            p.metadata.get("page_start"),
            p.metadata.get("page_end"),
        )
        if (k not in best) or (score(p) > score(best[k])):
            best[k] = p
    return list(best.values())

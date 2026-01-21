import os

from langchain_core.documents import Document
from rich.table import Table as RichTable

from src._load_env import console

from .loggers import Logger

logger = Logger.get_logger(__name__)


def print_sources(docs: list[Document]) -> str:
    """
    Create a formatted table of source documents and return it as plain text.
    """
    table = RichTable(title="Top Context Chunks")
    table.add_column("#")
    table.add_column("Source")
    table.add_column("Page")
    table.add_column("Chars")
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page = str(d.metadata.get("page", "?"))
        table.add_row(str(i), src, page, str(len(d.page_content)))

    # Capture only this table's output using the global console
    with console.capture() as capture:
        console.print(table)
    return capture.get()

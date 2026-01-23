import os

from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table as RichTable

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

    # Use a plain console to capture plain text output
    plain_console = Console(width=120, color_system=None, file=None)
    with plain_console.capture() as capture:
        plain_console.print(table)
    return capture.get()

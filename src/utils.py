import json
import os
import re

from langchain.schema import Document
from rich.table import Table as RichTable

from src._load_env import console

from .loggers import Logger

logger = Logger.get_logger(__name__)


def extract_last_json(raw_text: str):
    """
    Extract the last valid JSON object from the model output.
    If none found or invalid, return {"score": "no"}.
    """
    # Match any {...} including across line breaks
    matches = re.findall(r"\{[^{}]+\}", raw_text, re.DOTALL)
    if not matches:
        logger.debug("No JSON object found in output.")
        return {"score": "no"}

    last = matches[-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        # Clean potential trailing commas or artifacts
        cleaned = re.sub(r",\s*}", "}", last.strip())
        try:
            return json.loads(cleaned)
        except Exception:
            logger.debug("Failed to decode last JSON object.")
            return {"score": "no"}


def extract_answer_text(raw_text: str):
    """
    Extracts the text following the 'Answer:' section in model output.
    If no explicit 'Answer:' found, returns the full text.
    """
    # Try to find 'Answer:' ignoring case
    match = re.search(r"(?i)answer\s*:\s*(.*)", raw_text, re.DOTALL)
    if match:
        logger.debug("Extracted answer section from output.")
        return match.group(1).strip()
    else:
        logger.debug("No explicit 'Answer:' section found; returning full output.")
        # If no explicit "Answer:" header, return full output
        return raw_text.strip()


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

    return console.export_text()

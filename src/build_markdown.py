import re
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from tqdm import tqdm

from src._load_env import console
from src.loggers import Logger

logger = Logger.get_logger(__name__)


LIGATURE_MAP = {
    "/uniFB00": "ff",
    "/uniFB01": "fi",
    "/uniFB02": "fl",
    "/uniFB03": "ffi",
    "/uniFB04": "ffl",
    "/uniFB05": "st",
    "/uniFB06": "st",
}

GLYPH_MAP = {
    "/C210": "",
    "/C226": "",
    "/C21": ">=",
    "/C20": "<=",
    "/C15": "•",
    "/C0": "-",
}

HTML_ENTITY_MAP = {
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
}

ALL_MAP = {**GLYPH_MAP, **LIGATURE_MAP}
_sorted_tokens = sorted(ALL_MAP.keys(), key=len, reverse=True)
_glyph_pattern = re.compile("|".join(map(re.escape, _sorted_tokens)))
_html_pattern = re.compile("|".join(map(re.escape, HTML_ENTITY_MAP.keys())))


def clean_pdf_artifacts(text: str) -> str:
    # 1. Glyph codes and ligature tokens
    text = _glyph_pattern.sub(lambda m: ALL_MAP[m.group(0)], text)

    # 2. HTML entities
    text = _html_pattern.sub(lambda m: HTML_ENTITY_MAP[m.group(0)], text)

    # 3. Non-breaking space → regular space
    text = text.replace("\xa0", " ")

    # 4. Private Use Area glyph (unmappable artifact)
    text = text.replace("\ue840", "")

    # 5. þ used as plus sign (only in these docs — verify if adding new docs)
    text = text.replace("þ", "+")

    # 6. Stray spacing accents (not part of a composed character)
    text = text.replace("\u00b4", "'")  # ´ acute → apostrophe
    text = text.replace("\u00a8", "")  # ¨ diaeresis → strip
    text = text.replace("\u00bc", "=")  # ¼ → = (font encoding artifact in this corpus)

    # 7. Fix word-splitting from ligature replacement ("speci fi c" → "specific")
    text = re.sub(r"(\w) (ff|fi|fl|ffi|ffl|st) (\w)", r"\1\2\3", text)

    # 8. Fix spacing around comparison operators (">= 10" → ">=10")
    text = re.sub(r"([><]=)\s+(\d)", r"\1\2", text)

    # 9. Collapse multiple spaces (side effect of stripping tokens)
    text = re.sub(r"  +", " ", text)

    return text


def table_within(lines: list[str], start: int, window: int = 6) -> bool:
    """Return True if a table row appears within the next `window` non-blank lines."""
    text_lines_seen = 0
    blank_lines = 0
    for j in range(start, len(lines)):
        if lines[j] == "":
            blank_lines += 1
            if blank_lines >= 2:
                return False
            continue
        if lines[j].startswith("|"):
            return True
        text_lines_seen += 1
        if text_lines_seen >= window:
            return False
    return False


def clean_newlines(text: str) -> str:
    lines = text.split("\n")
    result = []

    for i, line in enumerate(lines):
        if line == "":
            prev_line = lines[i - 1] if i > 0 else ""
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            is_structural = (
                prev_line.startswith("|")  # blank line after table → KEEP
                or next_line.startswith("#")  # blank line before heading → KEEP
                or (
                    not next_line.startswith("|")  # blank line before caption → KEEP
                    and table_within(lines, i + 1)
                )  #   only if table comes ahead
                # next_line.startswith('|') → NOT here → blank between caption and table → DROP
            )
            if not is_structural:
                continue
        result.append(line)
    return "\n".join(result)


def fix_paragraph_ordering(text: str) -> str:
    """
    Fix mis-ordered paragraphs caused by multi-column layout detection:
    - If a paragraph starts with lowercase and previous has no terminal punctuation → join them.
    - If a paragraph starts with lowercase, previous ends with punctuation but
      the one before that does not → join with the paragraph before the previous.
    - Never join bullet/list items, headings, or tables.
    """
    terminal_re = re.compile(r"([.?!:\]\)]|\[\d+\])\s*$")
    heading_re = re.compile(r"^#{1,6} ")
    table_re = re.compile(r"^\|")
    bullet_re = re.compile(r"^[-•*▶■]|\d+\.")

    def ends_terminal(b):
        return bool(terminal_re.search(b))

    def is_special(b):
        return bool(heading_re.match(b) or table_re.match(b))

    def is_bullet(b):
        return bool(bullet_re.match(b))

    def starts_lower(b):
        return bool(re.match(r"^[a-z]", b))

    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]

    i = 0
    while i < len(blocks):
        block = blocks[i]

        if not starts_lower(block) or is_special(block) or is_bullet(block):
            i += 1
            continue

        if i > 0:
            prev = blocks[i - 1]

            # Case 1: previous block has no terminal punctuation → join directly
            if not is_special(prev) and not ends_terminal(prev):
                blocks[i - 1] = prev + " " + block
                blocks.pop(i)
                continue  # recheck same index (now points to next block)

            # Case 2: previous ends with punctuation but one before that does not
            # → current block is the tail of the paragraph before the previous one
            elif i > 1:
                prev_prev = blocks[i - 2]
                if not is_special(prev_prev) and not ends_terminal(prev_prev) and ends_terminal(prev):
                    blocks[i - 2] = prev_prev + " " + block
                    blocks.pop(i)
                    continue

        i += 1

    return "\n\n".join(blocks)


def clean_markdown_artifacts(text: str) -> str:
    """Remove docling image placeholders and extra blank lines they leave behind."""
    # Remove <!-- image --> tags (with optional whitespace variants)
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    # Remove blank lines that the removed tags may leave behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def find_missing_markdown(pdf_path: str, output_path: str) -> list[Path]:
    """
    Check which PDF/DOCX files in pdf_path don't have corresponding .md files in output_path.

    Args:
        pdf_path: Path to directory containing PDF/DOCX files
        output_path: Path to directory containing markdown files

    Returns:
        List of Path objects for missing files
    """
    input_dir = Path(pdf_path)
    output_dir = Path(output_path)

    # Supported input formats
    input_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.docx"))

    missing = []

    for input_file in sorted(input_files):
        stem = input_file.stem
        expected_md = output_dir / f"{stem}.md"

        if not expected_md.exists():
            missing.append(input_file)

    return missing


def create_markdown_from_pdf(
    pdf_path: str,
    output_path: str,
    files: list[str] | None = None,
) -> None:
    """Convert a PDF to cleaned markdown using docling."""

    INPUT_FOLDER = Path(pdf_path)
    OUTPUT_FOLDER = Path(output_path)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    logger.info(f"Processing PDF files: {pdf_path}")

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = False

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CUDA,
    )

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    if files is not None:
        input_files = files
    else:
        input_files = list(INPUT_FOLDER.glob("*.pdf")) + list(INPUT_FOLDER.glob("*.docx"))

    for doc_path in tqdm(sorted(input_files), desc=f"Processing PDFs in {pdf_path}"):
        logger.info(f"Processing: {doc_path.name}")
        result = converter.convert(doc_path)

        markdown_text = result.document.export_to_markdown()
        markdown_text = clean_pdf_artifacts(markdown_text)
        markdown_text = fix_paragraph_ordering(markdown_text)
        markdown_text = clean_markdown_artifacts(markdown_text)
        markdown_text = clean_newlines(markdown_text)

        out_path = OUTPUT_FOLDER / f"{doc_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")
        logger.info(f"  -> Saved: {out_path}")

    console.print("\nMarkdown creation done.")
    logger.info(f"Saved cleaned markdown to: {output_path}")

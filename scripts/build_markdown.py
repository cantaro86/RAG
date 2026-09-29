"""Convert top-level PDF and DOCX files into cleaned Markdown with Docling.

This is a repository utility, not part of the ``agentic_rag`` runtime package.
Run it from the repository root so that ``uv`` uses the project's locked
environment.

Install the optional conversion dependencies first::

    uv sync --locked --extra dev --extra markdown

Convert every ``.pdf`` and ``.docx`` file directly inside an input directory::

    uv run python scripts/build_markdown.py Colon-TC Markdown_IT

Convert only selected files. Relative file names are resolved against the input
directory; absolute paths are also accepted::

    uv run python scripts/build_markdown.py Colon-TC Markdown_IT --files patient-guide.pdf instructions.docx

Force a supported processing device when Docling's automatic selection is not
appropriate::

    uv run python scripts/build_markdown.py Colon-TC Markdown_IT --device cpu

The output directory is created when necessary. Each input produces
``<input-stem>.md`` and replaces an existing file with the same name. Directory
discovery is non-recursive. Conversion uses accurate table extraction without
OCR, then removes known PDF glyph, layout, image-placeholder, and newline
artifacts. Importing this module for its cleaning helpers does not require
Docling; Docling is loaded only when conversion starts.

The utility can also be used programmatically::

    from scripts.build_markdown import create_markdown_from_pdf

    create_markdown_from_pdf(
        "Colon-TC",
        "Markdown_IT",
        files=["patient-guide.pdf"],
        device="cpu",
    )
"""

import argparse
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from tqdm import tqdm

logger = logging.getLogger(__name__)
console = Console()


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
SUPPORTED_INPUT_SUFFIXES = {".docx", ".pdf"}
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


def _validate_input_directory(input_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")


def _discover_input_files(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )


def _normalize_input_files(input_dir: Path, files: Iterable[str | Path] | None) -> list[Path]:
    if files is None:
        input_files = _discover_input_files(input_dir)
    else:
        raw_files = [files] if isinstance(files, str | Path) else files
        input_files = []
        for file in raw_files:
            path = Path(file)
            if not path.is_absolute():
                candidate = input_dir / path
                if candidate.exists() or not path.exists():
                    path = candidate
            input_files.append(path)
        input_files.sort(key=lambda path: str(path))

    if not input_files:
        raise ValueError(f"No PDF or DOCX files found in: {input_dir}")

    for input_file in input_files:
        if input_file.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {input_file}")
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        if not input_file.is_file():
            raise ValueError(f"Input path is not a file: {input_file}")

    return input_files


def _load_docling() -> SimpleNamespace:
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "Markdown conversion requires Docling. Install it with `uv sync --locked --extra markdown`."
        ) from exc

    return SimpleNamespace(
        AcceleratorDevice=AcceleratorDevice,
        AcceleratorOptions=AcceleratorOptions,
        DocumentConverter=DocumentConverter,
        InputFormat=InputFormat,
        PdfFormatOption=PdfFormatOption,
        PdfPipelineOptions=PdfPipelineOptions,
        TableFormerMode=TableFormerMode,
    )


def find_missing_markdown(pdf_path: str | Path, output_path: str | Path) -> list[Path]:
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
    _validate_input_directory(input_dir)

    input_files = _discover_input_files(input_dir)

    missing = []

    for input_file in sorted(input_files):
        stem = input_file.stem
        expected_md = output_dir / f"{stem}.md"

        if not expected_md.exists():
            missing.append(input_file)

    return missing


def create_markdown_from_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    files: Iterable[str | Path] | None = None,
    device: str | None = None,
) -> None:
    """Convert PDF and DOCX files to cleaned Markdown using Docling."""

    input_folder = Path(pdf_path)
    output_folder = Path(output_path)
    _validate_input_directory(input_folder)
    input_files = _normalize_input_files(input_folder, files)
    output_folder.mkdir(parents=True, exist_ok=True)

    docling = _load_docling()

    logger.info(f"Processing PDF files: {pdf_path}")

    pipeline_options = docling.PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,
    )
    pipeline_options.table_structure_options.mode = docling.TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = False

    pipeline_options.accelerator_options = docling.AcceleratorOptions(
        num_threads=4,
        device=docling.AcceleratorDevice.AUTO if device is None else device,
    )

    converter = docling.DocumentConverter(
        format_options={docling.InputFormat.PDF: docling.PdfFormatOption(pipeline_options=pipeline_options)}
    )

    for doc_path in tqdm(sorted(input_files), desc=f"Processing PDFs in {pdf_path}"):
        logger.info(f"Processing: {doc_path.name}")
        result = converter.convert(doc_path)

        markdown_text = result.document.export_to_markdown()
        markdown_text = clean_pdf_artifacts(markdown_text)
        markdown_text = fix_paragraph_ordering(markdown_text)
        markdown_text = clean_markdown_artifacts(markdown_text)
        markdown_text = clean_newlines(markdown_text)

        out_path = output_folder / f"{doc_path.stem}.md"
        out_path.write_text(markdown_text, encoding="utf-8")
        logger.info(f"  -> Saved: {out_path}")

    console.print("\nMarkdown creation done.")
    logger.info(f"Saved cleaned markdown to: {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory containing top-level PDF and DOCX files.")
    parser.add_argument("output_dir", type=Path, help="Directory where generated Markdown files are written.")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Optional input file names or paths. By default, every supported file in input_dir is converted.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Docling accelerator device. By default, Docling selects the device automatically.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    create_markdown_from_pdf(
        args.input_dir,
        args.output_dir,
        files=args.files,
        device=args.device,
    )


if __name__ == "__main__":
    main()

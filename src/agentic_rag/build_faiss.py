# ------------------------
# Build / load PDFs
# ------------------------


import os
import re
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect

from agentic_rag._load_env import DEVICE, ONLINE, Config, cfg, console
from agentic_rag.loggers import Logger
from agentic_rag.utils import _normalize

logger = Logger.get_logger(__name__)


SKIP_SECTIONS = {"references", "riferimenti", "bibliografia", "referenze"}
NOSPLIT_SECTIONS = {
    "recommendation",
    "recommendations",
    "raccomandazione",
    "raccomandazioni",
    "main recommendation",
    "main recommendations",
}


def _detect_lang_safe(text: str) -> str:
    try:
        if text and len(text) >= 120:
            return detect(text)
    except Exception:
        pass
    return "unknown"


TABLE_RE = re.compile(r"^\|", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^(?=#{1,6} )", re.MULTILINE)

LIST_ITEM_RE = re.compile(
    r"^(?:"
    r"(?:\*\*)?(?P<num>\d{1,2})(?:\*\*)?(?:[\.\)]\s|\s(?=[A-Z]))|"
    r"[-•*]\s"
    r")"
)

CONTINUATION_RE = re.compile(r"^(?:\*\*)?(?P<num>\d{1,2})(?:\*\*)?(?:[\.\)]\s|\s(?=[A-Z]|\d))")


_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n", ". ", "? ", "! ", " ", ""],
    keep_separator="end",
)

# Splitter that never splits (chunk_size large enough for any section)
_NOSPLIT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=100_000,
    chunk_overlap=0,
    separators=[],
)


def extract_list_number(line: str) -> int | None:
    m = CONTINUATION_RE.match(line)
    if not m:
        return None
    return int(m.group("num"))


def split_block_by_type(block: str) -> list[tuple[str, str]]:
    lines = block.splitlines()
    segments = []
    current_type = None
    current_lines = []
    pending_intro = None  # last text line ending with ":" to attach to next list

    def commit():
        if current_lines:
            segments.append((current_type, "\n".join(current_lines)))
            current_lines.clear()

    for line in lines:
        is_list_line = bool(LIST_ITEM_RE.match(line))
        is_heading = bool(HEADING_RE.match(line))
        is_table = bool(TABLE_RE.match(line))

        if is_heading or is_table:
            pending_intro = None
            commit()
            current_type = "heading" if is_heading else "text"
            current_lines.append(line)

        elif is_list_line:
            if current_type == "list":
                # continuing same list
                current_lines.append(line)
            else:
                # new list starts: first commit whatever was open
                # but strip the last text line if it is an intro for this list
                if current_type == "text" and current_lines and current_lines[-1].rstrip().endswith(":"):
                    pending_intro = current_lines.pop(-1)
                commit()
                current_type = "list"
                if pending_intro is not None:
                    current_lines.append(pending_intro)
                    pending_intro = None
                current_lines.append(line)

        else:
            if current_type == "list":
                if line.startswith((" ", "\t")):
                    current_lines.append(line)
                else:
                    commit()
                    current_type = "text"
                    current_lines.append(line)
            else:
                if current_type != "text":
                    commit()
                    current_type = "text"
                current_lines.append(line)

    commit()
    return segments


def _make_metadata(source: str, section: str, language: str, **extra) -> dict:
    meta = {"source": source, "section": section, "language": language}
    meta.update(extra)
    return meta


def _flush_text(
    *,
    chunks: list[Document],
    pending_text: list[str],
    carried_heading: str,
    source: str,
    section: str,
    language: str,
    splitter: RecursiveCharacterTextSplitter,
) -> str:
    """Flush pending text and/or carried heading into chunks.
    Always returns the new value of carried_heading (empty string if flushed).
    """
    if pending_text:
        prefix = f"{carried_heading}\n" if carried_heading else ""
        combined = prefix + "\n".join(pending_text)

        for sub in splitter.create_documents(
            texts=[combined],
            metadatas=[_make_metadata(source, section, language, type="text")],
        ):
            chunks.append(sub)

        pending_text.clear()
        return ""

    if carried_heading:
        chunks.append(
            Document(
                page_content=carried_heading,
                metadata=_make_metadata(source, section, language, type="text"),
            )
        )
        return ""

    return carried_heading  # nothing to flush, pass through unchanged


def chunk_markdown(md_path: Path) -> list[Document]:
    raw = md_path.read_text(encoding="utf-8")
    doc_lang = _detect_lang_safe(raw)
    chunks: list[Document] = []
    current_section = ""
    carried_heading = ""

    sections = re.split(SECTION_RE, raw)

    for section in sections:
        if not section.strip():
            continue

        first_line = section.splitlines()[0].strip()
        heading_match = HEADING_RE.match(first_line)
        if heading_match:
            current_section = heading_match.group(2)

        # ── Section-level rules ──────────────────────────────────
        section_key = _normalize(current_section)

        if section_key in SKIP_SECTIONS:
            continue

        active_splitter = _NOSPLIT_SPLITTER if section_key in NOSPLIT_SECTIONS else _SPLITTER
        # ─────────────────────────────────────────────────────────

        blocks = [b.strip() for b in re.split(r"\n{2,}", section) if b.strip()]
        pending_text: list[str] = []

        for block in blocks:
            is_heading = "\n" not in block and bool(HEADING_RE.match(block))
            is_table = bool(TABLE_RE.search(block))

            if is_table:
                if pending_text:
                    carried_heading = _flush_text(
                        chunks=chunks,
                        pending_text=pending_text,
                        carried_heading=carried_heading,
                        source=md_path.name,
                        section=current_section,
                        language=doc_lang,
                        splitter=active_splitter,
                    )
                table_content = f"{carried_heading}\n{block}".strip() if carried_heading else block
                carried_heading = ""
                chunks.append(
                    Document(
                        page_content=table_content,
                        metadata=_make_metadata(md_path.name, current_section, doc_lang, type="table"),
                    )
                )

            elif is_heading:
                carried_heading = _flush_text(
                    chunks=chunks,
                    pending_text=pending_text,
                    carried_heading=carried_heading,
                    source=md_path.name,
                    section=current_section,
                    language=doc_lang,
                    splitter=active_splitter,
                )
                carried_heading = f"{carried_heading}\n{block}".strip() if carried_heading else block

            else:
                segments = split_block_by_type(block)
                if any(t == "list" for t, _ in segments):
                    for seg_type, seg_content in segments:
                        if seg_type == "list":
                            list_intro = ""
                            if pending_text:
                                last_lines = pending_text[-1].splitlines()
                                if last_lines[-1].rstrip().endswith(":"):
                                    list_intro = last_lines[-1].rstrip()
                                    if len(last_lines) == 1:
                                        pending_text.pop()
                                    else:
                                        pending_text[-1] = "\n".join(last_lines[:-1])
                            if pending_text:
                                carried_heading = _flush_text(
                                    chunks=chunks,
                                    pending_text=pending_text,
                                    carried_heading=carried_heading,
                                    source=md_path.name,
                                    section=current_section,
                                    language=doc_lang,
                                    splitter=active_splitter,
                                )
                            prefix_parts = []
                            if carried_heading:
                                prefix_parts.append(carried_heading)
                                carried_heading = ""
                            if list_intro:
                                prefix_parts.append(list_intro)
                            prefix = "\n".join(prefix_parts)
                            content = f"{prefix}\n{seg_content}".strip() if prefix else seg_content
                            chunks.append(
                                Document(
                                    page_content=content,
                                    metadata=_make_metadata(md_path.name, current_section, doc_lang, type="list"),
                                )
                            )
                        elif seg_type == "heading":
                            carried_heading = _flush_text(
                                chunks=chunks,
                                pending_text=pending_text,
                                carried_heading=carried_heading,
                                source=md_path.name,
                                section=current_section,
                                language=doc_lang,
                                splitter=active_splitter,
                            )
                            carried_heading = (
                                f"{carried_heading}\n{seg_content}".strip() if carried_heading else seg_content
                            )
                        else:
                            pending_text.append(seg_content)
                else:
                    pending_text.append(block)

        carried_heading = _flush_text(
            chunks=chunks,
            pending_text=pending_text,
            carried_heading=carried_heading,
            source=md_path.name,
            section=current_section,
            language=doc_lang,
            splitter=active_splitter,
        )

    return chunks


def merge_continuation_lists(chunks: list[Document]) -> list[Document]:
    """
    Merge list chunks that start at a numbered item > 1 into the most recent
    preceding numbered list chunk from the same source, but only if that
    predecessor's last numbered item is exactly n-1.

    Noise-only list chunks (with no numbered items) are skipped during the search.
    Leading noise lines before a continuation item are moved to the nearest
    preceding chunk of any type.
    """
    result = list(chunks)
    i = 0

    while i < len(result):
        chunk = result[i]

        if chunk.metadata.get("type") != "list":
            i += 1
            continue

        lines = chunk.page_content.splitlines()
        continuation_n = None
        noise_line_count = 0

        for idx, line in enumerate(lines[:5]):
            n = extract_list_number(line)
            if n is None:
                continue
            if n <= 1:
                break
            continuation_n = n
            noise_line_count = idx
            break

        if continuation_n is None:
            i += 1
            continue

        if noise_line_count > 0:
            noise_content = "\n".join(lines[:noise_line_count])
            clean_content = "\n".join(lines[noise_line_count:])

            result[i] = Document(
                page_content=clean_content,
                metadata=chunk.metadata,
            )
            chunk = result[i]

            if i > 0 and noise_content.strip():
                prev = result[i - 1]
                result[i - 1] = Document(
                    page_content=prev.page_content + "\n" + noise_content,
                    metadata=prev.metadata,
                )

        source = chunk.metadata.get("source")
        predecessor_idx = None

        for j in range(i - 1, -1, -1):
            prev_chunk = result[j]
            if prev_chunk.metadata.get("type") != "list":
                continue
            if prev_chunk.metadata.get("source") != source:
                continue

            has_numbered = any(extract_list_number(line) is not None for line in prev_chunk.page_content.splitlines())
            if has_numbered:
                predecessor_idx = j
                break

        if predecessor_idx is None:
            i += 1
            continue

        pred_last_n = None
        for line in reversed(result[predecessor_idx].page_content.splitlines()):
            n = extract_list_number(line)
            if n is not None:
                pred_last_n = n
                break

        if pred_last_n == continuation_n - 1:
            merged_content = result[predecessor_idx].page_content + "\n" + chunk.page_content
            result[predecessor_idx] = Document(
                page_content=merged_content,
                metadata=result[predecessor_idx].metadata,
            )
            result.pop(i)
        else:
            i += 1

    return result


# ------------------------
# Build FAISS
# ------------------------
def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    offline = not (ONLINE and getattr(cfg, "online", True))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={
            "device": DEVICE,
            "trust_remote_code": True,
            "local_files_only": offline,
        },
    )


def build_faiss_index(cfg: Config) -> None:
    console.rule("[bold]Indexing Markdown -> FAISS")
    logger.info("Indexing Markdown -> FAISS")

    all_chunks: list[Document] = []
    md_files = sorted(Path(cfg.md_dir).glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {cfg.md_dir}")

    for md_path in md_files:
        doc_chunks = chunk_markdown(md_path)  # structure-aware chunking
        all_chunks.extend(doc_chunks)
        logger.debug("%s: %d chunks", md_path.name, len(doc_chunks))

    chunks = merge_continuation_lists(all_chunks)
    chunks = [c for c in chunks if len(c.page_content) > cfg.min_chunk_length]

    console.print(f"Loaded [bold]{len(md_files)}[/bold] files -> [bold]{len(chunks)}[/bold] chunks.")
    logger.info("Loaded %d files -> %d chunks.", len(md_files), len(chunks))

    embedder = build_embedder(cfg.embed_model)
    vs = FAISS.from_documents(chunks, embedder)

    os.makedirs(cfg.index_dir, exist_ok=True)
    vs.save_local(cfg.index_dir)
    console.print(f"Saved FAISS index to [bold]{cfg.index_dir}[/bold]")
    logger.info("Saved FAISS index to %s", cfg.index_dir)


def load_vectorstore(index_dir: str, embed_model: str) -> FAISS:
    embedder = build_embedder(embed_model)
    vs = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    if getattr(cfg, "use_gpu_index", False):
        try:
            if faiss.get_num_gpus() > 0:
                logger.info("FAISS GPU detected: %d GPU(s). Moving loaded index to GPU...", faiss.get_num_gpus())
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)
                vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
            else:
                logger.warning("No GPU detected by FAISS. Using CPU index.")
        except ImportError:
            logger.warning("FAISS GPU not available. Using CPU index.")
    else:
        logger.info("Using FAISS index on CPU as per configuration.")

    return vs

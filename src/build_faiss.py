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

from src._load_env import DEVICE, ONLINE, Config, cfg, console
from src.loggers import Logger
from src.translate import translate_docs_it_to_en

logger = Logger.get_logger(__name__)


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
LIST_ITEM_RE = re.compile(r"^(\d{1,2}[\.\)]\s|\d{1,2}\s(?=[A-Z])|[-•*]\s)")
CONTINUATION_RE = re.compile(r"^(\d{1,2})[\.\)]?\s(?=[A-Z]|\d)")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n", ". ", "? ", "! ", " ", ""],
    keep_separator="end",
)


def split_block_by_type(block: str) -> list[tuple[str, str]]:
    lines = block.splitlines()
    segments = []
    current_type = None
    current_lines = []

    def commit():
        if current_lines:
            segments.append((current_type, "\n".join(current_lines)))
            current_lines.clear()

    for line in lines:
        is_list_line = bool(LIST_ITEM_RE.match(line))
        is_heading = bool(HEADING_RE.match(line))
        is_table = bool(TABLE_RE.match(line))

        if is_heading or is_table:
            # Always a hard boundary
            commit()
            current_type = "text"
            current_lines.append(line)
        elif is_list_line:
            if current_type != "list":
                commit()
                current_type = "list"
            current_lines.append(line)
        else:
            if current_type == "list":
                # Stay in list mode — could be continuation text of an item
                # or noise between items (handled by merge_continuation_lists)
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

        blocks = [b.strip() for b in re.split(r"\n{2,}", section) if b.strip()]
        pending_text: list[str] = []

        for block in blocks:
            is_heading = "\n" not in block and bool(HEADING_RE.match(block))
            is_table = bool(TABLE_RE.search(block))

            if is_table:
                carried_heading = _flush_text(
                    chunks=chunks,
                    pending_text=pending_text,
                    carried_heading=carried_heading,
                    source=md_path.name,
                    section=current_section,
                    language=doc_lang,
                    splitter=splitter,
                )
                chunks.append(
                    Document(
                        page_content=block,
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
                    splitter=splitter,
                )
                carried_heading = f"{carried_heading}\n{block}".strip() if carried_heading else block

            else:
                segments = split_block_by_type(block)
                if any(t == "list" for t, _ in segments):
                    for seg_type, seg_content in segments:
                        if seg_type == "list":
                            carried_heading = _flush_text(
                                chunks=chunks,
                                pending_text=pending_text,
                                carried_heading=carried_heading,
                                source=md_path.name,
                                section=current_section,
                                language=doc_lang,
                                splitter=splitter,
                            )
                            chunks.append(
                                Document(
                                    page_content=seg_content,
                                    metadata=_make_metadata(md_path.name, current_section, doc_lang, type="list"),
                                )
                            )
                        else:
                            pending_text.append(seg_content)
                else:
                    pending_text.append(block)

        # Capture return value to reset carried_heading between sections
        carried_heading = _flush_text(
            chunks=chunks,
            pending_text=pending_text,
            carried_heading=carried_heading,
            source=md_path.name,
            section=current_section,
            language=doc_lang,
            splitter=splitter,
        )

    return chunks


def merge_continuation_lists(chunks: list[Document]) -> list[Document]:
    """
    Merge list chunks that start at item > 1 into the most recent
    preceding list chunk from the same source document.
    """
    result = list(chunks)  # copy to avoid mutating original

    i = 0
    while i < len(result):
        chunk = result[i]
        if chunk.metadata.get("type") != "list":
            i += 1
            continue

        # Check if this list starts at item > 1 (continuation)
        first_line = chunk.page_content.splitlines()[0]
        m = CONTINUATION_RE.match(first_line)
        if not m or int(m.group(1)) <= 1:
            i += 1
            continue

        # Find the most recent list chunk from the same source
        source = chunk.metadata.get("source")
        predecessor_idx = None
        for j in range(i - 1, -1, -1):
            if result[j].metadata.get("type") == "list" and result[j].metadata.get("source") == source:
                predecessor_idx = j
                break

        if predecessor_idx is not None:
            # Merge: append continuation text to predecessor
            merged_content = result[predecessor_idx].page_content + "\n" + chunk.page_content
            result[predecessor_idx] = Document(
                page_content=merged_content,
                metadata=result[predecessor_idx].metadata,
            )
            result.pop(i)  # remove the continuation chunk
            # Don't increment i — recheck the same position
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

    it_chunks = [c for c in chunks if c.metadata.get("language") == "it"]
    logger.info("Found %d Italian chunks to translate.", len(it_chunks))

    en_or_other_chunks = [c for c in chunks if c.metadata.get("language") != "it"]

    if it_chunks and cfg.translate_pdf:
        en_chunks = translate_docs_it_to_en(
            it_chunks, src_lang="it", tgt_lang="en", parallel=cfg.translate_parallel, max_workers=cfg.translate_workers
        )
        chunks = en_chunks + en_or_other_chunks

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

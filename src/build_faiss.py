# ------------------------
# Build / load PDFs
# ------------------------


import glob
import hashlib
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import faiss
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from tqdm import tqdm

from src._load_env import DEVICE, ONLINE, Config, cfg, console
from src.loggers import Logger
from src.translate import translate_docs_it_to_en

logger = Logger.get_logger(__name__)


TESS_LANG_MAP: dict[str, str] = {
    "en": "eng+ita",
    "it": "ita+eng",
}


def _parent_id(meta: dict) -> str:
    s = f"{meta.get('source', '')}|{meta.get('page_start')}|{meta.get('page_end')}|{meta.get('window')}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _save_parents_map(parents_map: dict[str, Document], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(parents_map, f)


@lru_cache(maxsize=1)
def load_parents_map(path: str) -> dict[str, Document]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _detect_lang_safe(text: str) -> str:
    try:
        if text and len(text) >= 120:
            return detect(text)
    except Exception:
        pass
    return "unknown"


def _tess_lang(lang: str) -> str:
    return TESS_LANG_MAP.get(lang, "eng+ita")


def _ocr_one_page(task: tuple[str, int, str, int, str | None]) -> tuple[str, int, str, str, str | None]:
    """
    Worker task.
    Returns: (pdf_path, page_index, lang, ocr_text, ocr_error)
    """
    pdf_path, page_index, lang, ocr_dpi, tessdata = task
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        tp = page.get_textpage_ocr(
            language=_tess_lang(lang),
            dpi=ocr_dpi,
            full=True,
            tessdata=tessdata,  # can be None
        )
        text = page.get_text("text", textpage=tp) or ""
        doc.close()
        return pdf_path, page_index, lang, text, None
    except Exception as e:
        # Return empty text + error; caller may fallback if desired
        return pdf_path, page_index, lang, "", repr(e)


PAGE_RE = re.compile(r"\[PAGE\s+(\d+)\]\s*")


def split_joined_into_pages(joined_text: str) -> list[tuple[int, str]]:
    parts = PAGE_RE.split(joined_text)
    # ["", "1", "text...", "2", "text...", ...]
    pages = []
    for i in range(1, len(parts), 2):
        pnum = int(parts[i])
        ptxt = parts[i + 1].strip()
        pages.append((pnum, ptxt))
    return pages


def make_page_and_pagepair_windows(doc: Document) -> list[Document]:
    pages = split_joined_into_pages(doc.page_content)
    out: list[Document] = []

    for idx, (pnum, ptxt) in enumerate(pages):
        # 1-page window
        out.append(
            Document(
                page_content=f"[PAGE {pnum}]\n{ptxt}",
                metadata={**doc.metadata, "page_start": pnum, "page_end": pnum, "page": pnum, "window": "page"},
            )
        )

        # 2-page window (i + i+1)
        if idx + 1 < len(pages):
            pnum2, ptxt2 = pages[idx + 1]
            out.append(
                Document(
                    page_content=f"[PAGE {pnum}]\n{ptxt}\n[PAGE {pnum2}]\n{ptxt2}",
                    metadata={
                        **doc.metadata,
                        "page_start": pnum,
                        "page_end": pnum2,
                        "page": pnum,
                        "window": "pagepair",
                    },
                )
            )

    return out


def load_pdfs(
    pdf_dir: str,
    *,
    ocr: bool = False,
    mode: str = "page",
    pages_delimiter: str = "\n\f\n",
    ocr_dpi: int = 150,
    ocr_workers: int = 8,
    ocr_chunksize: int = 2,
    tessdata: str | None = None,
) -> list[Document]:
    """
    Load PDFs into LangChain Documents.

    - ocr=False: use LangChain PyMuPDFLoader with mode/page_delimiter support.
    - ocr=True: full-page OCR with PyMuPDF + multiprocessing, supports mode="page" or mode="single".
    - ocr_workers: how many processes (parallel OCR workers).
        - ocr_chunksize: how many tasks to give each worker at a time.
    - tessdata: path to Tesseract's tessdata directory
    """

    if mode not in ("page", "single"):
        raise ValueError("mode must be 'page' or 'single'")

    paths: list[str] = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(os.path.join(pdf_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    # -------------------------
    # No OCR: keep your old approach
    # -------------------------
    if not ocr:
        docs: list[Document] = []
        for p in tqdm(paths, desc=f"Loading PDFs (no OCR, mode={mode})"):
            loader = PyMuPDFLoader(p, mode=mode, pages_delimiter=pages_delimiter)  # supports both [web:41]
            ds = loader.load()
            for d in ds:
                d.metadata = d.metadata or {}
                d.metadata["source"] = p
                lang = _detect_lang_safe(d.page_content)
                d.metadata["language"] = lang
                d.metadata["ocr_used"] = False
            docs.extend(ds)
        return docs

    # -------------------------
    # OCR path: build tasks
    # -------------------------
    # First pass: open each PDF once to discover page_count and (optional) language guess per page
    tasks: list[tuple[str, int, str, int, str | None]] = []
    pdf_pagecounts: dict[str, int] = {}

    for p in tqdm(paths, desc="Scanning PDFs (for OCR tasks)"):
        doc = fitz.open(p)
        pdf_pagecounts[p] = doc.page_count

        # Find FIRST page with native text → set PDF language
        pdf_lang = "unknown"
        for page_index in range(doc.page_count):
            normal_text = doc[page_index].get_text("text").strip()
            if normal_text:
                pdf_lang = _detect_lang_safe(normal_text)
                logger.info(f"{os.path.basename(p)}: page {page_index} sets lang={pdf_lang}")
                break

        # Apply SAME language to ALL pages of this PDF
        for page_index in range(doc.page_count):
            tasks.append((p, page_index, pdf_lang, ocr_dpi, tessdata))

        doc.close()

    # Run OCR in parallel (PyMuPDF recommends multiprocessing; open document in worker) [web:160]
    results: list[tuple[str, int, str, str, str | None]] = []
    with ProcessPoolExecutor(max_workers=ocr_workers) as ex:
        it = ex.map(_ocr_one_page, tasks, chunksize=ocr_chunksize)
        for r in tqdm(it, total=len(tasks), desc=f"OCR pages ({ocr_workers} workers)"):
            results.append(r)

    # -------------------------
    # Build Documents
    # -------------------------
    if mode == "page":
        docs: list[Document] = []
        for pdf_path, page_index, lang, text, err in results:
            meta = {
                "source": pdf_path,
                "page": page_index,
                "language": lang,
                "ocr_used": err is None,
            }
            if err is not None:
                meta["ocr_error"] = err
            docs.append(Document(page_content=text, metadata=meta))

        # stable ordering
        docs.sort(key=lambda d: (d.metadata["source"], d.metadata["page"]))
        return docs

    # mode == "single": join OCR text for each PDF into one Document (cross-page chunks)
    by_pdf: dict[str, list[tuple[int, str, str, str | None]]] = {}
    for pdf_path, page_index, lang, text, err in results:
        by_pdf.setdefault(pdf_path, []).append((page_index, lang, text, err))

    docs_single: list[Document] = []
    for pdf_path, pages in by_pdf.items():
        pages.sort(key=lambda x: x[0])

        # Keep lightweight page markers so you can infer page ranges later if needed.
        joined_parts = [f"\n[PAGE {pi + 1}]\n{txt}" for (pi, _lang, txt, _err) in pages]
        joined_text = pages_delimiter.join(joined_parts)

        # Choose a document language: first non-unknown, else unknown
        langs = [lang_code for (_pi, lang_code, _t, _e) in pages if lang_code != "unknown"]
        doc_lang = langs[0] if langs else "unknown"

        # Track OCR errors at doc-level (optional)
        errors = [(pi, e) for (pi, _l, _t, e) in pages if e is not None]

        meta = {
            "source": pdf_path,
            "language": doc_lang,
            "ocr_used": len(errors) == 0,
            "mode": "single",
            "pages_delimiter": pages_delimiter,
            "page_count": pdf_pagecounts.get(pdf_path),
        }
        if errors:
            meta["ocr_errors"] = errors  # page-indexed list

        docs_single.append(Document(page_content=joined_text, metadata=meta))

    docs_single.sort(key=lambda d: d.metadata["source"])
    return docs_single


def preprocess_docs(docs: list[Document]) -> list[Document]:
    """Truncate at the last 'References' if present (usually at the end of papers)."""
    for doc in docs:
        content = doc.page_content
        idx = content.lower().rfind("references")
        if idx != -1:
            doc.page_content = content[:idx]
    return docs


def chunk_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    docs = preprocess_docs(docs)

    # Expand single-PDF docs into many window docs
    expanded: list[Document] = []
    for d in docs:
        if d.metadata.get("mode") == "single":
            expanded.extend(make_page_and_pagepair_windows(d))
        else:
            expanded.append(d)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(expanded)

    # Keep/compute your "section" metadata as before
    keywords = ["recommend", "raccomanda"]
    for c in chunks:
        c.metadata["section"] = "Recommendation" if any(w in c.page_content.lower() for w in keywords) else "main"

    return [c for c in chunks if len(c.page_content) > 100]


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


def build_faiss_index_old(cfg: Config) -> None:
    console.rule("[bold]Indexing PDFs -> FAISS")
    logger.info("Indexing PDFs -> FAISS")

    docs = load_pdfs(
        cfg.pdf_dir,
        ocr=getattr(cfg, "ocr", False),
        mode=getattr(cfg, "pdf_mode", "single"),  # "page" or "single"
        pages_delimiter=getattr(cfg, "pages_delimiter", "\n\f\n"),
        ocr_dpi=getattr(cfg, "ocr_dpi", 200),
        ocr_workers=getattr(cfg, "ocr_workers", 2),
        ocr_chunksize=getattr(cfg, "ocr_chunksize", 2),
        tessdata=getattr(cfg, "tessdata", None),
    )

    chunks = chunk_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    console.print(f"Loaded [bold]{len(docs)}[/bold] pages -> [bold]{len(chunks)}[/bold] chunks.")
    logger.info("Loaded %d pages -> %d chunks.", len(docs), len(chunks))

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


def build_faiss_index(cfg: Config) -> None:
    console.rule("[bold]Indexing PDFs -> FAISS (parents+children)")
    logger.info("Indexing PDFs -> FAISS (parents+children)")

    docs = load_pdfs(
        cfg.pdf_dir,
        ocr=getattr(cfg, "ocr", False),
        mode=getattr(cfg, "pdf_mode", "single"),  # "page" or "single"
        pages_delimiter=getattr(cfg, "pages_delimiter", "\n\f\n"),
        ocr_dpi=getattr(cfg, "ocr_dpi", 200),
        ocr_workers=getattr(cfg, "ocr_workers", 2),
        ocr_chunksize=getattr(cfg, "ocr_chunksize", 2),
        tessdata=getattr(cfg, "tessdata", None),
    )

    # --------
    # Build PARENT docs = page + pagepair windows (what you want to return to the LLM)
    # --------
    docs = preprocess_docs(docs)

    parents: list[Document] = []
    for d in docs:
        if d.metadata.get("mode") == "single":
            parents.extend(make_page_and_pagepair_windows(d))
        else:
            # If you ever run in mode="page", treat each page as a parent
            d.metadata.setdefault("page_start", d.metadata.get("page"))
            d.metadata.setdefault("page_end", d.metadata.get("page"))
            d.metadata.setdefault("window", "page")
            parents.append(d)

    # Stable parent_id + persistable parent store
    parents_map: dict[str, Document] = {}
    for p in parents:
        pid = _parent_id(p.metadata)
        p.metadata["parent_id"] = pid
        parents_map[pid] = p

    console.print(f"Built [bold]{len(parents)}[/bold] parent windows (page + pagepair).")
    logger.info("Built %d parent windows.", len(parents))

    # --------
    # Build CHILD chunks = what you embed/index in FAISS
    # --------
    # Use dedicated child chunk params; fallback to your current cfg values if not present
    child_chunk_size = getattr(cfg, "child_chunk_size", cfg.chunk_size)
    child_chunk_overlap = getattr(cfg, "child_chunk_overlap", min(cfg.chunk_overlap, int(child_chunk_size * 0.25)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    children = splitter.split_documents(parents)

    # ensure parent_id is present on children (split_documents should copy metadata, but be safe)
    for c in children:
        if "parent_id" not in c.metadata:
            c.metadata["parent_id"] = _parent_id(c.metadata)

    # Keep/compute your "section" metadata (on children, since that’s what you retrieve/rerank)
    keywords = ["recommend", "raccomanda"]
    for c in children:
        c.metadata["section"] = "Recommendation" if any(w in c.page_content.lower() for w in keywords) else "main"

    children = [c for c in children if len(c.page_content) > 100]

    console.print(
        f"Loaded [bold]{len(docs)}[/bold] docs -> [bold]{len(parents)}[/bold] parents \
        -> [bold]{len(children)}[/bold] child chunks."
    )
    logger.info("Loaded %d docs -> %d parents -> %d child chunks.", len(docs), len(parents), len(children))

    # --------
    # Translation: translate CHILDREN, but keep same parent_id to map back to parent windows
    # --------
    it_children = [c for c in children if c.metadata.get("language") == "it"]
    logger.info("Found %d Italian child chunks to translate.", len(it_children))

    en_or_other_children = [c for c in children if c.metadata.get("language") != "it"]
    if it_children and cfg.translate_pdf:
        en_children = translate_docs_it_to_en(
            it_children,
            src_lang="it",
            tgt_lang="en",
            parallel=cfg.translate_parallel,
            max_workers=cfg.translate_workers,
        )
        # keep mapping stable
        for c in en_children:
            if "parent_id" not in c.metadata:
                c.metadata["parent_id"] = _parent_id(c.metadata)
        children = en_children + en_or_other_children

    # --------
    # Index CHILDREN in FAISS
    # --------
    embedder = build_embedder(cfg.embed_model)
    vs = FAISS.from_documents(children, embedder)

    os.makedirs(cfg.index_dir, exist_ok=True)
    vs.save_local(cfg.index_dir)  # saves FAISS index + its child docstore [web:241][web:316]

    # Save PARENTS separately (FAISS does not know about them)
    parents_path = os.path.join(cfg.index_dir, "parents.pkl")
    _save_parents_map(parents_map, parents_path)

    console.print(f"Saved FAISS child index to [bold]{cfg.index_dir}[/bold]")
    console.print(f"Saved parents map to [bold]{parents_path}[/bold]")
    logger.info("Saved FAISS child index to %s", cfg.index_dir)
    logger.info("Saved parents map to %s", parents_path)


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


def children_to_parents(children: list[Document], parents_map: dict[str, Document], k_parents: int) -> list[Document]:
    out, seen = [], set()
    for ch in children:
        pid = ch.metadata.get("parent_id")
        if not pid or pid in seen:
            continue
        p = parents_map.get(pid)
        if p is None:
            continue
        # optionally carry rerank score onto parent for debugging
        if "rerank_score" in ch.metadata:
            p.metadata["rerank_score"] = ch.metadata["rerank_score"]
        seen.add(pid)
        out.append(p)
        if len(out) >= k_parents:
            break
    return out

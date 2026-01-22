# ------------------------
# Build / load PDFs
# ------------------------


import glob
import os
from concurrent.futures import ProcessPoolExecutor

import faiss
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from tqdm import tqdm

from ._load_env import ONLINE, Config, cfg, console
from .loggers import Logger

logger = Logger.get_logger(__name__)


TESS_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "it": "ita",
}


def _detect_lang_safe(text: str) -> str:
    try:
        if text and len(text) >= 50:
            return detect(text)
    except Exception:
        pass
    return "unknown"


def _tess_lang(lang: str) -> str:
    return TESS_LANG_MAP.get(lang, "eng")


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
        for page_index in range(doc.page_count):
            # cheap normal extraction just to pick OCR language
            normal_text = doc[page_index].get_text("text") or ""
            lang = _detect_lang_safe(normal_text)
            tasks.append((p, page_index, lang, ocr_dpi, tessdata))
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
        joined_parts = [f"\n[PAGE {pi}]\n{txt}" for (pi, _lang, txt, _err) in pages]
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    keywords = ["recommend", "raccomanda"]
    for chunk in chunks:
        if any(word in chunk.page_content.lower() for word in keywords):
            chunk.metadata["section"] = "Recommendation"
        else:
            chunk.metadata["section"] = "main"
    chunks = [c for c in chunks if len(c.page_content) > 200]
    return chunks


# ------------------------
# Build FAISS
# ------------------------
def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    offline = not (ONLINE and getattr(cfg, "online", True))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={
            "trust_remote_code": True,
            "local_files_only": offline,  # << KEY LINE
        },
    )


def build_faiss_index(cfg: Config) -> None:
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

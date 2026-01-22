# ------------------------
# Build / load PDFs
# ------------------------


import glob
import os

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


def load_pdfs(pdf_dir: str, ocr: bool, ocr_dpi: int) -> list[Document]:
    paths = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(os.path.join(pdf_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    if not ocr:
        docs: list[Document] = []
        for p in tqdm(paths, desc="Loading PDFs (no OCR)"):
            loader = PyMuPDFLoader(p, mode="page")
            ds = loader.load()
            for d in ds:
                d.metadata = d.metadata or {}
                d.metadata["source"] = p
                lang = _detect_lang_safe(d.page_content)
                d.metadata["language"] = lang
                d.metadata["ocr_used"] = False
            docs.extend(ds)
        return docs

    # ---- OCR every page (in-memory) using PyMuPDF OCR TextPage ----
    logger.info("OCR enabled: performing OCR on every page.")
    docs: list[Document] = []
    for p in tqdm(paths, desc="Loading PDFs (OCR every page)"):
        pdf = fitz.open(p)
        for page_index in range(pdf.page_count):
            page = pdf[page_index]

            # Use normal extraction only to choose OCR language
            normal_text = page.get_text("text") or ""
            lang = _detect_lang_safe(normal_text)

            # Full-page OCR: build OCR TextPage then extract from it
            tp = page.get_textpage_ocr(language=_tess_lang(lang), dpi=ocr_dpi, full=True)
            text = page.get_text("text", textpage=tp) or ""

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": p,
                        "page": page_index,
                        "language": lang,
                        "ocr_used": True,
                    },
                )
            )
        pdf.close()
    return docs


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
    docs = load_pdfs(cfg.pdf_dir, cfg.ocr, cfg.ocr_dpi)
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

from __future__ import annotations

import glob
import os

# from . import _load_env as _  # noqa: F401
# from ._load_env import Config, cfg
import _load_env as _  # noqa: F401
from _load_env import Config, cfg

import faiss  # noqa: F401
import torch

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders.base import BaseCrossEncoder

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langdetect import detect
from rich import print
from rich.console import Console
from rich.table import Table as RichTable
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ------------------------
# Console and device
# ------------------------
console = Console()

USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# CrossEncoder wrapper for MPS
# ------------------------
class MPSSentenceCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name: str):
        self.device = DEVICE
        self.model = CrossEncoder(model_name, device=self.device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


# ------------------------
# Build / load PDFs
# ------------------------
def load_pdfs(pdf_dir: str) -> list[Document]:
    paths = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(os.path.join(pdf_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    docs: list[Document] = []
    for p in tqdm(paths, desc="Loading PDFs"):
        loader = PyMuPDFLoader(p)
        ds = loader.load()
        for d in ds:
            d.metadata = d.metadata or {}
            d.metadata["source"] = p
            try:
                lang = detect(d.page_content)
            except Exception:
                lang = "unknown"
            d.metadata["language"] = lang
        docs.extend(ds)
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
    docs = preprocess_docs(docs)  # Truncate at "References"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Tag sections
    keywords = ["recommend", "raccomanda"]
    for chunk in chunks:
        if any(word in chunk.page_content.lower() for word in keywords):
            chunk.metadata["section"] = "Recommendation"
        else:
            chunk.metadata["section"] = "main"
    # Filter out short chunks
    chunks = [c for c in chunks if len(c.page_content) > 200]
    return chunks


# ------------------------
# Build FAISS
# ------------------------


def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"trust_remote_code": True},
    )


def build_faiss_index(cfg: Config) -> None:
    console.rule("[bold]Indexing PDFs -> FAISS")
    docs = load_pdfs(cfg.pdf_dir)
    chunks = chunk_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    console.print(f"Loaded [bold]{len(docs)}[/bold] pages -> [bold]{len(chunks)}[/bold] chunks.")

    embedder = build_embedder(cfg.embed_model)
    vs = FAISS.from_documents(chunks, embedder)

    os.makedirs(cfg.index_dir, exist_ok=True)
    vs.save_local(cfg.index_dir)
    console.print(f"Saved FAISS index to [bold]{cfg.index_dir}[/bold]")


def load_vectorstore(index_dir: str, embed_model: str) -> FAISS:
    embedder = build_embedder(embed_model)
    vs = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    if getattr(cfg, "use_gpu_index", False):
        try:
            if faiss.get_num_gpus() > 0:
                console.print(
                    f"[green]FAISS GPU detected: {faiss.get_num_gpus()} GPU(s).Moving loaded index to GPU...[/green]"
                )
                res = faiss.StandardGpuResources()
                res.setTempMemory(128 * 1024 * 1024)  # 128 MB scratch space
                vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
            else:
                console.print("[yellow]No GPU detected by FAISS. Using CPU index.[/yellow]")
        except ImportError:
            console.print("[red]FAISS GPU not available. Using CPU index.[/red]")

    return vs


# ------------------------
# Retriever
# ------------------------
def build_retriever(vs: FAISS, k: int, rerank_model: str | None, k_reranked: int):
    base_retriever = vs.as_retriever(search_kwargs={"k": k})
    if rerank_model:
        console.print(f"Using cross-encoder reranker ({DEVICE}): [bold]{rerank_model}[/bold]")
        cross_encoder = MPSSentenceCrossEncoder(rerank_model)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=k_reranked)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        return retriever
    else:
        return base_retriever


# ------------------------
# LLM pipeline
# ------------------------
def build_llm_pipe(model_name: str, max_new_tokens: int, temperature: float) -> HuggingFacePipeline:
    """
    Build a HuggingFace LLM pipeline.
    """
    console.print(f"Loading LLM: [bold]{model_name}[/bold] on device [bold]{DEVICE}[/bold]")

    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=os.environ.get("HF_TOKEN"),
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
    )

    if DEVICE == "mps":
        model.to("mps")

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen)


# ------------------------
# Prompt
# ------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise research assistant. Answer the user's question using only the provided context. "
            "If the answer isn't in the context, say you don't know. Cite sources as (source.pdf p. N). "
            "Prefer bullet points for lists; be concise and avoid speculation.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    # Prioritize Recommendation sections. This happens after the retrieved part
    recs = [d for d in docs if d.metadata.get("section") == "Recommendation"]
    others = [d for d in docs if d.metadata.get("section") != "Recommendation"]
    ordered = recs + others
    parts = []
    for d in ordered:
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page = d.metadata.get("page", "?")
        parts.append(f"[source: {src} p.{page}]\n{d.page_content}")
    return "\n\n".join(parts)


def print_sources(docs: list[Document]):
    table = RichTable(title="Top Context Chunks")
    table.add_column("#")
    table.add_column("Source")
    table.add_column("Page")
    table.add_column("Chars")
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown.pdf"))
        page = str(d.metadata.get("page", "?"))
        table.add_row(str(i), src, page, str(len(d.page_content)))
    console.print(table)


# ------------------------
# Interactive loop
# ------------------------
def interactive_loop(cfg: Config):
    vs = load_vectorstore(cfg.index_dir, cfg.embed_model)
    retriever = build_retriever(vs, cfg.k, cfg.rerank_model if cfg.rerank else None, cfg.k_reranked)
    llm = build_llm_pipe(cfg.llm_model, cfg.max_new_tokens, cfg.temperature)

    console.print("[bold green]Interactive RAG. Type 'exit' to quit.[/bold green]")

    while True:
        try:
            question = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        docs = retriever.invoke(question)
        print_sources(docs)
        ctx = format_docs(docs)
        prompt = RAG_PROMPT.format_messages(question=question, context=ctx)
        answer = llm.invoke(prompt)
        console.print(f"\n[bold]Answer[/bold]:\n{answer}")


# ------------------------
# Main
# ------------------------


def main():
    console.print(f"Using HF cache dir: [bold]{cfg.hf_home}[/bold]")

    # Rebuild FAISS index if requested
    if getattr(cfg, "reindex", False):
        build_faiss_index(cfg)

    # Ensure FAISS index exists
    if not os.path.isdir(cfg.index_dir) or not os.listdir(cfg.index_dir):
        console.print(
            f"[red]FAISS index not found or empty at {cfg.index_dir}. "
            "Set 'reindex: true' in config.yaml to build it.[/red]"
        )

    # Decide whether to run interactive chat or single query
    if getattr(cfg, "chat", False):
        interactive_loop(cfg)
    elif getattr(cfg, "web", None):
        pass
    else:
        console.print("[yellow]Nothing to do. Set 'chat: true' or provide 'web: true' in config.yaml.[/yellow]")


if __name__ == "__main__":
    main()


# implement the context window seguendo l'approccio di nvidia
# compare faiss with SKLearnVectorStore, InMemoryVectorStore and others
# what are good values for chunk size and overlap?

# Inserisci il prompt per dire di rispondere in italiano
# controlla github nvidia rag

# https://developer.nvidia.com/blog/tips-for-building-a-rag-pipeline-with-nvidia-ai-langchain-ai-endpoints/

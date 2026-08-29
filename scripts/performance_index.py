import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import psutil
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

console = Console()
INDEX_FILENAMES = ("index.faiss", "index.pkl")
INDEX_MANIFEST = "benchmark.json"

# These standalone benchmark settings are intentionally independent of config.yaml.
# "BAAI/bge-multilingual-gemma2"  # "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-MiniLM-L6-v2"

REPEAT = 20
INDEX_DIR = "performance_faiss_index"
MARKDOWN_DIR = "Markdown_IT"
EMBED_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def index_settings(embed_model: str, chunk_size: int, chunk_overlap: int) -> dict[str, int | str]:
    return {
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def index_is_compatible(index_dir: Path, settings: dict[str, int | str]) -> bool:
    """Return whether a complete index was built with the requested settings."""
    if not index_dir.is_dir() or any(not (index_dir / name).is_file() for name in INDEX_FILENAMES):
        return False

    try:
        manifest = json.loads((index_dir / INDEX_MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError, OSError, json.JSONDecodeError:
        return False
    return manifest == settings


def load_markdown_chunks(markdown_dir: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load top-level Markdown files and split them for the benchmark index."""
    markdown_path = Path(markdown_dir).expanduser().resolve()
    md_files = sorted(markdown_path.glob("*.md")) if markdown_path.is_dir() else []
    if not md_files:
        raise FileNotFoundError(f"No top-level Markdown files found in {markdown_path}.")

    documents = [
        Document(
            page_content=md_file.read_text(encoding="utf-8"),
            metadata={"source": str(md_file)},
        )
        for md_file in md_files
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No chunks were created from Markdown files in {markdown_path}.")

    console.print(f"Loaded [bold]{len(md_files)}[/bold] Markdown files into [bold]{len(chunks)}[/bold] chunks.")
    return chunks


def get_or_create_vectorstore(
    index_dir: str,
    markdown_dir: str,
    embed_model: str,
    embedder: HuggingFaceEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
) -> FAISS:
    """Load a compatible benchmark index or build an independent one."""
    index_path = Path(index_dir).expanduser().resolve()
    settings = index_settings(embed_model, chunk_size, chunk_overlap)
    if index_is_compatible(index_path, settings):
        console.print(f"[cyan]Loading benchmark index:[/cyan] {index_path}")
        return FAISS.load_local(str(index_path), embedder, allow_dangerous_deserialization=True)

    if index_path.exists() and not index_path.is_dir():
        raise NotADirectoryError(f"Benchmark index path is not a directory: {index_path}")

    console.print(f"[yellow]Creating benchmark index:[/yellow] {index_path}")
    chunks = load_markdown_chunks(markdown_dir, chunk_size, chunk_overlap)
    vectorstore = FAISS.from_documents(chunks, embedder)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    (index_path / INDEX_MANIFEST).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]Saved benchmark index:[/green] {index_path}")
    return vectorstore


def embedding_model_location(embed_model: str, cache_folder: str | None = None) -> Path:
    """Resolve a local model path or its downloaded Hugging Face snapshot."""
    local_path = Path(embed_model).expanduser()
    if local_path.exists():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=embed_model,
            cache_dir=cache_folder,
            local_files_only=True,
        )
    ).resolve()


def cuda_available() -> bool:
    return torch.cuda.is_available()


def embedding_device() -> str:
    if cuda_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def gpu_mem_info(device: int = 0) -> tuple[float, float]:
    """Return used and total accelerator memory in MB.

    MPS reports this process's driver allocation and recommended working-set
    limit because Apple GPUs use unified memory rather than dedicated VRAM.
    """
    if cuda_available():
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024**2, info.total / 1024**2
        finally:
            pynvml.nvmlShutdown()

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024**2, torch.mps.recommended_max_memory() / 1024**2

    return 0.0, 0.0


def faiss_gpu_available() -> bool:
    """Return whether FAISS exposes an accelerator, including Darwin Metal."""
    return faiss.get_num_gpus() > 0


def benchmark_faiss(vs, queries: list[str], top_k: int = 5, n_repeat: int = 5):
    """Benchmark retrieval performance of a FAISS retriever."""
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    timings = []

    for _ in range(n_repeat):
        for q in queries:
            start = time.perf_counter()
            _ = retriever.invoke(q)
            end = time.perf_counter()
            timings.append(end - start)

    timings = np.array(timings) * 1000  # ms
    return {
        "mean_ms": timings.mean(),
        "p95_ms": np.percentile(timings, 95),
        "min_ms": timings.min(),
        "max_ms": timings.max(),
    }


def load_vectorstore(
    index_dir: str,
    markdown_dir: str,
    embed_model: str,
    device: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS:
    """Create or load the standalone FAISS index and report memory usage."""
    device = device or embedding_device()
    embedder = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": device, "trust_remote_code": True},
    )
    model_path = embedding_model_location(embed_model, embedder.cache_folder)
    console.print(f"[cyan]Embedding model location:[/cyan] {model_path}")

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024**2
    gpu_before, gpu_total = gpu_mem_info()

    vs = get_or_create_vectorstore(
        index_dir,
        markdown_dir,
        embed_model,
        embedder,
        chunk_size,
        chunk_overlap,
    )

    ram_after = process.memory_info().rss / 1024**2
    gpu_after, _ = gpu_mem_info()

    console.print(f"[cyan]RAM used by index:[/cyan] {ram_after - ram_before:.2f} MB")
    total_mem = psutil.virtual_memory().total / 1024**4
    console.print(f"[cyan]Total system RAM:[/cyan] {total_mem:.2f} TB")

    console.print(f"[cyan]GPU used before:[/cyan] {gpu_before:.2f} MB / {gpu_total:.2f} MB")
    console.print(f"[cyan]GPU used after load:[/cyan] {gpu_after:.2f} MB")

    test_emb = embedder.embed_query("test query")
    console.print("[yellow]Embedding model output dimension:[/yellow]", len(test_emb))

    # Dimension of embeddings
    dim = vs.index.d
    console.print("[yellow]Embedding dimension:[/yellow]", dim)

    assert dim == len(test_emb), f"Dimension mismatch: index {dim} vs model {len(test_emb)}"

    # Number of vectors
    ntotal = vs.index.ntotal
    console.print(f"Number of vectors: {ntotal}")

    # FAISS usually stores float32 (4 bytes per value) on CPU
    bytes_per_vector = dim * 4
    console.print(f"Bytes per vector: {bytes_per_vector} B")

    # Total memory if stored as float32
    total_bytes = ntotal * bytes_per_vector
    console.print(f"Expected size: {total_bytes / 1024**2:.2f} MB")

    xb = vs.index.reconstruct_n(0, 1)  # reconstruct the first vector
    console.print(f"Reconstructed vector shape: {xb.shape}, dtype: {xb.dtype}")

    return vs


def main():
    device = embedding_device()
    console.print(f"Using embedding device: {device}")

    queries = [
        "What is quantum entanglement?",
        "Explain gradient descent.",
        "What is the efficient frontier in finance?",
        "What is colon-TC?",
        "Explain the risks of colon-TC.",
        "What are the symptoms of diabetes?",
        "Can I do the exam online?",
        "What is the size of polyps?",
        "Can I eat before the exam?",
        "What is the preparation for colon-TC?",
    ]

    vs = load_vectorstore(
        INDEX_DIR,
        MARKDOWN_DIR,
        EMBED_MODEL,
        device=device,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    console.rule("[bold]Benchmark: CPU Index[/bold]")
    stats_cpu = benchmark_faiss(vs, queries, n_repeat=REPEAT)
    console.print(stats_cpu)

    if faiss_gpu_available():
        console.rule("[bold]Benchmark: FAISS Accelerator Index[/bold]")

        gpu_before_move, _ = gpu_mem_info()
        res = faiss.StandardGpuResources()
        if hasattr(res, "setTempMemory"):
            res.setTempMemory(128 * 1024 * 1024)  # 128 MB scratch space
        vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)
        gpu_after_move, _ = gpu_mem_info()
        if device != "cpu":
            console.print(f"[cyan]GPU increase after FAISS move:[/cyan] {gpu_after_move - gpu_before_move:.2f} MB")

        stats_gpu = benchmark_faiss(vs, queries, n_repeat=REPEAT)
        console.print(stats_gpu)
    else:
        console.print("[yellow]No FAISS accelerator available. Using CPU index.[/yellow]")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from None

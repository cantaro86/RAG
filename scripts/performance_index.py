import os
import time

import faiss
import numpy as np
import psutil
import pynvml
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

import src.agentic_rag._load_env  # noqa: F401

console = Console()

USE_MPS = torch.backends.mps.is_available()
USE_CUDA = torch.cuda.is_available()
DEVICE = "mps" if USE_MPS else ("cuda" if USE_CUDA else "cpu")
console.print(f"Using device: {DEVICE}")


def gpu_mem_info(device: int = 0):
    """Return used and total GPU memory in MB (device 0)."""
    if not (USE_CUDA or USE_MPS):
        return 0.0, 0.0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2, info.total / 1024**2


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


def load_vectorstore(index_dir: str, embed_model: str) -> FAISS:
    """Load FAISS index (CPU or GPU) and report memory usage."""
    embedder = HuggingFaceEmbeddings(
        model_name=embed_model, encode_kwargs={"normalize_embeddings": True}, model_kwargs={"trust_remote_code": True}
    )

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1024**2
    gpu_before, gpu_total = gpu_mem_info()

    vs = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    ram_after = process.memory_info().rss / 1024**2
    gpu_after, _ = gpu_mem_info()

    console.print(f"[cyan]RAM used by index:[/cyan] {ram_after - ram_before:.2f} MB")
    total_mem = psutil.virtual_memory().total / 1024**4
    console.print(f"[cyan]Total DGX memory on RAM:[/cyan] {total_mem:.2f} TB")  # free -m

    console.print(f"[cyan]GPU used before:[/cyan] {gpu_before:.2f} MB / {gpu_total:.2f} MB")
    console.print(f"[cyan]GPU used after load:[/cyan] {gpu_after:.2f} MB")

    test_emb = embedder.embed_query("test query")
    console.print("[yellow]Embedding model output dimension:[/yellow]", len(test_emb))

    # Dimension of embeddings
    dim = vs.index.d
    console.print("[yellow]Embedding dimension:[/yellow]", {dim})

    assert dim == len(test_emb), f"Dimension mismatch: index {dim} vs model {len(test_emb)}"

    # Number of vectors
    ntotal = vs.index.ntotal
    print(f"Number of vectors: {ntotal}")

    # FAISS usually stores float32 (4 bytes per value) on CPU
    bytes_per_vector = dim * 4
    print(f"Bytes per vector: {bytes_per_vector} B")

    # Total memory if stogrey as float32
    total_bytes = ntotal * bytes_per_vector
    print(f"Expected size: {total_bytes / 1024**2:.2f} MB")

    xb = vs.index.reconstruct_n(0, 1)  # reconstruct the first vector
    print(type(xb), xb.shape, xb.dtype)

    if faiss.get_num_gpus() > 0 and (USE_CUDA or USE_MPS):
        console.print("[green]Moving index to GPU...[/green]")
        res = faiss.StandardGpuResources()
        res.setTempMemory(128 * 1024 * 1024)  # 128 MB scratch space
        _ = faiss.index_cpu_to_gpu(res, 0, vs.index)
        gpu_after_move, _ = gpu_mem_info()
        console.print(f"[cyan]GPU increase after FAISS move:[/cyan] {gpu_after_move - gpu_after:.2f} MB")
    else:
        console.print("[yellow]No GPU available for FAISS. Using CPU index.[/yellow]")

    return vs


def main():
    # Hardcode your settings or read from config.yaml
    REPEAT = 20
    index_dir = "faiss_index"
    embed_model = "intfloat/multilingual-e5-base"
    # "BAAI/bge-multilingual-gemma2"  # "BAAI/bge-small-en-v1.5" # "sentence-transformers/all-MiniLM-L6-v2"

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

    vs = load_vectorstore(index_dir, embed_model)
    console.rule("[bold]Benchmark: CPU Index[/bold]")
    stats_cpu = benchmark_faiss(vs, queries, n_repeat=REPEAT)
    console.print(stats_cpu)

    if faiss.get_num_gpus() > 0 and (USE_CUDA or USE_MPS):
        console.rule("[bold]Benchmark: GPU Index[/bold]")

        res = faiss.StandardGpuResources()
        res.setTempMemory(128 * 1024 * 1024)  # 128 MB scratch space
        vs.index = faiss.index_cpu_to_gpu(res, 0, vs.index)

        stats_gpu = benchmark_faiss(vs, queries, n_repeat=REPEAT)
        console.print(stats_gpu)


if __name__ == "__main__":
    main()

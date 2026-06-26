from spack.package import *


class PyAgenticRag(PythonPackage, CudaPackage):
    """Medical Agentic RAG retrieval and LLM pipeline using LangChain."""

    homepage = "https://github.com/cantaro86/RAG"
    git = "https://github.com/cantaro86/RAG.git"

    version("0.1.0", tag="v0.1.0", commit="ffa56411125cdfd0482e4fc772db3e36b30bd172", preferred=True)
    version("main", branch="main")

    # build_system("pyproject")
    build_system("python_pip")

    # ── compiler ────────────────────────────────────────────────────────────
    # requires("%gcc@14:", msg="agentic_rag must be built with GCC 14 or later")

    conflicts(
        "cuda_arch=none", when="+cuda platform=linux", msg="CUDA architecture is required, e.g. cuda_arch=90 for H100"
    )

    # ── build deps ──────────────────────────────────────────────────────────
    depends_on("python@3.12:", type=("build", "run"))
    depends_on("py-setuptools@65:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-setuptools-scm@8:", type="build")

    # ── runtime deps ────────────────────────────────────────────────────────
    # depends_on("py-accelerate",                         type=("build", "run"))
    depends_on("faiss+python", type=("build", "run"))
    depends_on("py-fasttext-numpy2", type=("build", "run"))
    # depends_on("py-gradio",                             type=("build", "run"))
    depends_on("py-langchain-core", type=("build", "run"))
    # depends_on("py-langchain@1.2.10",                   type=("build", "run"))
    # depends_on("py-langchain-community@0.4.1",          type=("build", "run"))
    # depends_on("py-langchain-core@1.2.13",              type=("build", "run"))
    # depends_on("py-langchain-huggingface@1.2.0",        type=("build", "run"))
    # depends_on("py-langchain-text-splitters@1.1.0",     type=("build", "run"))
    # depends_on("py-langdetect",                         type=("build", "run"))
    depends_on("py-langgraph", type=("build", "run"))
    depends_on("py-numpy@2.4.0:", type=("build", "run"))
    depends_on("py-openpyxl", type=("build", "run"))
    depends_on("py-pyyaml", type=("build", "run"))
    depends_on("py-rich", type=("build", "run"))
    depends_on("py-sacremoses", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    # depends_on("py-sentence-transformers",              type=("build", "run"))
    depends_on("py-tqdm", type=("build", "run"))
    depends_on("py-transformers@4.57.0:", type=("build", "run"))

    # ── CUDA-backed packages (Linux only, CUDA 12.8) ─────────────────────

    depends_on("cuda@12.8", when="+cuda platform=linux", type=("build", "run"))
    # depends_on("py-bitsandbytes", when="+cuda platform=linux", type=("build", "run"))
    depends_on("py-torch@2.9.0:+cuda cuda_arch=90", when="+cuda platform=linux", type=("build", "run"))
    depends_on("py-torch@2.9.0:~cuda", when="~cuda platform=linux", type=("build", "run"))
    depends_on("py-torch@2.9.0:~cuda", when="platform=darwin", type=("build", "run"))

    conflicts("+cuda", when="platform=darwin", msg="CUDA is supported only on Linux")

    # ── Apple Silicon (macOS arm64 only) ─────────────────────────────────
    depends_on("py-mlx", when="platform=darwin target=aarch64:", type=("build", "run"))
    depends_on("py-mlx-lm", when="platform=darwin target=aarch64:", type=("build", "run"))

    # ── dev variant ──────────────────────────────────────────────────────
    variant("dev", default=False, description="Install development dependencies")
    depends_on("py-ruff", when="+dev", type=("build", "run"))
    depends_on("py-coverage", when="+dev", type=("build", "run"))
    depends_on("py-pre-commit", when="+dev", type=("build", "run"))
    depends_on("py-pytest", when="+dev", type=("build", "run"))
    depends_on("py-jupyter", when="+dev", type=("build", "run"))
    depends_on("py-debugpy", when="+dev", type=("build", "run"))
    depends_on("py-pynvml", when="+dev", type=("build", "run"))

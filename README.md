# RAG

[![Build](https://github.com/cantaro86/RAG/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/cantaro86/RAG/actions/workflows/build.yml)
[![Tests](https://github.com/cantaro86/RAG/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cantaro86/RAG/actions/workflows/test.yml)
[![Pre-commit](https://github.com/cantaro86/RAG/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/cantaro86/RAG/actions/workflows/pre-commit.yml)
[![Top language](https://img.shields.io/github/languages/top/cantaro86/RAG)](https://github.com/cantaro86/RAG)
[![Last commit](https://img.shields.io/github/last-commit/cantaro86/RAG)](https://github.com/cantaro86/RAG/commits/main)

[![Maintainer](https://img.shields.io/badge/maintainer-%40cantaro86-blue)](https://github.com/cantaro86)
[![Release](https://img.shields.io/github/v/release/cantaro86/RAG)](https://github.com/cantaro86/RAG/releases)
[![GitHub tag](https://img.shields.io/github/v/tag/cantaro86/RAG)](https://github.com/cantaro86/RAG/tags)





### HF TOKEN

Create a file called `.env` in the root path of the project with inside:
```
HF_TOKEN="hf_your_hugging_face_token"
```


### Installation (recommended)

Alternatively, you can use [uv](https://docs.astral.sh/uv/) to manage dependencies and virtual environments:

```bash
module load python3.14
module load uv
uv sync --locked
# or, for development
uv sync --locked --extra dev
```

If you want to specify the Python version and the environment name (Not recommended):
```bash
# Create venv with specific Python version
uv venv --prompt RAG ./uv-venv --python 3.14
source ./uv-venv/bin/activate
uv run --active which python
uv sync --active --locked --extra dev
# or
uv pip install -e .
```


### Tests and coverage

Run the CPU test suite from the repository root:

```bash
uv run pytest -m cpu
```

To run a specific test file, append its path, for example:

```bash
uv run pytest -m cpu tests/test_guardrail.py
```

Run the tests with branch coverage and display the source-only report:

```bash
uv run coverage run -m pytest -m cpu
uv run coverage report
```

Coverage is configured in `pyproject.toml` for `src/agentic_rag` and must remain at or above 85%.


### Run the program
Cluster allocation:
```bash
salloc --job-name="rag" --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --gpus-per-node=1 --time=01:45:00 --qos=normal
```

Run it inside the SLURM interactive allocation with
```bash
uv run agentic_rag
```
Or, if the virtual environment is active:
```bash
agentic_rag
```

As a SBATCH script with a gradio web interface:
```bash
sbatch rag_gradio.sbatch
```

Do not run this:
```bash
python src/agentic_rag/cli.py
```

If you do not want to install the package, you can run the program with:

```bash
PYTHONPATH=src python -m agentic_rag
```


### Pip compatibility

Virtual environment:
```bash
module load python3.14
python -m venv --prompt RAG ./python-venv
source ./python-venv/bin/activate
```

Install the locked export, which includes this project and the development dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, let pip resolve dependencies directly from `pyproject.toml`:

```bash
python -m pip install -e .
# or for developers
python -m pip install -e ".[dev]"
```

`requirements.txt` is generated from `uv.lock` for Linux x86_64 and macOS 14+ ARM64. Hashes are omitted so pip can install the editable project entry. Regenerate it only after changing the lockfile:

```bash
uv export --locked --extra dev --no-hashes --output-file requirements.txt
```


### Conda compatibility

`environment.yml` is a platform-neutral Conda-forge bootstrap for those targets. It installs Python 3.14 and `uv`; `uv` then creates the same locked project environment used by the recommended installation:

```bash
conda env create -f environment.yml
conda activate RAG
uv sync --locked --extra dev
```





### Spack (work in progress)

module load spack
spacktivate RAG
spack install py-agentic-rag

```bash
spack repo add ./spack-repo

spack install --add py-agentic-rag +cuda cuda_arch=90

spack spec py-torch+cuda cuda_arch=90 %gcc@14.2.0 ^cuda@12.8

spack spec py-agentic-rag +cuda cuda_arch=90
```



#### Debugging

```bash
module load conda
salloc --job-name="rag" --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --time=08:45:00 --nodelist=dgx01 --qos=mira
conda activate RAG
python -m debugpy --listen 0.0.0.0:5643 --wait-for-client -m agentic_rag
```

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

In another terminal

```bash
ssh -N -L 5643:dgx01:5643 dgx01
```


![AI AGENT](graph.png)




We use faiss-cpu, but if we really want faiss gpu we can:
conda install -c pytorch -c nvidia faiss-gpu=1.8.0  # H100 compatible
This one installs numpy-base which is a numpy version 1.26.4 of conda. This may create conflicts.



##### OLD TRANSLATION VERSION
check the branch inglese

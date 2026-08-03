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

uv sync
# or
uv sync --group dev
```

If you want to specify the Python version and the environment name (Not recommended):
```bash
# Create venv with specific Python version
uv venv --prompt RAG ./uv-venv --python 3.14
source ./uv-venv/bin/activate
uv run --active which python
uv run --active sync
# or
uv pip install -e .
```


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


### Legacy installation instructions:

Virtual environment:
```bash
module load python3.14
python -m venv --prompt RAG ./python-venv
source ./python-venv/bin/activate
```

Install the package:
```bash
python -m pip install --upgrade pip

python -m pip install -e .
# or for developers
python -m pip install -e ".[dev]"
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
python -m debugpy --listen 0.0.0.0:5643 --wait-for-client main.py
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



pip install -r requirements.txt --no-cache



##### OLD TRANSLATION VERSION
check the branch inglese

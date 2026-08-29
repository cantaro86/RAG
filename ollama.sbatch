#!/bin/bash
#SBATCH --job-name=ollama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=08:30:00
#SBATCH --output=ollama-%j.out
#SBATCH --error=ollama-%j.err

set -euo pipefail

module load ollama

PORT="${PORT:-11432}"
MODEL_NAME="${MODEL_NAME:-qwen3-coder:30b}"
NODE="$(hostname -s)"

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_MODELS="$HOME/.ollama/models"

ollama serve >"ollama-${SLURM_JOB_ID:-manual}.log" 2>&1 &
OLLAMA_PID=$!

READY=false
for ((attempt = 0; attempt < 60; attempt++)); do
    if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        echo "Error: Ollama failed to start" >&2
        wait "${OLLAMA_PID}"
        exit 1
    fi

    if ollama list >/dev/null 2>&1 && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        READY=true
        break
    fi
    sleep 1
done

if [[ "${READY}" != true ]]; then
    echo "Error: Ollama did not become ready within 60 seconds" >&2
    exit 1
fi

if ! ollama run "${MODEL_NAME}" "warm up" >/dev/null; then
    echo "Error: failed to warm up Ollama model ${MODEL_NAME}" >&2
    exit 1
fi

if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    echo "Error: Ollama stopped during model warm-up" >&2
    wait "${OLLAMA_PID}"
    exit 1
fi

echo "To connect to this Ollama instance, run:"
echo "ssh -N -L ${PORT}:127.0.0.1:${PORT} ${NODE}"

wait "${OLLAMA_PID}"

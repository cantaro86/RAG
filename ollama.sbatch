#!/bin/bash
#SBATCH --job-name=ollama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=08:30:00
#SBATCH --output=ollama-%j.out
#SBATCH --error=ollama-%j.err

module load ollama

# Configuration variables - no defaults for easy modification
PORT=11432
MODEL_NAME=qwen3-coder:30b
HOSTNAME=$(hostname -s)

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MODELS="$HOME/.ollama/models"

ollama serve > ollama-${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

sleep 5

# Validate Ollama is running
if ! ps -p $OLLAMA_PID > /dev/null; then
    echo "Error: Ollama failed to start"
    exit 1
fi

ollama run ${MODEL_NAME} "warm up" >/dev/null 2>&1 || true

# Print SSH tunnel information
echo "To connect to this Ollama instance, run:"
echo "ssh -N -L ${PORT}:127.0.0.1:${PORT} ${HOSTNAME}"

wait $OLLAMA_PID

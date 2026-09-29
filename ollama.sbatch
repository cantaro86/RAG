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
# export OLLAMA_MODELS="$HOME/.ollama/models"

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

cat <<EOF
Ollama is ready on compute node ${NODE}.

To use Ollama interactively from the login node:

1. In a terminal on the login node, start this SSH tunnel:
   ssh -N -L ${PORT}:127.0.0.1:${PORT} ${NODE}

   Keep this terminal open. No output is expected while the tunnel runs.

2. Open another terminal on the SAME login node and run:
   module load ollama
   export OLLAMA_HOST=127.0.0.1:${PORT}
   ollama list
   ollama run ${MODEL_NAME}

   'ollama list' shows the models available on the compute node.
   'ollama run' opens an interactive prompt; inference runs on the compute node.
   Type /bye to exit the interactive prompt.

The tunnel forwards the login node's port ${PORT} to Ollama on ${NODE}.
Closing the prompt or tunnel does not stop the Slurm job.
To stop the job when finished, run:
   scancel ${SLURM_JOB_ID:-JOBID}

Alternatively you can run Ollama commands directly on the compute node using:
   sjoin ${SLURM_JOB_ID:-JOBID}
   module load ollama
   export OLLAMA_HOST=127.0.0.1:${PORT}
EOF

wait "${OLLAMA_PID}"

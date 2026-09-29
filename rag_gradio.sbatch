#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --qos=notimelimit
#SBATCH --output=rag.out
#SBATCH --error=rag.err
#SBATCH --nodelist=dgx02

set -euo pipefail

module load python3.14
module load uv

PORT=7860
NODE="$(hostname -s)"
USER="$(whoami)"

echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${NODE}"
echo "Port:  ${PORT}"
echo
echo "1) On your laptop run:"
echo "   ssh -N -L 7861:127.0.0.1:${PORT} ${USER}@158.110.146.245"
echo
echo "2) In another terminal on the head node run:"
echo "   ssh -N -L ${PORT}:127.0.0.1:${PORT} ${NODE}"
echo
echo "3) Open on laptop: http://localhost:7861"
echo

JOB_SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}')"
SCRIPT_DIR="$(dirname -- "${JOB_SCRIPT}")"
cd -- "${SCRIPT_DIR}"

export AGENTIC_RAG_MODE="gradio"
export GRADIO_SERVER_NAME="127.0.0.1"
export GRADIO_SERVER_PORT="${PORT}"

# Fail before startup if another process already uses the Gradio port.
python "${SCRIPT_DIR}/scripts/cloudflare_wait.py" "${PORT}" --check-port-free

uv run --locked agentic_rag &
APP_PID=$!
# Wait for Gradio readiness, then start and manage the mandatory Cloudflare tunnel.
python "${SCRIPT_DIR}/scripts/cloudflare_wait.py" "${PORT}" --app-pid "${APP_PID}" &
TUNNEL_PID=$!

trap 'kill "${TUNNEL_PID}" "${APP_PID}" 2>/dev/null || true' EXIT INT TERM
wait -n "${APP_PID}" "${TUNNEL_PID}"

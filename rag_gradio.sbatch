#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --time=08:45:00
#SBATCH --qos=mira
#SBATCH --output=rag.out
#SBATCH --error=rag.err

set -euo pipefail

PORT=7860
NODE="$(hostname -s)"
USER=$(whoami)

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

# Recommended: don't expose Gradio beyond the compute node
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=${PORT}

python -u main.py

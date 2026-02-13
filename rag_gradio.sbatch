#!/bin/bash
#SBATCH --job-name=rag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --time=2-01:00:00
#SBATCH --qos=notimelimit
#SBATCH --output=rag.out
#SBATCH --error=rag.err

module load conda
eval "$(conda shell.bash hook)"
conda activate RAG


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

# The -u flag stands for "unbuffered"
python -u main.py

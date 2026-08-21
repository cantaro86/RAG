#!/bin/bash
#SBATCH --job-name=ragas-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --time=0-05:00:00
#SBATCH --output=ragas-%j.out
#SBATCH --error=ragas-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load python3.14
module load uv

JOB_SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}')"
PROJECT_DIR="$(dirname -- "${JOB_SCRIPT}")"
cd -- "${PROJECT_DIR}"

RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"
RESULTS_DIR="${PROJECT_DIR}/evaluation/results"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

printf '\n===== VALIDATE =====\n'
uv run --locked python -m evaluation validate

printf '\n===== COLLECT =====\n'
uv run --locked python -m evaluation collect \
    --results-dir "${RESULTS_DIR}" \
    --run-id "${RUN_ID}" \
    "${LIMIT_ARGS[@]}"

printf '\n===== SCORE =====\n'
uv run --locked --extra evaluation python -m evaluation score \
    "${RESULTS_DIR}/${RUN_ID}" \
    "${LIMIT_ARGS[@]}"

printf '\n===== EVALUATION COMPLETE =====\n'

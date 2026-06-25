#!/usr/bin/env bash
set -euo pipefail

module load spack
module load gcc/14.2.0

ENV_NAME="RAG"
SPEC="py-agentic-rag"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Register the embedded Spack repo if not already known
spack repo list | awk 'NR>1{print $2}' | grep -Fxq "$REPO_ROOT/spack-repo" \
    || spack repo add "$REPO_ROOT/spack-repo"

# Create the environment if it does not exist, then activate it
spack env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" \
    || spack env create "$ENV_NAME"
. <(spack env activate --sh "$ENV_NAME")

# Add the spec to the environment if not yet an explicit root
spack find --explicit --format '{name}' | grep -Fxq "$SPEC" \
    || spack add "$SPEC"

spack install

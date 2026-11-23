#!/usr/bin/env bash
# Sourced inside every allocation. Caller supplies --nv for GPU commands.
set -euo pipefail
source "${NEUROMRI_HPC_ENV:-slurm/hpc.env}"
if [[ -f slurm/hpc.local.env ]]; then source slurm/hpc.local.env; fi
# Node-local copy of the dataset (see stage_data.sh); falls back to DATA_ROOT on shared storage.
if [[ -n "${DATA_TAR:-}" && -n "${SLURM_JOB_ID:-}" ]]; then source slurm/stage_data.sh; fi
export NEUROMRI_DATA="$DATA_ROOT"
export NEUROMRI_WEIGHTS="$WEIGHTS_ROOT"
export NEUROMRI_CONTAINER_SHA256
NEUROMRI_CONTAINER_SHA256=$(sha256sum "$SIF" | awk '{print $1}')
expected_sha=$(awk '{print $1}' "$SIF.sha256")
[[ "$NEUROMRI_CONTAINER_SHA256" == "$expected_sha" ]] || { echo "SIF SHA256 mismatch: $SIF" >&2; exit 2; }
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 XFORMERS_DISABLED=1
export OMP_NUM_THREADS=${CPUS_PER_FIT:-4} MKL_NUM_THREADS=${CPUS_PER_FIT:-4}
# Explicit variables survive --cleanenv; --bind preserves arbitrary Vega paths.
export APPTAINERENV_NEUROMRI_DATA="$DATA_ROOT"
export APPTAINERENV_NEUROMRI_WEIGHTS="$WEIGHTS_ROOT"
export APPTAINERENV_NEUROMRI_CONTAINER_SHA256="$NEUROMRI_CONTAINER_SHA256"
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export APPTAINERENV_PYTHONHASHSEED=0
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_MKL_NUM_THREADS="$MKL_NUM_THREADS"
export APPTAINERENV_PYTHONPATH="$REPO"
CONTAINER=(apptainer exec --cleanenv --bind "$REPO:$REPO" --bind "$DATA_ROOT:$DATA_ROOT:ro" --bind "$WEIGHTS_ROOT:$WEIGHTS_ROOT:ro" --pwd "$REPO")

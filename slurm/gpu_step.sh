#!/usr/bin/env bash
# This executes INSIDE srun, after Slurm has assigned this step its GPU mask.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one GPU to this step}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
exec "$@"

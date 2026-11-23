#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/runtime.sh
trap 'rm -rf "/tmp/${USER}/nmri-${SLURM_JOB_ID}"' EXIT
srun bash slurm/gpu_step.sh "${CONTAINER[@]}" --nv "$SIF" python -m neuromri.run_row \
    --runlist "$RUNLIST" --row "$SLURM_ARRAY_TASK_ID"

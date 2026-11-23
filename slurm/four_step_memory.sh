#!/usr/bin/env bash
# Slurm reports SLURM_MEM_PER_NODE in MiB. Without --mem each srun inherits
# the entire allocation, serializing otherwise exclusive single-GPU steps.
[[ "${SLURM_MEM_PER_NODE:-}" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Four-step launch needs a finite, positive SLURM_MEM_PER_NODE (MiB)' >&2; exit 2;
}
GPU_STEP_MEM_MB=${GPU_STEP_MEM_MB:-$((SLURM_MEM_PER_NODE / 4))}
[[ "$GPU_STEP_MEM_MB" =~ ^[1-9][0-9]*$ ]] &&
    (( GPU_STEP_MEM_MB * 4 <= SLURM_MEM_PER_NODE )) || {
    echo 'GPU_STEP_MEM_MB must be positive and at most one quarter of job memory' >&2; exit 2;
}
echo "Four GPU steps: ${GPU_STEP_MEM_MB} MiB/step of ${SLURM_MEM_PER_NODE} MiB/job" >&2

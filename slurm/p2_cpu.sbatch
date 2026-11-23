#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/runtime.sh
trap 'rm -rf "/tmp/${USER}/nmri-${SLURM_JOB_ID}"' EXIT
srun "${CONTAINER[@]}" "$SIF" python -m neuromri.p2_threshold_transport \
    --set "jobs=$CPU_CPUS" "threads_per_fit=1" "output=${P2_OUTPUT:-results/p2}"

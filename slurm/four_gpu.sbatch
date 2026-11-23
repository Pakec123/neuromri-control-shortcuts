#!/usr/bin/env bash
# One array element consumes one complete A100 node and four adjacent run rows.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/runtime.sh
source slurm/four_step_memory.sh
trap 'rm -rf "/tmp/${USER}/nmri-${SLURM_JOB_ID}"' EXIT
base=$(( SLURM_ARRAY_TASK_ID * 4 ))
pids=()
for slot in 0 1 2 3; do
    row=$(( base + slot ))
    # Slurm assigns one visible GPU to each exclusive step; do not overwrite its mask.
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:1 --mem="$GPU_STEP_MEM_MB" \
        bash slurm/gpu_step.sh "${CONTAINER[@]}" --nv "$SIF" python -m neuromri.run_row \
        --runlist "$RUNLIST" --row "$row" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"

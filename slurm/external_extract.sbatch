#!/usr/bin/env bash
# Four independent encoders; one GPU per extraction, packed on one node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/external_runtime.sh
source slurm/four_step_memory.sh
: "${EXTERNAL_ROOT:?Set processed external PNG root in hpc.local.env}"
CONTAINER+=(--bind "$EXTERNAL_ROOT:$EXTERNAL_ROOT:ro")
encoders=(resnet18 resnet50 dinov2_s biomedclip)
pids=()
for name in "${encoders[@]}"; do
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:1 --mem="$GPU_STEP_MEM_MB" \
        bash slurm/external_encoder_step.sh "$name" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"

#!/usr/bin/env bash
# Four frozen image encoders, one per GPU, all results generated on Vega.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/runtime.sh
source slurm/four_step_memory.sh
trap 'rm -rf "/tmp/${USER}/nmri-${SLURM_JOB_ID}"' EXIT
encoders=(resnet18 resnet50 dinov2_s biomedclip)
pids=()
for name in "${encoders[@]}"; do
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:1 --mem="$GPU_STEP_MEM_MB" \
        bash slurm/gpu_step.sh "${CONTAINER[@]}" --nv "$SIF" python -m neuromri.extract_embeddings \
        --set "encoder=$name" "workers=4" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"

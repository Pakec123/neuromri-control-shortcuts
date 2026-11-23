#!/usr/bin/env bash
set -euo pipefail
source "${NEUROMRI_HPC_ENV:-slurm/hpc.env}"
if [[ -f slurm/hpc.local.env ]]; then source slurm/hpc.local.env; fi
: "${EXTERNAL_RUN_DIR:?Start with bash tools/run_external.sh}"
source "$EXTERNAL_RUN_DIR/run.env"
for key in ACCOUNT PARTITION CPU_PARTITION SIF DATA_ROOT EXTERNAL_ROOT; do
    value=${!key:-}
    [[ -n "$value" && "$value" != *'<TBD>'* ]] || { echo "Set $key in hpc.local.env" >&2; exit 2; }
done
[[ -f "$SIF.sha256" && -f "$EXTERNAL_SIF.sha256" ]] || { echo 'Prepare both SIF hashes'; exit 2; }
mkdir -p logs
opts=(--account="$ACCOUNT" --output='logs/%x-%j.out' --error='logs/%x-%j.err' --nodes=1 --kill-on-invalid-dep=yes)
[[ -z "$QOS" ]] || opts+=(--qos="$QOS")
pre=$(sbatch --parsable "${opts[@]}" --job-name=p2-ext-preprocess --time="${EXTERNAL_PREPROCESS_TIME:-24:00:00}" \
    --partition="$CPU_PARTITION" --ntasks=1 --cpus-per-task=4 --mem="${EXTERNAL_PREPROCESS_MEM:-32G}" \
    slurm/external_preprocess.sbatch)
pre=${pre%%;*}
gpu=$(sbatch --parsable "${opts[@]}" --job-name=p2-ext-extract --time="$TIME" \
    --dependency="afterok:$pre" \
    --partition="$PARTITION" --ntasks=4 --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:4 --mem="$MEM" \
    slurm/external_extract.sbatch)
gpu=${gpu%%;*}
cpu=$(sbatch --parsable "${opts[@]}" --job-name=p2-ext-probes --dependency="afterok:$gpu" \
    --time="$CPU_TIME" --partition="$CPU_PARTITION" --ntasks=1 --cpus-per-task="$CPU_CPUS" \
    --mem="$CPU_MEM" slurm/external_cpu.sbatch)
apptainer exec --cleanenv --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" \
    python tools/external_run_manifest.py submitted --run "$EXTERNAL_RUN_DIR" --jobs "$pre" "$gpu" "$cpu"
echo "Preprocessing=$pre -> embeddings=$gpu -> probes=$cpu; $EXTERNAL_RUN_DIR/RESULTS_MANIFEST.json"

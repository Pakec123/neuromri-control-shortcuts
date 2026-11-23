#!/usr/bin/env bash
# Submit from repository root. All scheduler policy values live in vega.env.
set -euo pipefail
source "${NEUROMRI_HPC_ENV:-slurm/hpc.env}"
if [[ -f slurm/hpc.local.env ]]; then source slurm/hpc.local.env; fi
[[ "$ACCOUNT" != '<TBD>' && "$DATA_ROOT" != '<TBD>/Brain Disease' ]] || { echo 'Set ACCOUNT/DATA_ROOT in slurm/hpc.local.env'; exit 2; }
mkdir -p logs
opts=(--account="$ACCOUNT" --time="$TIME" --output='logs/%x-%A_%a.out' --error='logs/%x-%A_%a.err')
[[ -z "$QOS" ]] || opts+=(--qos="$QOS")
mode=${1:-four}
if [[ "$mode" == cpu ]]; then
    [[ "$CPU_PARTITION" != '<TBD>' ]] || { echo 'Set CPU_PARTITION'; exit 2; }
    sbatch "${opts[@]}" --time="$CPU_TIME" --partition="$CPU_PARTITION" --cpus-per-task="$CPU_CPUS" \
        --mem="$CPU_MEM" --nodes=1 --ntasks=1 slurm/p2_cpu.sbatch
elif [[ "$mode" == extract ]]; then
    sbatch "${opts[@]}" --partition="$PARTITION" --nodes=1 --ntasks=4 \
        --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:4 --mem="$MEM" slurm/extract.sbatch
else
    # Header-only CSV count, no pandas dependency on the login host.
    n=$(python -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$RUNLIST")
    [[ "$n" -gt 0 ]] || exit 2
    if [[ "$mode" == four ]]; then
        last=$(( (n+3)/4 - 1 ))
        sbatch "${opts[@]}" --partition="$PARTITION" --nodes=1 --ntasks=4 \
            --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:4 --mem="$MEM" \
            --array="0-${last}%${MAX_ARRAY_CONCURRENT}" slurm/four_gpu.sbatch
    elif [[ "$mode" == array ]]; then
        sbatch "${opts[@]}" --partition="$PARTITION" --nodes=1 --ntasks=1 \
            --cpus-per-task="$CPUS_PER_FIT" --gres=gpu:1 --mem=32G \
            --array="0-$(( n-1 ))%${MAX_ARRAY_CONCURRENT}" slurm/array.sbatch
    else
        echo 'Usage: bash slurm/submit.sh {four|array|extract|cpu}'; exit 2
    fi
fi
